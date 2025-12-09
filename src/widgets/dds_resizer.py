import concurrent.futures
import fnmatch
import os
import shutil
import struct
import subprocess
import time
from typing import List, Tuple, Optional

from PySide6 import QtWidgets, QtCore
from PySide6.QtWidgets import QDialog, QVBoxLayout as QVBoxLayoutQt, QLabel, QDialogButtonBox
from qfluentwidgets import (
    FluentIcon as FIF,
    PrimaryPushButton,
    PushSettingCard,
    ConfigItem,
    PushButton,
    InfoBar,
)

from src.help.dds_help import DDSHelp
from src.settings.dds_settings import DDSSettings
from src.utils.appconfig import cfg, TEXCONV_EXE
from src.utils.cards import TextSettingCard
from src.utils.helpers import BaseWidget
from src.utils.logging_utils import logger
from src.utils import dds_utils
from PIL import Image


# --- Utility: read DDS dimensions ---
def read_dds_size(path: str) -> Tuple[int, int]:
    """Return (width, height) for a DDS file.
    Raises ValueError if not a DDS file or header too small.
    """
    with open(path, 'rb') as f:
        header = f.read(128)
    if len(header) < 128 or header[:4] != b'DDS ':
        raise ValueError('Not a DDS file')
    # DDS_HEADER structure: https://docs.microsoft.com/en-us/windows/win32/direct3ddds/dx-graphics-dds-pguide
    # DWORD size at offset 4 should be 124
    # height at +12 (offset 12), width at +16
    height = struct.unpack_from('<I', header, 12)[0]
    width = struct.unpack_from('<I', header, 16)[0]
    return width, height

# --- Utility: read DDS DXGI format if DX10 header present ---
DXGI_FORMAT_BC7_UNORM = 98
DXGI_FORMAT_BC7_UNORM_SRGB = 99

def read_dds_dxgi_format(path: str) -> Optional[int]:
    """Return DXGI format integer if DDS has DX10 header, else None.
    Only DX10+ DDS can be BC7. Safe to call on any DDS file path.
    """
    try:
        with open(path, 'rb') as f:
            header = f.read(148)  # 128 + at least 20 for DX10
        if len(header) < 128 or header[:4] != b'DDS ':
            return None
        # DDS_PIXELFORMAT starts at offset 76, dwFourCC at offset 84
        if len(header) < 128:
            return None
        fourcc = header[84:88]
        if fourcc != b'DX10':
            return None
        if len(header) < 148:
            # need DX10 struct (20 bytes)
            with open(path, 'rb') as f:
                header = f.read(148)
            if len(header) < 148:
                return None
        dxgi_fmt = struct.unpack_from('<I', header, 128)[0]
        return int(dxgi_fmt)
    except Exception:
        return None


def is_bc7_dxgi(dxgi: Optional[int]) -> bool:
    return dxgi in (DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_BC7_UNORM_SRGB)


def is_bc7_linear(dxgi: Optional[int]) -> bool:
    return dxgi == DXGI_FORMAT_BC7_UNORM


class Worker(QtCore.QThread):
    progress = QtCore.Signal(int, int, str)  # processed, total, message
    finished = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(self, src_dir: str, sizes: List[int], out_root: str,
                 per_size_subfolders: bool = True, no_upscale: bool = True,
                 generate_mips: bool = True, threads: int | None = None, ignore_patterns: Optional[List[str]] = None,
                 convert_to_srgb: bool = False, downscale_method: str = "texconv", parent=None):
        super().__init__(parent)
        self.src_dir = src_dir
        self.sizes = sorted(set([int(s) for s in sizes if s > 0]))
        self.out_root = out_root
        self.per_size_subfolders = per_size_subfolders
        self.no_upscale = no_upscale
        self.generate_mips = generate_mips
        self.convert_to_srgb = convert_to_srgb
        self.downscale_method = (downscale_method or "texconv").lower()
        self.threads = max(1, int(threads or (os.cpu_count() or 1)))
        # Normalize ignore patterns: use forward slashes for matching and strip whitespace
        self.ignore_patterns = []
        for p in (ignore_patterns or []):
            p = str(p).strip().replace('\\', '/').lstrip('./')
            if p:
                self.ignore_patterns.append(p)
        self._abort = False

    def _match_ignored(self, rel_dir: str) -> bool:
        """Return True if rel_dir (relative path) matches any ignore pattern.
        Matching is done on normalized forward-slash paths and on the basename.
        """
        if not self.ignore_patterns:
            return False
        rel_dir = (rel_dir or '').strip()
        if not rel_dir or rel_dir == '.':
            return False
        norm = rel_dir.replace('\\', '/').strip('/')
        base = os.path.basename(norm)
        for pat in self.ignore_patterns:
            try:
                if fnmatch.fnmatchcase(norm, pat) or fnmatch.fnmatchcase(base, pat):
                    return True
            except Exception:
                # If a bad pattern slips in, ignore it
                pass
        return False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            # Gather DDS files with ignore patterns and prune output roots if inside src
            dds_files: List[str] = []
            # Compute skip roots if out_root resides under src_dir
            try:
                src_abs = os.path.abspath(self.src_dir)
                out_abs = os.path.abspath(self.out_root)
            except Exception:
                src_abs = self.src_dir
                out_abs = self.out_root
            skip_rel_roots: list[str] = []
            if out_abs and src_abs and os.path.normcase(out_abs).startswith(os.path.normcase(src_abs)):
                out_rel = os.path.relpath(out_abs, src_abs).replace('\\', '/').strip('/')
                # Normalize: if output equals source, out_rel becomes '.'; treat as empty
                out_rel_norm = '' if out_rel in ('', '.') else out_rel
                if self.per_size_subfolders and self.sizes:
                    # When output equals source, only skip the per-size folders at the root (e.g., '512')
                    if out_rel_norm:
                        skip_rel_roots = [f"{out_rel_norm}/{size}" for size in self.sizes]
                    else:
                        skip_rel_roots = [str(size) for size in self.sizes]
                else:
                    # Only skip the explicit output subfolder if it exists under source; do not skip '.'
                    skip_rel_roots = [out_rel_norm] if out_rel_norm else []
            # Guard: remove empties to avoid skipping the entire tree
            skip_rel_roots = [sk for sk in skip_rel_roots if sk and sk != '.']

            def _should_skip_dir(rel_child: str) -> bool:
                rel_child = (rel_child or '').replace('\\', '/').strip('/')
                if any(rel_child == sk or rel_child.startswith(sk + '/') for sk in skip_rel_roots):
                    return True
                return self._match_ignored(rel_child)

            for root, dirs, files in os.walk(self.src_dir):
                rel_root = os.path.relpath(root, self.src_dir)
                if _should_skip_dir(rel_root):
                    dirs[:] = []
                    continue
                # Prune ignored/out subdirectories in-place
                dirs[:] = [d for d in dirs if not _should_skip_dir(os.path.join(rel_root, d))]
                for fn in files:
                    if fn.lower().endswith('.dds'):
                        dds_files.append(os.path.join(root, fn))

            # Detect BC7 textures for conversion during resize (if enabled)
            bc7_sources: set[str] = set()
            if self.convert_to_srgb:
                for src_path in dds_files:
                    dxgi = read_dds_dxgi_format(src_path)
                    if is_bc7_dxgi(dxgi):
                        bc7_sources.add(src_path)

            total = len(dds_files) * len(self.sizes)
            if total == 0:
                self.finished.emit('No DDS files found.')
                return

            # Ensure top-level destination roots exist
            for size in self.sizes:
                dest_root = os.path.join(self.out_root, str(size)) if self.per_size_subfolders else self.out_root
                os.makedirs(dest_root, exist_ok=True)

            processed = 0
            processed_lock = QtCore.QMutex()

            def _pillow_resize_and_save(src_path: str, out_path: str, size: int, needs_bc7_conversion: bool,
                                        orig_w: int, orig_h: int) -> str:
                # Determine target format for texconv after PNG save
                target_fmt = 'BC3_UNORM' if needs_bc7_conversion else None
                # Compute new dimensions preserving aspect ratio
                if orig_w <= 0 or orig_h <= 0:
                    try:
                        im_probe = dds_utils.load_image(src_path, f='RGBA')
                        orig_w_local, orig_h_local = im_probe.width, im_probe.height
                        try:
                            im_probe.close()
                        except Exception:
                            pass
                    except Exception:
                        orig_w_local, orig_h_local = 0, 0
                else:
                    orig_w_local, orig_h_local = orig_w, orig_h

                new_w, new_h = orig_w_local, orig_h_local
                if orig_w_local and orig_h_local:
                    scale = size / max(orig_w_local, orig_h_local)
                    if self.no_upscale and scale >= 1.0 and not needs_bc7_conversion:
                        # Just copy if allowed
                        try:
                            shutil.copy2(src_path, out_path)
                            return f'Copied (no upscale): {os.path.relpath(src_path, self.src_dir)} -> {os.path.relpath(out_path, self.out_root)}'
                        except Exception as e:
                            return f'ERROR copying {src_path}: {e}'
                    if scale < 1.0:
                        new_w = max(1, int(round(orig_w_local * scale)))
                        new_h = max(1, int(round(orig_h_local * scale)))
                # Choose resampling filter
                method = self.downscale_method
                resample_map = {
                    'nearest': Image.Resampling.NEAREST,
                    'bilinear': Image.Resampling.BILINEAR,
                    'bicubic': Image.Resampling.BICUBIC,
                    'lanczos': Image.Resampling.LANCZOS,
                    'box': Image.Resampling.BOX,
                    'hamming': Image.Resampling.HAMMING,
                }
                resample = resample_map.get(method, Image.Resampling.LANCZOS)

                # Load, resize (if needed), save to temporary PNG then convert via texconv
                tmp_png = dds_utils.add_temp_to_filename(out_path)
                try:
                    im = dds_utils.load_image(src_path, f='RGBA')
                    if new_w != im.width or new_h != im.height:
                        im = im.resize((new_w, new_h), resample=resample)
                    im.save(tmp_png)
                    dds_utils.convert_to_dds(tmp_png, out_path, is_palette=False,
                                             generate_mips=self.generate_mips,
                                             target_format=target_fmt)
                    return f'Resized to {size}: {os.path.relpath(src_path, self.src_dir)}'
                except Exception as e:
                    return f'ERROR Pillow resize for {os.path.relpath(src_path, self.src_dir)}: {e}'
                finally:
                    try:
                        im.close()
                    except Exception:
                        pass
                    try:
                        if os.path.exists(tmp_png):
                            os.remove(tmp_png)
                    except Exception:
                        pass

            def task(size: int, src_path: str) -> str:
                if self._abort:
                    return 'aborted'
                rel_path = os.path.relpath(src_path, self.src_dir)
                rel_dir = os.path.dirname(rel_path)
                dest_root = os.path.join(self.out_root, str(size)) if self.per_size_subfolders else self.out_root
                out_dir = os.path.join(dest_root, rel_dir)
                try:
                    os.makedirs(out_dir, exist_ok=True)
                except Exception:
                    pass
                out_path = os.path.join(out_dir, os.path.basename(src_path))

                # Decide whether to copy or convert
                w = h = 0
                try:
                    w, h = read_dds_size(src_path)
                except Exception:
                    pass
                max_dim = max(w, h)

                # Check if BC7 conversion is needed
                needs_bc7_conversion = self.convert_to_srgb and src_path in bc7_sources

                if self.downscale_method != 'texconv':
                    return _pillow_resize_and_save(src_path, out_path, size, needs_bc7_conversion, w, h)

                if self.no_upscale and max_dim and max_dim <= size and not needs_bc7_conversion:
                    if self._abort:
                        return 'aborted'
                    try:
                        shutil.copy2(src_path, out_path)
                        return f'Copied (no upscale): {rel_path} -> {os.path.relpath(out_path, self.out_root)}'
                    except Exception as e:
                        pass

                cmd = [
                    TEXCONV_EXE,
                    '-nologo',
                    '-y',
                    '-o', out_dir,
                    '-w', str(size),
                    '-h', str(size),
                ]
                # Convert BC7 to BC3 linear if conversion is enabled
                try:
                    if self.convert_to_srgb and src_path in bc7_sources:
                        cmd.extend(['-f', 'BC3_UNORM'])
                except Exception:
                    pass
                cmd.append(src_path)
                if not self.generate_mips:
                    cmd.insert(-1, '-m')
                    cmd.insert(-1, '1')
                if self._abort:
                    return 'aborted'
                try:
                    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=False)
                    msg = proc.stdout.strip()
                    if proc.returncode != 0:
                        return f'ERROR ({proc.returncode}) {rel_path}: {msg}'
                    else:
                        return f'Resized to {size}: {rel_path}'
                except Exception as e:
                    return f'ERROR running texconv for {rel_path}: {e}'

            # Submit tasks
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as ex:
                for size in self.sizes:
                    for src_path in dds_files:
                        futures.append(ex.submit(task, size, src_path))
                for fut in concurrent.futures.as_completed(futures):
                    if self._abort:
                        # Drain remaining futures quickly
                        break
                    try:
                        message = fut.result()
                    except Exception as e:
                        message = f'ERROR: {e}'
                    # Update processed count safely
                    processed_lock.lock()
                    processed += 1
                    p = processed
                    processed_lock.unlock()
                    if message == 'aborted':
                        self.progress.emit(p, total, 'Aborted.')
                    else:
                        self.progress.emit(p, total, message)

            if self._abort:
                self.finished.emit('Aborted by user.')
                return

            self.finished.emit('Done.')
        except Exception as e:
            self.error.emit(str(e))


class DDSResizerWindow(BaseWidget):
    def __init__(self, parent=None, text=None):
        super().__init__(parent=parent, text=text, vertical=True)

        # Runtime tracking
        self._start_ts: float | None = None
        self._last_processed: int = 0
        self._last_total: int = 0
        self._run_ctx: dict = {}

        # Persistent settings
        self.src_cfg = ConfigItem("dds_resizer", "src", "")
        self.out_cfg = ConfigItem("dds_resizer", "out", "")

        # --- Cards (Flattened, all on same level) ---
        self.src_card = PushSettingCard(
            self.tr("Source folder"),
            FIF.FOLDER,
            self.tr("DDS files Location"),
            self.src_cfg.value or ""
        )
        self.out_card = PushSettingCard(
            self.tr("Output folder"),
            FIF.FOLDER,
            self.tr("Output Location"),
            self.out_cfg.value or ""
        )
        self.sizes_card = TextSettingCard(
            cfg.sizes_cfg,
            FIF.EDIT,
            self.tr("Target sizes (comma seperated)"),
            self.tr("e.g., 512,1024,2048")
        )
        self.ignore_card = TextSettingCard(
            cfg.ignore_cfg,
            FIF.FILTER,
            self.tr("Ignore subfolders (comma seperated, supports wildcards)"),
            cfg.ignore_cfg.value or self.tr("e.g., temp, *_bak, */Generated/*")
        )


        # Add all settings (including switches) to layout
        self.addToFrame(self.src_card)
        self.addToFrame(self.out_card)
        self.addToFrame(self.sizes_card)
        self.addToFrame(self.ignore_card)
        self.boxLayout.addStretch(1)

        # --- Bottom Controls ---
        self.run_button = PrimaryPushButton(icon=FIF.RIGHT_ARROW, text=self.tr("Run"))
        self.stop_button = PushButton(icon=FIF.CANCEL, text=self.tr("Stop"))
        self.stop_button.setEnabled(False)

        # --- Wire up ---
        self.src_card.clicked.connect(self.on_input_dir_card)
        self.out_card.clicked.connect(self.on_output_root_card)
        self.run_button.clicked.connect(self.on_run)
        self.stop_button.clicked.connect(self.stop)


        self.buttons_layout.addWidget(self.stop_button, stretch=1)
        self.addButtonBarToBottom(self.run_button)

        self.settings_widget = DDSSettings(self)
        self.settings_drawer.addWidget(self.settings_widget)

        self.help_widget = DDSHelp(self)
        self.help_drawer.addWidget(self.help_widget)


    def _parse_sizes(self, csv: str) -> list[int]:
        out: list[int] = []
        for tok in (csv or "").replace(';', ',').split(','):
            tok = tok.strip()
            if not tok:
                continue
            if tok.isdigit():
                n = int(tok)
                if 1 <= n <= 16384 and n not in out:
                    out.append(n)
        return sorted(out)

    def on_input_dir_card(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr("Select source folder"), self.src_cfg.value or os.getcwd())
        if directory:
            self.src_cfg.value = directory
            self.src_card.setContent(directory)
            # Suggest output if empty
            if not (self.out_cfg.value or '').strip():
                base = os.path.basename(os.path.normpath(directory))
                suggested = os.path.join(os.path.dirname(directory), base + '_resized')
                self.out_cfg.value = suggested
                self.out_card.setContent(suggested)

    def on_output_root_card(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr("Select output folder"), self.out_cfg.value or os.getcwd())
        if directory:
            self.out_cfg.value = directory
            self.out_card.setContent(directory)

    def set_running(self, running: bool):
        self.run_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.src_card.setEnabled(not running)
        self.out_card.setEnabled(not running)
        self.sizes_card.setEnabled(not running)
        self.ignore_card.setEnabled(not running)

    # --- Dialog helpers ---
    def _format_elapsed(self) -> str:
        try:
            if not self._start_ts:
                return "-"
            secs = max(0.0, time.time() - self._start_ts)
            m, s = divmod(int(secs), 60)
            h, m = divmod(m, 60)
            if h:
                return f"{h:d}h {m:d}m {s:d}s"
            if m:
                return f"{m:d}m {s:d}s"
            return f"{s:d}s"
        except Exception:
            return "-"

    def _show_info_dialog(self, title: str, lines: list[str], offer_open: bool = False):
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        layout = QVBoxLayoutQt(dlg)
        lbl = QLabel("<br/>".join([QtWidgets.QApplication.translate("DDSResizerWindow", l) if isinstance(l, str) else str(l) for l in lines]))
        lbl.setTextFormat(QtCore.Qt.RichText)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        open_btn = None
        if offer_open:
            open_btn = buttons.addButton(self.tr("Open output folder"), QDialogButtonBox.ActionRole)
        layout.addWidget(buttons)

        def on_open():
            try:
                out = (self._run_ctx.get('out') if isinstance(self._run_ctx, dict) else None) or self.out_cfg.value
                if out and os.path.isdir(out):
                    os.startfile(out)
            except Exception:
                pass

        if open_btn is not None:
            open_btn.clicked.connect(on_open)
        buttons.accepted.connect(dlg.accept)
        dlg.exec()

    def on_run(self):
        src = (self.src_cfg.value or '').strip()
        out = (self.out_cfg.value or '').strip()
        sizes_csv = (cfg.sizes_cfg.value or '').strip()
        ignore_csv = (cfg.ignore_cfg.value or '').strip()
        per_size = bool(cfg.per_size_cfg.value)
        no_upscale = bool(cfg.no_upscale_cfg.value)
        gen_mips = bool(cfg.mips_cfg.value)
        to_bc3 = bool(cfg.bc3_cfg.value)
        threads = cfg.get(cfg.threads_cfg)
        method = (cfg.dds_downscale_method.value if hasattr(cfg, 'dds_downscale_method') else 'texconv') or 'texconv'
        sizes = self._parse_sizes(sizes_csv)

        if not src or not os.path.isdir(src):
            InfoBar.warning(
                title=self.tr("Validation"),
                content=self.tr("Please select a valid source folder."),
                duration=3000,
                parent=self,
            )
            return
        if not out:
            InfoBar.warning(
                title=self.tr("Validation"),
                content=self.tr("Please choose an output folder."),
                duration=3000,
                parent=self,
            )
            return
        if not sizes:
            InfoBar.warning(
                title=self.tr("Validation"),
                content=self.tr("Please enter at least one valid size (CSV)."),
                duration=3000,
                parent=self,
            )
            return
        if not per_size and len(sizes) > 1:
            InfoBar.warning(
                title=self.tr("Validation"),
                content=self.tr("When not creating per-size subfolders, provide only one size to avoid overwriting outputs."),
                duration=4000,
                parent=self,
            )
            return

        # Ensure output exists
        try:
            os.makedirs(out, exist_ok=True)
        except Exception:
            pass

        # Parse ignore patterns
        ignore_patterns = [p.strip() for p in ignore_csv.split(',') if p.strip()] if ignore_csv else []

        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'show_progress'):
            try:
                p.show_progress()
            except Exception:
                pass
        logger.debug('Starting...')
        self.set_running(True)

        # Track run context for the completion dialog
        self._start_ts = time.time()
        self._last_processed = 0
        self._last_total = 0
        self._run_ctx = {
            'src': src,
            'out': out,
            'sizes': sizes,
            'per_size': per_size,
            'no_upscale': no_upscale,
            'gen_mips': gen_mips,
            'to_bc3': to_bc3,
            'threads': threads,
            'ignore_patterns': ignore_patterns,
            'method': method,
        }

        self.worker = Worker(
            src_dir=src,
            sizes=sizes,
            out_root=out,
            per_size_subfolders=per_size,
            no_upscale=no_upscale,
            generate_mips=gen_mips,
            threads=threads,
            ignore_patterns=ignore_patterns,
            convert_to_srgb=to_bc3,
            downscale_method=method,
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def stop(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.abort()
            logger.debug('Stopping...')

    @QtCore.Slot(int, int, str)
    def on_progress(self, processed: int, total: int, message: str):
        # Track latest counts for the completion dialog
        self._last_processed = processed
        self._last_total = total
        # Forward a percentage update to parent mask
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'update_progress') and total:
            try:
                percent = int(max(0, min(100, round((processed / total) * 100))))
                p.update_progress(percent)
            except Exception:
                pass
        logger.debug(message)

    @QtCore.Slot(str)
    def on_finished(self, message: str):
        logger.debug(message)
        self.set_running(False)
        # Build completion info dialog
        processed = self._last_processed or 0
        total = self._last_total or 0
        ctx = self._run_ctx if isinstance(self._run_ctx, dict) else {}
        sizes = ctx.get('sizes') or []
        per_size = ctx.get('per_size')
        no_upscale = ctx.get('no_upscale')
        gen_mips = ctx.get('gen_mips')
        to_bc3 = ctx.get('to_bc3')
        threads = ctx.get('threads') or os.cpu_count()
        src = ctx.get('src') or (self.src_cfg.value or '')
        out = ctx.get('out') or (self.out_cfg.value or '')
        ignore = ctx.get('ignore_patterns') or []
        elapsed = self._format_elapsed()

        status_line = f"Status: {message}"
        count_line = f"Processed: {processed} / {total}"
        elapsed_line = f"Elapsed: {elapsed}"
        sizes_line = f"Target sizes: {', '.join(str(s) for s in sizes) if sizes else '-'}"
        opts_line = (
            f"Options: per-size folders = {per_size}, no upscale = {no_upscale}, generate mips = {gen_mips}, convert BC7→BC3 = {to_bc3}"
        )
        method_line = f"Downscale method: {ctx.get('method') or 'texconv'}"
        threads_line = f"Threads: {threads}"
        src_line = f"Source: {src}"
        out_line = f"Output: {out}"
        ignore_line = f"Ignored: {', '.join(ignore) if ignore else '-'}"
        hint_line = "Hint: You can run again or open the output folder to review results."

        lines = [
            f"<b>{status_line}</b>",
            count_line,
            elapsed_line,
            sizes_line,
            opts_line,
            method_line,
            threads_line,
            src_line,
            out_line,
            ignore_line,
            "",
            hint_line,
        ]
        self._show_info_dialog(self.tr("Resize Completed"), lines, offer_open=True)
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'complete_loader'):
            try:
                p.complete_loader()
            except Exception:
                pass

    @QtCore.Slot(str)
    def on_error(self, message: str):
        logger.debug('ERROR: ' + message)
        self.set_running(False)
        ctx = self._run_ctx if isinstance(self._run_ctx, dict) else {}
        src = ctx.get('src') or (self.src_cfg.value or '')
        out = ctx.get('out') or (self.out_cfg.value or '')
        lines = [
            f"<b>{self.tr('An error occurred')}</b>",
            f"Message: {message}",
            f"Source: {src}",
            f"Output: {out}",
        ]
        self._show_info_dialog(self.tr("Resize Error"), lines, offer_open=False)
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'complete_loader'):
            try:
                p.complete_loader()
            except Exception:
                pass



