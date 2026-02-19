import concurrent.futures
import fnmatch
import os
from typing import List

from PIL import Image, ImageOps
from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QDialog, QVBoxLayout as QVBoxLayoutQt, QLabel, QDialogButtonBox
from qfluentwidgets import (
    FluentIcon as FIF,
    PrimaryPushButton,
    PushSettingCard,
    ConfigItem,
    PushButton,
    InfoBar,
    SwitchSettingCard,
)

from src.utils import dds_utils
from src.utils.cards import TextSettingCard
from src.utils.helpers import BaseWidget
from src.utils.logging_utils import logger


class ExportWorker(QtCore.QThread):
    progress = QtCore.Signal(int, int, str)  # processed, total, message
    finished = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(self, src_dir: str, out_root: str, ignore_patterns: List[str] | None = None,
                 fallout76_mode: bool = False, fo4_metallic_grayscale: bool = False, parent=None):
        super().__init__(parent)
        self.src_dir = src_dir
        self.out_root = out_root
        self.ignore_patterns = [p.strip().replace('\\', '/').lstrip('./') for p in (ignore_patterns or []) if p.strip()]
        self.fallout76_mode = bool(fallout76_mode)
        self.fo4_metallic_grayscale = bool(fo4_metallic_grayscale)
        self._abort = False

    def abort(self):
        self._abort = True

    def _match_ignored(self, rel_dir: str) -> bool:
        if not self.ignore_patterns:
            return False
        norm = (rel_dir or '').replace('\\', '/').strip('/')
        base = os.path.basename(norm)
        for pat in self.ignore_patterns:
            try:
                if fnmatch.fnmatchcase(norm, pat) or fnmatch.fnmatchcase(base, pat):
                    return True
            except Exception:
                pass
        return False

    def _save_png(self, img: Image.Image, out_path: str) -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.save(out_path)

    @staticmethod
    def _sanitize_stem(name: str) -> str:
        """Strip a trailing .png (case-insensitive) from a base name to avoid .png.png outputs.

        Example: "M.png" -> "M"
        """
        if name.lower().endswith('.png'):
            return name[:-4]
        return name

    def _export_spec_variants(self, img: Image.Image, base_out_dir: str, stem_no_suffix: str) -> list[str]:
        messages: list[str] = []
        try:
            rgb = img.convert('RGB')
            r, g, _ = rgb.split()

            # Spec/Gloss workflow
            specular = r  # R channel
            glossiness = g  # G channel (do NOT invert)

            # Metal/Rough workflow
            roughness = ImageOps.invert(g)  # invert G
            # Metallic: threshold (default FO4) or grayscale from R based on toggle
            if self.fo4_metallic_grayscale:
                metallic = r  # full-range metallic from R
            else:
                metallic = r.point(lambda p: 255 if p >= 128 else 0)

            out_spec = os.path.join(base_out_dir, f"{stem_no_suffix}_specular.png")
            out_gloss = os.path.join(base_out_dir, f"{stem_no_suffix}_glossiness.png")
            out_rough = os.path.join(base_out_dir, f"{stem_no_suffix}_roughness.png")
            out_metal = os.path.join(base_out_dir, f"{stem_no_suffix}_metallic.png")

            self._save_png(specular, out_spec)
            self._save_png(glossiness, out_gloss)
            self._save_png(roughness, out_rough)
            self._save_png(metallic, out_metal)

            messages.extend([
                f"Saved {os.path.relpath(out_spec, self.out_root)}",
                f"Saved {os.path.relpath(out_gloss, self.out_root)}",
                f"Saved {os.path.relpath(out_rough, self.out_root)}",
                f"Saved {os.path.relpath(out_metal, self.out_root)}",
            ])
        except Exception as e:
            logger.error(f"Failed creating spec variants: {e}")
            messages.append(f"ERROR spec variants: {e}")
        return messages

    def _export_fo76_l_variants(self, img: Image.Image, base_out_dir: str, stem_no_suffix: str) -> list[str]:
        """FO76 lighting map derivatives:
        - glossiness: R channel (smoothness)
        - roughness: invert(R)
        - ao: G channel
        - subsurface: B channel (optional)
        - emissive: A channel if present
        """
        messages: list[str] = []
        try:
            rgba = img.convert('RGBA')
            r, g, b, a = rgba.split()

            glossiness = r
            roughness = ImageOps.invert(r)
            ao = g
            subsurface = b
            emissive = a

            out_gloss = os.path.join(base_out_dir, f"{stem_no_suffix}_glossiness.png")
            out_rough = os.path.join(base_out_dir, f"{stem_no_suffix}_roughness.png")
            out_ao = os.path.join(base_out_dir, f"{stem_no_suffix}_ao.png")
            out_sss = os.path.join(base_out_dir, f"{stem_no_suffix}_subsurface.png")
            out_emis = os.path.join(base_out_dir, f"{stem_no_suffix}_emissive.png")

            self._save_png(glossiness, out_gloss)
            self._save_png(roughness, out_rough)
            self._save_png(ao, out_ao)
            # Subsurface and emissive may be empty; still save as provided (no resizing/filtering)
            self._save_png(subsurface, out_sss)
            self._save_png(emissive, out_emis)

            messages.extend([
                f"Saved {os.path.relpath(out_gloss, self.out_root)}",
                f"Saved {os.path.relpath(out_rough, self.out_root)}",
                f"Saved {os.path.relpath(out_ao, self.out_root)}",
                f"Saved {os.path.relpath(out_sss, self.out_root)}",
                f"Saved {os.path.relpath(out_emis, self.out_root)}",
            ])
        except Exception as e:
            logger.error(f"Failed creating FO76 lighting derivatives: {e}")
            messages.append(f"ERROR FO76 L variants: {e}")
        return messages

    def _export_fo76_spec_from_r(self, img: Image.Image, base_out_dir: str, stem_no_suffix: str) -> list[str]:
        """FO76 reflectance map → Spec/Gloss specular (RGB from grayscale)."""
        messages: list[str] = []
        try:
            gray = img.convert('L')
            spec_rgb = Image.merge('RGB', (gray, gray, gray))
            out_spec = os.path.join(base_out_dir, f"{stem_no_suffix}_specular.png")
            self._save_png(spec_rgb, out_spec)
            messages.append(f"Saved {os.path.relpath(out_spec, self.out_root)}")
        except Exception as e:
            logger.error(f"Failed creating FO76 specular from _r: {e}")
            messages.append(f"ERROR FO76 _r → specular: {e}")
        return messages

    def run(self):
        try:
            dds_files: list[str] = []
            for root, dirs, files in os.walk(self.src_dir):
                rel_root = os.path.relpath(root, self.src_dir)
                if self._match_ignored(rel_root):
                    dirs[:] = []
                    continue
                dirs[:] = [d for d in dirs if not self._match_ignored(os.path.join(rel_root, d))]
                for fn in files:
                    if fn.lower().endswith('.dds'):
                        dds_files.append(os.path.join(root, fn))

            total = len(dds_files)
            if total == 0:
                self.finished.emit('No DDS files found.')
                return

            # Thread-safe progress accounting
            processed = 0
            processed_lock = QtCore.QMutex()

            def task(src_path: str) -> str:
                if self._abort:
                    return 'aborted'
                rel_path = os.path.relpath(src_path, self.src_dir)
                rel_dir = os.path.dirname(rel_path)
                base_name = os.path.splitext(os.path.basename(src_path))[0]
                out_dir = os.path.join(self.out_root, rel_dir)

                # Straight PNG export (no resize/filtering/gamma changes)
                try:
                    img = dds_utils.load_image(src_path, f='RGBA')

                    # Normalize base name to avoid double extensions like .png.png
                    safe_base = self._sanitize_stem(base_name)

                    # If this is a normal map, ensure blue channel is white when needed
                    lower = os.path.basename(src_path).lower()
                    if lower.endswith('_n.dds'):
                        try:
                            r, g, b, a = img.split()
                            # Check blue channel statistics using getextrema (fast)
                            b_min, b_max = b.getextrema()
                            from PIL import Image as _PILImage  # local import to avoid top-level clutter
                            # Case 1: entirely black blue channel → force to white for all games
                            if b_min == 0 and b_max == 0:
                                white_b = _PILImage.new('L', b.size, 255)
                                img = Image.merge('RGBA', (r, g, white_b, a))
                            # Case 2 (FO76 specific): many FO76 normals decode with a flat grey B channel.
                            # If FO76 mode is enabled and B is uniform (min==max) but not already 255, promote to 255.
                            elif self.fallout76_mode and b_min == b_max and b_max < 255:
                                white_b = _PILImage.new('L', b.size, 255)
                                img = Image.merge('RGBA', (r, g, white_b, a))
                        except Exception:
                            # Be permissive: if anything goes wrong, fall back to original img
                            pass

                    out_png = os.path.join(out_dir, safe_base + '.png')
                    self._save_png(img, out_png)
                    msg_local = f"Saved {os.path.relpath(out_png, self.out_root)}"
                except Exception as e:
                    logger.error(f"Failed converting '{src_path}' to PNG: {e}")
                    return f"ERROR {rel_path}: {e}"
                finally:
                    try:
                        img.close()
                    except Exception:
                        pass

                # Derivatives
                lower = os.path.basename(src_path).lower()
                # Fallout 4 specular derivatives
                if lower.endswith('_s.dds'):
                    try:
                        img2 = dds_utils.load_image(src_path, f='RGBA')
                        # Also sanitize any trailing .png in stem
                        core = base_name[:-2] if base_name.lower().endswith('_s') else base_name
                        stem_no_suffix = self._sanitize_stem(core)
                        messages = self._export_spec_variants(img2, out_dir, stem_no_suffix)
                        if messages:
                            msg_local += " | " + " | ".join(messages)
                    except Exception as e:
                        logger.error(f"Failed spec variants for '{src_path}': {e}")
                        msg_local += f" | ERROR spec variants: {e}"
                    finally:
                        try:
                            img2.close()
                        except Exception:
                            pass

                # Fallout 76 specific handling
                if self.fallout76_mode:
                    try:
                        if lower.endswith('_l.dds'):
                            img3 = dds_utils.load_image(src_path, f='RGBA')
                            core = base_name[:-2] if base_name.lower().endswith('_l') else base_name
                            stem = self._sanitize_stem(core)
                            messages = self._export_fo76_l_variants(img3, out_dir, stem)
                            if messages:
                                msg_local += " | " + " | ".join(messages)
                        elif lower.endswith('_r.dds'):
                            img4 = dds_utils.load_image(src_path, f='L')
                            core = base_name[:-2] if base_name.lower().endswith('_r') else base_name
                            stem = self._sanitize_stem(core)
                            messages = self._export_fo76_spec_from_r(img4, out_dir, stem)
                            if messages:
                                msg_local += " | " + " | ".join(messages)
                    except Exception as e:
                        logger.error(f"Failed FO76 derivatives for '{src_path}': {e}")
                        msg_local += f" | ERROR FO76: {e}"
                    finally:
                        try:
                            if 'img3' in locals():
                                img3.close()
                        except Exception:
                            pass
                        try:
                            if 'img4' in locals():
                                img4.close()
                        except Exception:
                            pass

                return msg_local

            futures = []
            max_workers = max(1, os.cpu_count() or 1)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                for src_path in dds_files:
                    futures.append(ex.submit(task, src_path))
                for fut in concurrent.futures.as_completed(futures):
                    if self._abort:
                        # Stop processing additional results
                        break
                    try:
                        message = fut.result()
                    except Exception as e:
                        message = f"ERROR: {e}"
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


class DDSPNGExporterWindow(BaseWidget):
    def __init__(self, parent=None, text=None):
        super().__init__(parent=parent, text=text, vertical=True)

        # Persistent settings
        self.src_cfg = ConfigItem("dds_png_exporter", "src", "")
        self.out_cfg = ConfigItem("dds_png_exporter", "out", "")
        self.fo76_cfg = ConfigItem("dds_png_exporter", "fallout76", False)
        self.fo4_metal_gray_cfg = ConfigItem("dds_png_exporter", "fo4_metallic_grayscale", False)

        # UI cards
        self.src_card = PushSettingCard(
            self.tr("Source folder"),
            FIF.FOLDER,
            self.tr("DDS files Location"),
            self.src_cfg.value or ""
        )
        self.out_card = PushSettingCard(
            self.tr("Output folder"),
            FIF.FOLDER,
            self.tr("PNG Output Location"),
            self.out_cfg.value or ""
        )
        # Toggle: Fallout 76 mode (Fallout 4 is default OFF)
        self.fo76_card = SwitchSettingCard(
            icon=FIF.GAME,
            title=self.tr("Fallout 76 mode"),
            configItem=self.fo76_cfg,
            content=self.tr("Use Fallout 76 texture rules (derive roughness/glossiness/AO/emissive from _l, specular from _r)"),
        )
        # Toggle: Fallout 4 metallic policy
        self.fo4_metal_gray_card = SwitchSettingCard(
            icon=FIF.PALETTE,
            title=self.tr("Fallout 4: Grayscale Metallic"),
            configItem=self.fo4_metal_gray_cfg,
            content=self.tr("When processing _s.dds: use full-range metallic from R instead of binary (>=128 → 1, else 0)"),
        )
        self.ignore_cfg = ConfigItem("dds_png_exporter", "ignore", "")
        self.ignore_card = TextSettingCard(
            self.ignore_cfg,
            FIF.FILTER,
            self.tr("Ignore subfolders (comma separated, supports wildcards)"),
            self.tr("e.g., temp, *_bak, */Generated/*")
        )

        self.addToFrame(self.src_card)
        self.addToFrame(self.out_card)
        self.addToFrame(self.fo76_card)
        self.addToFrame(self.fo4_metal_gray_card)
        self.addToFrame(self.ignore_card)
        self.boxLayout.addStretch(1)

        self.run_button = PrimaryPushButton(icon=FIF.RIGHT_ARROW, text=self.tr("Run"))
        self.stop_button = PushButton(icon=FIF.CANCEL, text=self.tr("Stop"))
        self.stop_button.setEnabled(False)
        self.src_card.clicked.connect(self.on_input_dir_card)
        self.out_card.clicked.connect(self.on_output_root_card)
        self.run_button.clicked.connect(self.on_run)
        self.stop_button.clicked.connect(self.stop)

        self.buttons_layout.addWidget(self.stop_button, stretch=1)
        self.addButtonBarToBottom(self.run_button)

    def on_input_dir_card(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr("Select source folder"), self.src_cfg.value or os.getcwd())
        if directory:
            self.src_cfg.value = directory
            self.src_card.setContent(directory)
            if not (self.out_cfg.value or '').strip():
                base = os.path.basename(os.path.normpath(directory))
                suggested = os.path.join(os.path.dirname(directory), base + '_png')
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
        self.ignore_card.setEnabled(not running)
        self.fo76_card.setEnabled(not running)
        self.fo4_metal_gray_card.setEnabled(not running)

    def on_run(self):
        src = (self.src_cfg.value or '').strip()
        out = (self.out_cfg.value or '').strip()
        ignore_csv = (self.ignore_cfg.value or '').strip()
        fo76 = bool(self.fo76_cfg.value)
        fo4_metal_gray = bool(self.fo4_metal_gray_cfg.value)

        if not src or not os.path.isdir(src):
            InfoBar.warning(title=self.tr("Validation"), content=self.tr("Please select a valid source folder."), duration=3000, parent=self)
            return
        if not out:
            InfoBar.warning(title=self.tr("Validation"), content=self.tr("Please choose an output folder."), duration=3000, parent=self)
            return
        try:
            os.makedirs(out, exist_ok=True)
        except Exception:
            pass

        ignore_patterns = [p.strip() for p in ignore_csv.split(',') if p.strip()] if ignore_csv else []

        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'show_progress'):
            try:
                p.show_progress()
            except Exception:
                pass

        self.set_running(True)
        self.worker = ExportWorker(src, out, ignore_patterns, fallout76_mode=fo76, fo4_metallic_grayscale=fo4_metal_gray)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def stop(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.abort()

    @QtCore.Slot(int, int, str)
    def on_progress(self, processed: int, total: int, message: str):
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
        # Stop the parent loader/spinner immediately upon completion
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'complete_loader'):
            try:
                p.complete_loader()
            except Exception:
                pass
        dlg = QDialog(self)
        dlg.setWindowTitle(self.tr("Export Completed"))
        layout = QVBoxLayoutQt(dlg)
        lbl = QLabel(message)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        layout.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        dlg.exec()

    @QtCore.Slot(str)
    def on_error(self, message: str):
        logger.debug('ERROR: ' + message)
        self.set_running(False)
        # Stop the parent loader/spinner immediately on error as well
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'complete_loader'):
            try:
                p.complete_loader()
            except Exception:
                pass
        dlg = QDialog(self)
        dlg.setWindowTitle(self.tr("Export Error"))
        layout = QVBoxLayoutQt(dlg)
        lbl = QLabel(self.tr("An error occurred: ") + message)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        layout.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        dlg.exec()
        # No further action; loader already stopped above
