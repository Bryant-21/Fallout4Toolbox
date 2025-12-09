import io
import os
import traceback
from typing import Iterable

from PySide6 import QtWidgets
from PySide6.QtWidgets import QFileDialog
from qfluentwidgets import FluentIcon as FIF, InfoBar
from qfluentwidgets import PrimaryPushButton, PushSettingCard, SwitchSettingCard

from src.help.matfiles_help import MatfilesHelp
from src.material_tools.bgem_bin import read_bgem, BGEMData
from src.material_tools.bgsm_bin import read_bgsm, BGSMData
from src.material_tools.json_handler import (
    load_json,
    save_json,
    update_textures_json,
    detect_material_type_from_json,
)

from src.settings.matfiles_settings import MatFilesSettings
from src.utils.appconfig import cfg
from src.utils.cards import TextSettingCard, DoubleSpinSettingCard
from src.utils.helpers import BaseWidget
from src.utils.logging_utils import logger

BGSM_SIGNATURE = 0x4D534742
BGEM_SIGNATURE = 0x4D454742

def is_json_file(path: str) -> bool:
    try:
        with open(path, 'rb') as f:
            head = f.read(1)
            if not head:
                return False
            return head in (b'{', b'[')
    except OSError:
        return False


def detect_binary_type(path: str) -> str | None:
    try:
        with open(path, 'rb') as f:
            data = f.read(4)
            if len(data) < 4:
                return None
            sig = int.from_bytes(data, 'little')
            if sig == BGSM_SIGNATURE:
                return 'BGSM'
            if sig == BGEM_SIGNATURE:
                return 'BGEM'
            return None
    except OSError:
        return None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def each_material_file(root: str, skip_dirnames: set[str] | None = None, skip_roots: list[str] | None = None, recursive: bool = False) -> Iterable[str]:
    """Yield material files from the root directory.
    By default, this only scans the top-level of `root` (no subdirectories).
    Set `recursive=True` to walk subdirectories as well.
    `skip_dirnames` and `skip_roots` are only relevant when `recursive=True`.
    """
    if not recursive:
        # Non-recursive: only consider files directly under `root`
        try:
            for name in os.listdir(root):
                full_path = os.path.join(root, name)
                if not os.path.isfile(full_path):
                    continue
                lower = name.lower()
                if lower.endswith('.bgsm') or lower.endswith('.bgem') or lower.endswith('.json'):
                    yield full_path
        except OSError:
            return
        return

    # Recursive mode
    skip_dirnames = skip_dirnames or set()
    skip_roots = [os.path.abspath(p) for p in (skip_roots or [])]
    for dirpath, dirnames, filenames in os.walk(root):
        # prevent descending into specified output folders or roots to avoid infinite copying
        if skip_dirnames:
            dirnames[:] = [d for d in dirnames if d not in skip_dirnames]
        if skip_roots:
            # prune any child dir that is under a skipped root
            keep = []
            parent = os.path.abspath(dirpath)
            for d in dirnames:
                full = os.path.abspath(os.path.join(parent, d))
                if any(full.startswith(rp) for rp in skip_roots):
                    continue
                keep.append(d)
            dirnames[:] = keep
        for name in filenames:
            lower = name.lower()
            if lower.endswith('.bgsm') or lower.endswith('.bgem') or lower.endswith('.json'):
                yield os.path.join(dirpath, name)


def prefix_texture(s: str | None, folder: str) -> str | None:
    """Insert the folder immediately before the filename component of s.
    Examples:
      - s="Weapons/B21_GaussMinigun/barrel1_d.dds", folder="asd"
        -> "Weapons/B21_GaussMinigun/asd/barrel1_d.dds"
      - s="barrel1_d.dds", folder="asd" -> "asd/barrel1_d.dds"
    Avoid double-inserting if the folder is already directly before the filename.
    """
    if not s:
        return s
    q = s.replace('\\', '/')  # normalize to forward slashes for materials
    # Split into dir part and filename
    if '/' in q:
        dirpart, fname = q.rsplit('/', 1)
    else:
        dirpart, fname = '', q
    # If the folder is already the last directory before the filename, keep as-is
    if dirpart and dirpart.endswith('/' + folder):
        return q
    if not dirpart and q.startswith(folder + '/'):
        # Path starts with the folder but there was no explicit dirpart split
        # This means format is already folder/filename
        return q
    # Insert folder between dirpart (if any) and filename
    if dirpart:
        return f"{dirpart}/{folder}/{fname}"
    else:
        return f"{folder}/{fname}"


def process_binary_bgsm(src_path: str, folders: list[str], out_root: str | None, selected_paths: set[str] | None = None, grayscale_to_palette_scale: float | None = None, logger=None) -> None:
    if logger:
        logger(f"Reading BGSM: {src_path}")
    with open(src_path, 'rb') as f:
        br = io.BufferedReader(f)
        bgsm = read_bgsm(br)
    # Update and write per folder
    for folder in folders:
        new_bgsm = BGSMData(**{**bgsm.__dict__})
        def maybe_update(attr: str):
            if selected_paths is None or attr in selected_paths:
                val = getattr(new_bgsm, attr, None)
                if isinstance(val, str) and val:
                    setattr(new_bgsm, attr, prefix_texture(val, folder) or "")
        # Optionally set grayscale-to-palette scale
        if grayscale_to_palette_scale is not None:
            try:
                new_bgsm.GrayscaleToPaletteScale = float(grayscale_to_palette_scale)
            except Exception:
                pass
        # Apply selection across supported BGSM fields
        for attr in (
            "DiffuseTexture",
            "NormalTexture",
            "SmoothSpecTexture",
            "GreyscaleTexture",
            "EnvmapTexture",
            "GlowTexture",
            "InnerLayerTexture",
            "WrinklesTexture",
        ):
            maybe_update(attr)
        # Build output path
        base_dir = out_root or os.path.dirname(src_path)
        target_dir = os.path.join(base_dir, folder)
        ensure_dir(target_dir)
        out_path = os.path.join(target_dir, os.path.basename(src_path))
        with open(out_path, 'wb') as out_f:
            bw = io.BufferedWriter(out_f)
            new_bgsm.write(bw)
            bw.flush()
        if logger:
            logger(f"Wrote BGSM: {out_path} (folder={folder})")


def process_binary_bgem(src_path: str, folders: list[str], out_root: str | None, selected_paths: set[str] | None = None, logger=None) -> None:
    if logger:
        logger(f"Reading BGEM: {src_path}")
    with open(src_path, 'rb') as f:
        br = io.BufferedReader(f)
        bgem = read_bgem(br)
    for folder in folders:
        new_bgem = BGEMData(**{**bgem.__dict__})
        # Map BGSM logical names to BGEM attributes
        mapping = {
            "DiffuseTexture": "BaseTexture",
            "NormalTexture": "NormalTexture",
            "SmoothSpecTexture": "SpecularTexture",
            "GreyscaleTexture": "GrayscaleTexture",
            "EnvmapTexture": "EnvmapTexture",
            "GlowTexture": "GlowTexture",
            # InnerLayerTexture and WrinklesTexture don't exist in BGEM
        }
        for logical, attr in mapping.items():
            if selected_paths is None or logical in selected_paths:
                val = getattr(new_bgem, attr, None)
                if isinstance(val, str) and val:
                    setattr(new_bgem, attr, prefix_texture(val, folder) or "")
        base_dir = out_root or os.path.dirname(src_path)
        target_dir = os.path.join(base_dir, folder)
        ensure_dir(target_dir)
        out_path = os.path.join(target_dir, os.path.basename(src_path))
        with open(out_path, 'wb') as out_f:
            bw = io.BufferedWriter(out_f)
            new_bgem.write(bw)
            bw.flush()
        if logger:
            logger(f"Wrote BGEM: {out_path} (folder={folder})")


def process_json(src_path: str, folders: list[str], out_root: str | None, include_bgsm: bool = True, include_bgem: bool = True, selected_paths: set[str] | None = None, grayscale_to_palette_scale: float | None = None, logger=None) -> None:
    mat_type, obj = load_json(src_path)
    # Filter by type if requested
    if (mat_type == 'BGSM' and not include_bgsm) or (mat_type == 'BGEM' and not include_bgem):
        if logger:
            logger(f"Skipping JSON ({mat_type}) due to filter: {src_path}")
        return
    for folder in folders:
        new_obj = update_textures_json(obj, mat_type, folder, selected_paths=selected_paths)
        # Optionally set grayscale-to-palette scale for BGSM JSONs
        if grayscale_to_palette_scale is not None and mat_type == 'BGSM':
            try:
                new_obj['fGrayscaleToPaletteScale'] = float(grayscale_to_palette_scale)
            except Exception:
                pass
        base_dir = out_root or os.path.dirname(src_path)
        target_dir = os.path.join(base_dir, folder)
        ensure_dir(target_dir)
        out_path = os.path.join(target_dir, os.path.basename(src_path))
        save_json(out_path, new_obj)
        if logger:
            logger(f"Wrote JSON {mat_type}: {out_path} (folder={folder})")


def run(input_dir: str, folders: list[str], out_root: str | None = None, include_bgsm: bool = True, include_bgem: bool = True, selected_paths: set[str] | None = None, exclude_patterns: list[str] | None = None, grayscale_to_palette_scale: float | None = None, clean_output: bool = False, logger=None) -> None:
    def emit(msg: str) -> None:
        if logger:
            try:
                logger(msg)
            except Exception:
                # Never let logging break the run
                pass
    input_dir_abs = os.path.abspath(input_dir)
    out_root_abs = os.path.abspath(out_root) if out_root else None
    write_root_abs = out_root_abs or input_dir_abs

    sel_display = sorted(selected_paths) if selected_paths else '(default)'
    ex_display = ', '.join(exclude_patterns) if exclude_patterns else '(none)'
    emit(
        f"Run params:\n  input_dir={input_dir_abs}\n  out_root={out_root_abs or '(same as input)'}\n  folders={folders}\n  include_bgsm={include_bgsm} include_bgem={include_bgem}\n  selected_paths={sel_display}\n  exclude_patterns={ex_display}\n  grayscale_to_palette_scale={grayscale_to_palette_scale if grayscale_to_palette_scale is not None else '(unchanged)'}\n  clean_output={clean_output}"
    )

    # Normalize exclude patterns to lowercase for case-insensitive matching
    norm_excludes = [p.lower() for p in exclude_patterns] if exclude_patterns else []

    # Determine directories to skip during scan to avoid rescanning outputs
    skip_names: set[str] = set()
    skip_roots: list[str] = []
    if out_root_abs is None:
        # Writing under input_dir/<folder>; skip those folder names during the same walk
        skip_names = set(folders)
    else:
        # If output root resides inside input_dir, skip those concrete output roots
        out_root_abs_norm = os.path.normcase(out_root_abs)
        input_dir_norm = os.path.normcase(input_dir_abs)
        if out_root_abs_norm.startswith(input_dir_norm):
            skip_roots = [os.path.join(out_root_abs, f) for f in folders]

    if skip_names:
        emit(f"Skipping subdirectories by name: {sorted(skip_names)}")
    if skip_roots:
        emit(f"Skipping concrete output roots: {skip_roots}")

    # Optionally clean target subfolders before copying
    if clean_output:
        try:
            for f in folders:
                target_dir = os.path.join(write_root_abs, f)
                if not os.path.exists(target_dir):
                    continue
                emit(f"Cleaning output folder: {target_dir}")
                # Remove files and empty subdirectories inside target_dir
                for dirpath, dirnames, filenames in os.walk(target_dir, topdown=False):
                    for name in filenames:
                        fp = os.path.join(dirpath, name)
                        try:
                            os.remove(fp)
                        except Exception as ex:
                            emit(f"  Failed to remove file {fp}: {ex}")
                    for d in dirnames:
                        dp = os.path.join(dirpath, d)
                        try:
                            # Only remove if empty after files removed
                            os.rmdir(dp)
                        except OSError:
                            # Non-empty, leave it
                            pass
        except Exception as ex:
            emit(f"Failed to clean output folders: {ex}")

    scanned = 0
    processed_bgsm = 0
    processed_bgem = 0
    processed_json_bgsm = 0
    processed_json_bgem = 0
    skipped_filtered = 0
    skipped_excluded = 0
    skipped_unknown = 0
    errors = 0

    for path in each_material_file(input_dir_abs, skip_dirnames=skip_names, skip_roots=skip_roots):
        scanned += 1
        try:
            # Apply filename exclude patterns (case-insensitive, substring match)
            if norm_excludes:
                base_lower = os.path.basename(path).lower()
                matched = None
                for pat in norm_excludes:
                    if pat and pat in base_lower:
                        matched = pat
                        break
                if matched:
                    skipped_filtered += 1
                    skipped_excluded += 1
                    emit(f"Skipped by exclude pattern '{matched}': {path}")
                    continue

            if is_json_file(path):
                # Determine type from JSON for logging
                try:
                    with open(path, 'r', encoding='utf-8') as jf:
                        import json as _json
                        obj = _json.load(jf)
                    jtype = detect_material_type_from_json(obj)
                except Exception:
                    jtype = 'JSON'
                emit(f"Found JSON ({jtype}): {path}")
                if (jtype == 'BGSM' and not include_bgsm) or (jtype == 'BGEM' and not include_bgem):
                    skipped_filtered += 1
                    emit(f"  -> Skipped by type filter")
                else:
                    process_json(path, folders, write_root_abs, include_bgsm=include_bgsm, include_bgem=include_bgem, selected_paths=selected_paths, grayscale_to_palette_scale=grayscale_to_palette_scale, logger=logger)
                    if jtype == 'BGSM':
                        processed_json_bgsm += 1
                    elif jtype == 'BGEM':
                        processed_json_bgem += 1
                continue
            mtype = detect_binary_type(path)
            if mtype == 'BGSM':
                if not include_bgsm:
                    skipped_filtered += 1
                    emit(f"Found BGSM (binary), skipped by filter: {path}")
                else:
                    emit(f"Processing BGSM (binary): {path}")
                    process_binary_bgsm(path, folders, write_root_abs, selected_paths=selected_paths, grayscale_to_palette_scale=grayscale_to_palette_scale, logger=logger)
                    processed_bgsm += 1
            elif mtype == 'BGEM':
                if not include_bgem:
                    skipped_filtered += 1
                    emit(f"Found BGEM (binary), skipped by filter: {path}")
                else:
                    emit(f"Processing BGEM (binary): {path}")
                    process_binary_bgem(path, folders, out_root_abs, selected_paths=selected_paths, logger=logger)
                    processed_bgem += 1
            else:
                skipped_unknown += 1
                emit(f"Unknown or unsupported file signature, skipped: {path}")
        except Exception as ex:
            errors += 1
            emit(f"Error processing {path}: {ex}")

    emit(
        "Summary:\n"
        f"  scanned={scanned}\n"
        f"  processed_bgsm_binary={processed_bgsm}\n"
        f"  processed_bgem_binary={processed_bgem}\n"
        f"  processed_json_bgsm={processed_json_bgsm}\n"
        f"  processed_json_bgem={processed_json_bgem}\n"
        f"  skipped_filtered={skipped_filtered}\n"
        f"  skipped_excluded={skipped_excluded}\n"
        f"  skipped_unknown={skipped_unknown}\n"
        f"  errors={errors}"
    )


class MaterialToolUI(BaseWidget):
    def __init__(self, parent, text):
        super().__init__(parent=parent, text=text, vertical=True)

        # Use shared app config items instead of local definitions
        self.input_dir = cfg.input_dir
        self.output_root = cfg.output_root
        self.folders_cfg = cfg.folders_cfg
        self.excludes_cfg = cfg.excludes_cfg

        # --- Main scroll area ---
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        content_widget = QtWidgets.QWidget()
        scroll_area.setWidget(content_widget)

        # --- Base cards ---
        self.input_dir_card = PushSettingCard(
            self.tr('Source Directory'),
            FIF.FOLDER,
            self.tr("Where to read BGSM or BGEM"),
            self.input_dir.value
        )
        self.output_root_card = PushSettingCard(
            self.tr('Output Root (optional)'),
            FIF.FOLDER,
            self.tr("Where to write updated files. If empty, writes next to input."),
            self.output_root.value
        )
        self.folders_card = TextSettingCard(
            self.folders_cfg,
            FIF.EDIT,
            self.tr("Folder Names to Add. Comma-separated."),
            self.folders_cfg.value or self.tr('(e.g. bos, enclave, gold, etc)')
        )
        self.excludes_card = TextSettingCard(
            self.excludes_cfg,
            FIF.FILTER,
            self.tr("Matching file names will be skipped."),
            self.excludes_cfg.value or self.tr('(copper, glass, etc)')
        )

        self.addToFrame(self.input_dir_card)
        self.addToFrame(self.output_root_card)
        self.addToFrame(self.folders_card)
        self.addToFrame(self.excludes_card)

        # Clean output folders option
        self.clean_output_card = SwitchSettingCard(
            FIF.DELETE,
            self.tr("Clean target folders before copy"),
            self.tr("Removes any files inside each target subfolder before writing"),
            cfg.clean_output_cfg,
        )
        self.addToFrame(self.clean_output_card)

        # Grayscale To Palette Scale (optional)
        self.grayscale_scale_card = DoubleSpinSettingCard(
            cfg.grayscale_to_palette_scale_cfg,
            FIF.EDIT,
            self.tr("Grayscale To Palette Scale"),
        )
        self.addToFrame(self.grayscale_scale_card)

        self.boxLayout.addStretch(1)


        self.run_button = PrimaryPushButton(icon=FIF.RIGHT_ARROW, text=self.tr("Run"))

        # --- Wire up events ---
        self.run_button.clicked.connect(self.on_run)
        self.input_dir_card.clicked.connect(self.on_input_dir_card)
        self.output_root_card.clicked.connect(self.on_output_root_card)

        self.settings_widget = MatFilesSettings(self)
        self.settings_drawer.addWidget(self.settings_widget)
        self.help_widget = MatfilesHelp(self)
        self.help_drawer.addWidget(self.help_widget)

        self.addButtonBarToBottom(self.run_button)

    def on_input_dir_card(self):
        directory = QFileDialog.getExistingDirectory(self, self.tr("Select input directory"), self.input_dir.value or os.getcwd())
        if directory:
            self.input_dir.value = directory
            self.input_dir_card.setContent(directory)

    def on_output_root_card(self):
        directory = QFileDialog.getExistingDirectory(self, self.tr("Select output root"), self.output_root.value)
        if directory:
            self.output_root.value = directory
            self.output_root_card.setContent(directory)

    def on_run(self) -> None:
        input_dir = (self.input_dir.value or '').strip()
        out_root = (self.output_root.value or '').strip() or None
        folders_raw = (self.folders_cfg.value or '').strip()
        excludes_raw = (self.excludes_cfg.value or '').strip()
        include_bgsm = cfg.bgsm_cfg.value
        include_bgem = cfg.bgem_cfg.value

        # Validation
        if not input_dir or not os.path.isdir(input_dir):
            InfoBar.warning(
                title=self.tr("Validation"),
                content=self.tr("Please select a valid input directory."),
                duration=3000,
                parent=self,
            )
            return

        if not folders_raw:
            InfoBar.warning(
                title=self.tr("Validation"),
                content=self.tr("Please enter at least one folder name (comma-separated)."),
                duration=3000,
                parent=self,
            )
            return
        if not include_bgsm and not include_bgem:
            InfoBar.warning(
                title=self.tr("Validation"),
                content=self.tr("Please select at least one of: Include BGSM or Include BGEM."),
                duration=3000,
                parent=self,
            )
            return

            folders = [f.strip() for f in folders_raw.split(',') if f.strip()]
            if not folders:
                InfoBar.warning(
                    title=self.tr("Validation"),
                    content=self.tr("No valid folder names were provided."),
                    duration=3000,
                    parent=self,
                )
                return

        # Collect selected texture fields from SwitchSettingCards (persistent ConfigItems)
        selected_paths = set()
        if cfg.tex_diffuse_cfg.value:
            selected_paths.add("DiffuseTexture")
        if cfg.tex_normal_cfg.value:
            selected_paths.add("NormalTexture")
        if cfg.tex_smoothspec_cfg.value:
            selected_paths.add("SmoothSpecTexture")
        if cfg.tex_greyscale_cfg.value:
            selected_paths.add("GreyscaleTexture")
        if cfg.tex_envmap_cfg.value:
            selected_paths.add("EnvmapTexture")
        if cfg.tex_glow_cfg.value:
            selected_paths.add("GlowTexture")
        if cfg.tex_inner_cfg.value:
            selected_paths.add("InnerLayerTexture")
        if cfg.tex_wrinkles_cfg.value:
            selected_paths.add("WrinklesTexture")

        excludes = [p.strip() for p in excludes_raw.split(',') if p.strip()] if excludes_raw else []
        logger.debug(f"Starting…\n  Input: {input_dir}\n  Output: {out_root or '(same as input)'}\n  Folders: {', '.join(folders)}\n  Types: {'BGSM' if include_bgsm else ''} {'BGEM' if include_bgem else ''}\n  Selected: {', '.join(sorted(selected_paths)) or '(default)'}\n  Excludes: {', '.join(excludes) if excludes else '(none)'}\n")
        self.run_button.setEnabled(False)

        try:
            run(input_dir, folders, out_root, include_bgsm=include_bgsm, include_bgem=include_bgem, selected_paths=selected_paths or None, exclude_patterns=excludes, grayscale_to_palette_scale=(None if cfg.grayscale_to_palette_scale_cfg.value <= 0 else float(cfg.grayscale_to_palette_scale_cfg.value)), clean_output=bool(cfg.clean_output_cfg.value), logger=logger.debug)
            logger.debug("Done.")
        except Exception as ex:
            tb = traceback.format_exc()
            logger.debug(f"Error: {ex}\n{tb}")
            InfoBar.error(
                title=self.tr("Error"),
                content=self.tr(f"An error occurred:\n{ex}"),
                duration=5000,
                parent=self,
            )
        finally:
            self.run_button.setEnabled(True)
