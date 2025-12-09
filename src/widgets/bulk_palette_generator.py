import os
from fnmatch import fnmatch
from typing import Optional

import numpy as np
from PIL import Image
from PySide6 import QtWidgets
from PySide6.QtCore import QThread, Signal
from qfluentwidgets import (
    PushSettingCard,
    PrimaryPushButton,
    ConfigItem,
    FluentIcon as FIF,
    SwitchSettingCard,
    InfoBar,
)

from src.help.bulkpalette_help import BulkPaletteHelp
from src.settings.palette_settings import PaletteSettings
from src.utils.appconfig import cfg
from src.utils.cards import TextSettingCard
from src.utils.dds_utils import load_image, save_image
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger
from src.utils.palette_utils import (
    auto_create_islands_from_rgba,
    build_grayscale_and_palette_from_islands,
    save_islands_npz,
)


class BulkPaletteWorker(QThread):
    """Background worker that walks a directory and generates greyscale + palette for textures.

    This intentionally mirrors the per‑file pipeline used by the interactive
    palette tools, but runs headless over a whole folder.
    """

    progress = Signal(int, int)  # current, total
    info = Signal(str)
    error = Signal(str)
    finished = Signal(int, int, int)  # processed, skipped, failed

    def __init__(
        self,
        src_dir: str,
        out_dir: Optional[str],
        include_patterns: list[str],
        exclude_patterns: list[str],
        include_subdirs: bool,
    ):
        super().__init__()
        self._abort = False
        self.src_dir = src_dir
        self.out_dir = out_dir or ""
        self.include_patterns = include_patterns or ["*_d.dds", "*_d.png"]
        self.exclude_patterns = exclude_patterns or []
        self.include_subdirs = include_subdirs

    def abort(self):
        self._abort = True

    # ---------------------------- helpers ----------------------------
    def _iter_files(self) -> list[str]:
        files: list[str] = []
        base = os.path.abspath(self.src_dir)

        if self.include_subdirs:
            walker = os.walk(base)
        else:
            walker = [(base, [], [f for f in os.listdir(base) if os.path.isfile(os.path.join(base, f))])]

        for root, _dirs, filenames in walker:
            rel_root = os.path.relpath(root, base)
            for name in filenames:
                rel_path = os.path.normpath(os.path.join(rel_root, name)) if rel_root != "." else name

                # Exclude patterns match against the relative path
                if any(fnmatch(rel_path, pat.strip()) for pat in self.exclude_patterns if pat.strip()):
                    continue

                # Include patterns match against filename only
                if self.include_patterns:
                    if not any(fnmatch(name, pat.strip()) for pat in self.include_patterns if pat.strip()):
                        continue

                files.append(os.path.join(root, name))

        return sorted(files)

    def _build_output_paths(self, src_path: str) -> tuple[str, str, str]:
        src_dir, filename = os.path.split(src_path)
        base, ext = os.path.splitext(filename)

        # If an explicit output root is provided, mirror the relative layout
        if self.out_dir:
            rel = os.path.relpath(src_dir, self.src_dir)
            target_dir = os.path.join(self.out_dir, rel)
        else:
            target_dir = src_dir

        os.makedirs(target_dir, exist_ok=True)

        # Respect original extension where possible
        output_ext = ".dds" if ext.lower() == ".dds" else ".png"

        grey_path = os.path.join(target_dir, f"{base}_grayscale{output_ext}")
        palette_path = os.path.join(target_dir, f"{base}_palette{output_ext}")
        applied_path = os.path.join(target_dir, f"{base}_applied{output_ext}")
        return grey_path, palette_path, applied_path

    def _compute_luminance(self, rgba: np.ndarray) -> np.ndarray:
        rgb = rgba[:, :, :3].astype(np.float32)
        lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        return np.clip(lum, 0, 255).astype(np.uint8)

    def _process_one(self, path: str) -> tuple[bool, str | None]:
        try:
            grey_path, palette_path, applied_path = self._build_output_paths(path)

            self.info.emit(f"Processing {os.path.relpath(path, self.src_dir)}")

            # --- Load image (use existing DDS loader so formats stay consistent) ---
            img = load_image(path)
            if img is None:
                return False, "Unsupported format or failed to load"

            # Ensure RGBA for consistent luminance + alpha handling
            if img.mode not in ("RGBA", "RGBa"):
                img = img.convert("RGBA")

            rgba = np.array(img, dtype=np.uint8)

            palette_size = int(cfg.get(cfg.ci_default_palette_size))
            palette_height = int(cfg.get(cfg.ci_palette_row_height))

            # Auto-create islands (headless) and then generate grayscale + palette matching palette_creator.generate_both
            islands, mask_stack, _overflow = auto_create_islands_from_rgba(rgba, palette_size)

            grayscale_np, palette_img, mask_stack_out = build_grayscale_and_palette_from_islands(
                rgba,
                islands,
                mask_stack,
                palette_size,
                palette_height,
            )

            grey_img = Image.fromarray(grayscale_np, mode="L")

            save_image(grey_img, grey_path)
            save_image(palette_img, palette_path, True)

            # Save NPZ state like palette_creator auto-save
            save_islands_npz(path, islands, mask_stack_out, rgba.shape[1], rgba.shape[0])

            return True, None
        
        except Exception as e:
            logger.exception("Bulk palette generation failed for %s", path)
            return False, str(e)

    # ----------------------------- run -----------------------------
    def run(self):  # noqa: D401
        """Entry point for the worker thread."""
        try:
            files = self._iter_files()
            total = len(files)

            if not files:
                self.info.emit("No matching textures found.")
                self.finished.emit(0, 0, 0)
                return

            processed = 0
            skipped = 0
            failed = 0

            for i, path in enumerate(files, start=1):
                if self._abort:
                    self.info.emit("Aborted by user.")
                    break

                ok, err = self._process_one(path)
                if ok:
                    processed += 1
                else:
                    if err:
                        self.error.emit(f"{os.path.basename(path)}: {err}")
                    failed += 1

                self.progress.emit(i, total)

            # If aborted mid‑run, count remaining as skipped
            if self._abort and processed + failed < total:
                skipped += total - (processed + failed)

            self.finished.emit(processed, skipped, failed)

        except Exception as e:  # pragma: no cover - catastrophic failures
            logger.exception("Bulk palette worker crashed")
            self.error.emit(str(e))
            self.finished.emit(0, 0, 1)


class BulkPaletteGeneratorWidget(BaseWidget):
    """Bulk Palette Generator UI.

    Takes an input directory and optional include / exclude patterns (similar
    to the DDS Bulk Resizer) and generates greyscale + palette textures for
    every matched image using the shared palette pipeline.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget], text: str = "Bulk Palette Generator"):
        super().__init__(parent=parent, text=text, vertical=True)

        # --- Persistent config items ---
        self.src_cfg = ConfigItem("bulk_palette", "src", "")
        self.out_cfg = ConfigItem("bulk_palette", "out", "")
        self.include_cfg = ConfigItem("bulk_palette", "include", "*_d.dds,*_d.png")
        self.exclude_cfg = ConfigItem("bulk_palette", "exclude", "")

        # --- Runtime state ---
        self._worker: Optional[BulkPaletteWorker] = None

        # --- Settings / help drawers ---
        self.settings_widget = PaletteSettings(self)
        self.settings_drawer.addWidget(self.settings_widget)
        self.help_widget = BulkPaletteHelp(self)
        self.help_drawer.addWidget(self.help_widget)

        # --- Cards ---
        self.src_card = PushSettingCard(
            self.tr("Source Folder"),
            CustomIcons.FOLDER_IMAGE.icon(stroke=True),
            self.tr("Textures location (all parts/textures)"),
            self.src_cfg.value or "Please Select Source Folder",
        )
        self.out_card = PushSettingCard(
            self.tr("Output folder"),
            CustomIcons.FOLDERRIGHT.icon(stroke=True),
            self.tr("Where generated greyscales and palettes will be written"),
            self.out_cfg.value or self.tr("Same as input if empty"),
        )

        self.include_card = TextSettingCard(
            self.include_cfg,
            CustomIcons.FILTER_ADD.icon(stroke=True),
            self.tr("Include Filter"),
            self.tr("Comma/semicolon separated filename patterns (e.g. *_d.dds,*_d.png)"),
        )
        self.exclude_card = TextSettingCard(
            self.exclude_cfg,
            CustomIcons.FILTER_REMOVE.icon(stroke=True),
            self.tr("Exclude Filter"),
            self.tr("Comma/semicolon separated patterns for subfolders/files to skip"),
        )

        self.subdirs_switch = SwitchSettingCard(
            icon=CustomIcons.SUB.icon(stroke=False),
            title=self.tr("Include subfolders"),
            content=self.tr("Scan all subdirectories under the source folder"),
            configItem=cfg.ci_include_subdirs,
        )

        # Add to layout
        self.addToFrame(self.src_card)
        self.addToFrame(self.subdirs_switch)
        self.addToFrame(self.out_card)
        self.addToFrame(self.include_card)
        self.addToFrame(self.exclude_card)
        self.boxLayout.addStretch(1)

        # --- Bottom controls ---
        self.run_button = PrimaryPushButton(icon=FIF.RIGHT_ARROW, text=self.tr("Run"))
        self.stop_button = PrimaryPushButton(icon=FIF.CANCEL, text=self.tr("Stop"))
        self.stop_button.setEnabled(False)

        self.src_card.clicked.connect(self._on_pick_src)
        self.out_card.clicked.connect(self._on_pick_out)
        self.run_button.clicked.connect(self._on_run)
        self.stop_button.clicked.connect(self._on_stop)

        self.buttons_layout.addWidget(self.stop_button, stretch=1)
        self.addButtonBarToBottom(self.run_button)

    # ----------------------------- UI slots -----------------------------
    def _on_pick_src(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select source folder"), self.src_cfg.value or os.getcwd()
        )
        if directory:
            self.src_cfg.value = directory
            self.src_card.setContent(directory)

            # Suggest output if empty
            if not (self.out_cfg.value or "").strip():
                base = os.path.basename(os.path.normpath(directory))
                suggested = os.path.join(os.path.dirname(directory), base + "_palettes")
                self.out_cfg.value = suggested
                self.out_card.setContent(suggested)

    def _on_pick_out(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select output folder"), self.out_cfg.value or self.src_cfg.value or os.getcwd()
        )
        if directory:
            self.out_cfg.value = directory
            self.out_card.setContent(directory)

    def _set_running(self, running: bool):
        self.run_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.src_card.setEnabled(not running)
        self.out_card.setEnabled(not running)
        self.include_card.setEnabled(not running)
        self.exclude_card.setEnabled(not running)
        self.subdirs_switch.setEnabled(not running)

    # --------------------------- run / stop ---------------------------
    def _on_run(self):
        src = (self.src_cfg.value or "").strip()
        out = (self.out_cfg.value or "").strip()

        if not src:
            InfoBar.warning(
                title=self.tr("Missing source"),
                content=self.tr("Please choose a source folder."),
                duration=3000,
                parent=self,
            )
            return

        if not os.path.isdir(src):
            InfoBar.warning(
                title=self.tr("Invalid source"),
                content=self.tr("Source folder does not exist."),
                duration=3000,
                parent=self,
            )
            return

        # Parse patterns
        include_raw = (self.include_cfg.value or "").replace(";", ",")
        exclude_raw = (self.exclude_cfg.value or "").replace(";", ",")
        include_patterns = [p.strip() for p in include_raw.split(",") if p.strip()]
        exclude_patterns = [p.strip() for p in exclude_raw.split(",") if p.strip()]

        include_subdirs = bool(cfg.get(cfg.ci_include_subdirs))

        self._worker = BulkPaletteWorker(
            src_dir=src,
            out_dir=out,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            include_subdirs=include_subdirs,
        )

        self._worker.progress.connect(self._on_worker_progress)
        self._worker.info.connect(self._on_worker_info)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._on_worker_finished)

        self._set_running(True)
        self._worker.start()

    def _on_stop(self):
        if self._worker is not None:
            self._worker.abort()

    # ------------------------ worker callbacks ------------------------
    def _on_worker_progress(self, i: int, total: int):  # noqa: D401
        """Update main window progress ring if available."""
        try:
            if hasattr(self.window(), "update_progress"):
                self.window().update_progress(int(100 * (i / max(1, total))))
        except Exception:
            pass

    def _on_worker_info(self, text: str):
        logger.info("[BulkPalette] %s", text)

    def _on_worker_error(self, message: str):
        logger.error("[BulkPalette] %s", message)

    def _on_worker_finished(self, processed: int, skipped: int, failed: int):
        self._set_running(False)
        self._worker = None

        InfoBar.info(
            title=self.tr("Bulk Palette Generator"),
            content=self.tr(
                "Finished. Processed: {0}, Skipped: {1}, Failed: {2}".format(
                    processed, skipped, failed
                )
            ),
            duration=4000,
            parent=self,
        )
