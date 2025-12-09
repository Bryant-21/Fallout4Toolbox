from pathlib import Path
from typing import Optional, List

from PIL import Image
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget, QFileDialog
from mipflooding.wrapper import image_processing as _mip_image_processing
from qfluentwidgets import (
    PushSettingCard,
    PrimaryPushButton,
    InfoBar,
    FluentIcon as FIF,
)

from src.help.mip_help import MipHelp
from src.settings.basic_settings import BasicSettings
from src.utils.appconfig import cfg
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger
from src.utils.mipflooding import _apply_mip_flooding_to_png


class MipFloodingWorker(QThread):
    progress = Signal(int, int)
    info = Signal(str)
    finished = Signal(int, int, int)
    error = Signal(str)

    def __init__(self, input_dir: str, output_dir: Optional[str]):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self._stop = False

    def run(self):
        try:
            root = Path(self.input_dir)
            out_dir = Path(self.output_dir) if self.output_dir else None
            if out_dir and not out_dir.exists():
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    self.error.emit(f"Cannot create output folder: {e}")
                    return
            files = MipFloodingWidget._iter_image_files_static(root, True)
            if not files:
                self.finished.emit(0, 0, 0)
                return
            processed = skipped = failed = 0
            for i, f in enumerate(files, start=1):
                if self._stop:
                    break
                try:
                    if not f.exists():
                        skipped += 1
                        continue
                    ok = MipFloodingWidget._process_one_static(f, out_dir)
                    if ok:
                        processed += 1
                    else:
                        skipped += 1
                except Exception as e:
                    logger.warning(f"Failed processing {f}: {e}")
                    failed += 1
                self.progress.emit(i, len(files))
                if i % 10 == 0 or i == len(files):
                    self.info.emit(f"Processed {i}/{len(files)} (ok={processed}, skipped={skipped}, failed={failed})")
            self.finished.emit(processed, skipped, failed)
        except Exception as e:
            self.error.emit(str(e))

    def abort(self):
        self._stop = True


class MipFloodingWidget(BaseWidget):
    # --- worker signal handlers ---
    def _on_worker_progress(self, i: int, total: int):
        try:
            if total:
                percent = int(max(0, min(100, round((i / total) * 100))))
                p = getattr(self, 'parent', None)
                if p and hasattr(p, 'update_progress'):
                    p.update_progress(percent)
        except Exception:
            pass

    def _on_worker_info(self, text: str):
        try:
            InfoBar.info(self.tr("Processing"), self.tr(text), duration=3000, parent=self)
        except Exception:
            pass

    def _on_worker_finished(self, processed: int, skipped: int, failed: int):
        try:
            self.btn_process.setEnabled(True)
            InfoBar.success(self.tr("Done"), self.tr(f"Finished. ok={processed}, skipped={skipped}, failed={failed}"), duration=5000, parent=self)
        finally:
            p = getattr(self, 'parent', None)
            if p and hasattr(p, 'complete_loader'):
                try:
                    p.complete_loader()
                except Exception:
                    pass

    def _on_worker_error(self, message: str):
        try:
            InfoBar.error(self.tr("Error"), self.tr(message), duration=5000, parent=self)
        finally:
            self.btn_process.setEnabled(True)
            p = getattr(self, 'parent', None)
            if p and hasattr(p, 'complete_loader'):
                try:
                    p.complete_loader()
                except Exception:
                    pass
    """UI to apply mip flooding to all transparent PNGs or DDS images in a folder.

    Rules:
    - PNG: only process if the image has transparency (alpha < 255). If fully opaque, skip.
    - DDS: load as RGBA using texconv; use alpha as transparency; if fully opaque alpha, skip.
    - Output: write a PNG next to the source (or to selected output directory) with suffix _mip.png.
    """

    def __init__(self, parent: Optional[QWidget] = None, text: str = "MIp Flooding"):
        super().__init__(text=text, parent=parent, vertical=True)

        # Input/Output folder cards
        self.input_dir_card = PushSettingCard(
            self.tr("Input Folder"),
            CustomIcons.FOLDER_IMAGE.icon(stroke=True),
            self.tr("Select a folder containing images"),
            cfg.get(cfg.textures_dir_cfg),
        )
        self.input_dir_card.clicked.connect(self._on_pick_input_dir)

        self.output_dir_card = PushSettingCard(
            self.tr("Output Folder (optional)"),
            FIF.SAVE.icon(),
            self.tr("Select output folder for processed PNGs (default: alongside source)"),
            cfg.get(cfg.output_dir_cfg),
        )
        self.output_dir_card.clicked.connect(self._on_pick_output_dir)

        self.addToFrame(self.input_dir_card)
        self.addToFrame(self.output_dir_card)

        # Options: currently processes subfolders by default to simplify configuration
        self._recursive_state = True

        # Buttons
        self.btn_process = PrimaryPushButton(icon=FIF.RIGHT_ARROW, text=self.tr("Process Folder"))
        self.btn_process.clicked.connect(self._on_process_folder)
        self.addButtonBarToBottom(self.btn_process)

        self.settings_widget = BasicSettings(self)
        self.settings_drawer.addWidget(self.settings_widget)

        self.help_widget = MipHelp(self)
        self.help_drawer.addWidget(self.help_widget)

        # Internal state
        self._input_dir: Optional[str] = None
        self._output_dir: Optional[str] = None
        self._processed: int = 0
        self._skipped: int = 0
        self._failed: int = 0

    # ---------- UI callbacks ----------
    def _on_pick_input_dir(self):
        d = QFileDialog.getExistingDirectory(self, self.tr("Select input folder"), self._input_dir or "")
        if d:
            self._input_dir = d
            try:
                self.input_dir_card.setContent(d)
            except Exception:
                pass

    def _on_pick_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, self.tr("Select output folder"), self._output_dir or "")
        if d:
            self._output_dir = d
            try:
                self.output_dir_card.setContent(d)
            except Exception:
                pass

    # ---------- Core processing ----------
    @staticmethod
    def _iter_image_files_static(root: Path, recursive: bool) -> List[Path]:
        patterns = ["*.png", "*.PNG", "*.dds", "*.DDS"]
        files: List[Path] = []
        if recursive:
            for pat in patterns:
                files.extend(root.rglob(pat))
        else:
            for pat in patterns:
                files.extend(root.glob(pat))
        return files

    @staticmethod
    def _has_transparency_alpha(img: Image.Image) -> bool:
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        a = img.getchannel('A')
        mn, mx = a.getextrema()
        return mn < 255  # any transparency present

    @staticmethod
    def _process_one_static(path: Path, out_dir: Optional[Path]) -> bool:
        try:
            rgba = load_image(str(path), 'RGBA')
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return False
        if not MipFloodingWidget._has_transparency_alpha(rgba):
            logger.info(f"Skipping {path.name}: no transparency in alpha channel")
            return False
        target_dir = out_dir or path.parent
        out_path = target_dir / f"{path.stem}_mip.png"
        ok = _apply_mip_flooding_to_png(out_path, rgba)
        if ok:
            logger.info(f"Mip flooded: {path.name} -> {out_path.name}")
        return ok
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        a = img.getchannel('A')
        mn, mx = a.getextrema()
        return mn < 255  # any transparency present


    def _on_process_folder(self):
        if not self._input_dir:
            InfoBar.warning(self.tr("Missing Input"), self.tr("Please select an input folder."), duration=3000, parent=self)
            return

        # Disable UI and show progress mask
        self.btn_process.setEnabled(False)
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'show_progress'):
            try:
                p.show_progress()
            except Exception:
                pass

        # Start worker thread
        self.worker = MipFloodingWorker(self._input_dir, self._output_dir)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.info.connect(self._on_worker_info)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()
        return
