import os
from typing import Optional, Tuple, List

from PIL import Image
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import QFileDialog, QWidget, QLabel, QGridLayout, QMessageBox
from qfluentwidgets import (
    PushSettingCard,
    PrimaryPushButton,
    PushButton,
    SwitchSettingCard,
    FluentIcon as FIF,
)

from help.upscaler_help import UpscalerHelp
from settings.basic_settings import BasicSettings
from src.utils.appconfig import cfg
from src.utils.chainner_utils import CHAINNER_EXE, run_chainner, resolve_model_path, get_or_download_model, upscale_directory_two_pass
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger
from src.utils.cards import ComboBoxSettingsCard
from src.utils.dds_utils import load_image


class UpscaleWorker(QThread):
    progress = Signal(int, int, str)  # current, total, message
    finished = Signal(int, int, int)  # saved, skipped, failed
    error = Signal(str)

    def __init__(self, folder: str, output_dir: Optional[str], preview_only: Optional[str] = None, include_subdirs: bool = True):
        super().__init__()
        self.folder = folder
        self.output_dir = output_dir
        self.preview_only = preview_only  # when set, process only this file
        self.include_subdirs = include_subdirs
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            if not os.path.exists(CHAINNER_EXE):
                raise Exception("ChaiNNer.exe not found. Please place it under resource/ChaiNNer.")

            # Preview mode: process a single image using the single-file chain
            if self.preview_only:
                files: List[str] = [self.preview_only]
                total = len(files)
                saved = skipped = failed = 0
                for i, f in enumerate(files, start=1):
                    if self._abort:
                        break
                    try:
                        out_file, did = self._process_one(f)
                        if did:
                            saved += 1
                        else:
                            failed += 1
                    except Exception:
                        logger.exception("Upscale error on %s", f)
                        failed += 1
                    self.progress.emit(i, total, os.path.basename(f))
                self.finished.emit(saved, skipped, failed)
                return

            # Batch mode: use directory-based chain; two passes (_d then _n)
            out_dir = self.output_dir or self.folder
            textures_model = cfg.get(cfg.upscale_textures_cfg)
            normals_model = cfg.get(cfg.upscale_normals_cfg)

            # Emit coarse progress for two passes
            self.progress.emit(1, 2, "Diffuse (_d) pass")
            saved, skipped, failed = upscale_directory_two_pass(
                self.folder, out_dir, self.include_subdirs, textures_model, normals_model
            )
            # Final progress mark
            self.progress.emit(2, 2, "Normals (_n) pass")
            self.finished.emit(saved, skipped, failed)
        except Exception as e:
            self.error.emit(str(e))

    def _pick_model_for(self, path: str) -> Tuple[str, str]:
        name = os.path.splitext(os.path.basename(path))[0].lower()
        # By convention: *_n* are normal maps, *_d* are diffuse/albedo
        if name.endswith("_n"):
            model = cfg.get(cfg.upscale_normals_cfg)
        else:
            # default to textures cfg for all others, including *_d
            model = cfg.get(cfg.upscale_textures_cfg)
        local = resolve_model_path(model)
        return model, local

    def _convert_to_png_if_needed(self, path: str) -> str:
        """ChaiNNer can load DDS directly; no conversion needed for processing."""
        return path

    def _compute_output_dir_and_expected(self, src_path: str) -> Tuple[str, str]:
        """Return (output_dir, expected_output_png_path) using original stem plus '_upscaled'.
        """
        out_dir = self.output_dir or os.path.dirname(src_path)
        name, _ = os.path.splitext(os.path.basename(src_path))
        # Expected file name is '<stem>_upscaled.png'
        expected = os.path.join(out_dir, f"{name}_upscaled.png")
        return out_dir, expected

    def _run_chainner(self, input_png: str, model_path: str, out_dir: str, expected_output_png: str) -> bool:
        """Delegate to the shared ChaiNNer runner to execute the chain."""
        return run_chainner(input_png, model_path, out_dir, expected_output_png)

    def _process_one(self, path: str) -> Tuple[Optional[str], bool]:
        model_name, _ = self._pick_model_for(path)
        # Ensure model exists locally or download it automatically
        model_path = get_or_download_model(model_name)
        out_dir, expected_png = self._compute_output_dir_and_expected(path)
        os.makedirs(out_dir, exist_ok=True)
        ok = self._run_chainner(path, model_path, out_dir, expected_png)
        return (expected_png if ok else None), ok


class UpscaleWidget(BaseWidget):
    def __init__(self, parent=None, text: str = "Upscale"):
        super().__init__(text=text, parent=parent, vertical=True)

        # Cards
        self.folder_card = PushSettingCard(
            self.tr("Textures Folder"),
            CustomIcons.FOLDER_IMAGE.icon(stroke=True),
            self.tr("Select a folder with textures"),
            cfg.get(cfg.convert_dir_cfg),
        )
        self.folder_card.clicked.connect(self._on_pick_folder)

        self.output_card = PushSettingCard(
            self.tr("Output Folder (optional)"),
            FIF.SAVE.icon(),
            self.tr("Select output folder (default: alongside source)"),
            "",
        )
        self.output_card.clicked.connect(self._on_pick_output)

        self.addToFrame(self.folder_card)
        self.addToFrame(self.output_card)

        # Model selectors info (read-only cards displaying chosen cfgs)
        self.normals_card = ComboBoxSettingsCard(
            icon=CustomIcons.ARROW_UP.icon(),
            title=self.tr("Normals Model"),
            content=cfg.get(cfg.upscale_normals_cfg),
            parent=self,
            configItem=cfg.upscale_normals_cfg
        )

        self.textures_card = ComboBoxSettingsCard(
            icon=CustomIcons.ENHANCE.icon(),
            title=self.tr("Textures Model"),
            content=cfg.get(cfg.upscale_textures_cfg),
            parent=self,
            configItem=cfg.upscale_textures_cfg
        )

        self.addToFrame(self.normals_card)
        self.addToFrame(self.textures_card)

        self.chk_include_subdirs = SwitchSettingCard(
            icon=CustomIcons.FOLDERRIGHT.icon(stroke=True),
            title=self.tr("Include Subfolders"),
            content=self.tr("Process images in subdirectories too"),
            configItem=None,
        )
        # Not using cfg item; maintain local state
        self._include_subdirs = True
        self.chk_include_subdirs.switchButton.setChecked(True)
        self.chk_include_subdirs.checkedChanged.connect(self._on_include_changed)
        self.addToFrame(self.chk_include_subdirs)

        # Preview
        self.original_label = QLabel(self.tr("Original"))
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setMinimumSize(450, 450)
        self.original_label.setStyleSheet("border: 1px dashed gray;")

        self.upscaled_label = QLabel(self.tr("Upscaled Preview"))
        self.upscaled_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.upscaled_label.setMinimumSize(450, 450)
        self.upscaled_label.setStyleSheet("border: 1px dashed gray;")

        grid = QGridLayout()
        grid.addWidget(QLabel(self.tr("Original")), 0, 0)
        grid.addWidget(QLabel(self.tr("Upscaled")), 0, 1)
        grid.addWidget(self.original_label, 1, 0)
        grid.addWidget(self.upscaled_label, 1, 1)
        container = QWidget()
        container.setLayout(grid)
        self.boxLayout.addStretch(1)
        self.addToFrame(container)

        # Buttons
        self.btn_preview = PushButton(icon=FIF.ZOOM_IN, text=self.tr("Preview One"))
        self.btn_preview.clicked.connect(self._on_preview)
        self.btn_process = PrimaryPushButton(icon=FIF.RIGHT_ARROW, text=self.tr("Process Folder"))
        self.btn_process.clicked.connect(self._on_process)

        self.buttons_layout.addWidget(self.btn_preview, stretch=1)
        self.addButtonBarToBottom(self.btn_process)

        self.settings_widget = BasicSettings(self)
        self.settings_drawer.addWidget(self.settings_widget)

        self.help_widget = UpscalerHelp(self)
        self.help_drawer.addWidget(self.help_widget)

        # Internal
        self.current_preview_file: Optional[str] = None
        self.worker: Optional[UpscaleWorker] = None

    # ----------------- UI handlers -----------------
    def _on_include_changed(self, checked: bool):
        self._include_subdirs = checked

    def _on_pick_folder(self):
        base = os.path.expanduser(cfg.get(cfg.convert_dir_cfg))
        path = QFileDialog.getExistingDirectory(self, self.tr("Select Textures Folder"), base)
        if path:
            cfg.set(cfg.convert_dir_cfg, path)
            self.output_card.setContent(path)
            self.folder_card.setContent(path)

    def _on_pick_output(self):
        base = os.path.expanduser(cfg.get(cfg.convert_dir_cfg))
        path = QFileDialog.getExistingDirectory(self, self.tr("Select Output Folder"), base)
        if path:
            self.output_card.setContent(path)

    def _display_on_label(self, image: Image.Image, label: QLabel):
        from PySide6.QtGui import QImage, QPixmap
        img = image.convert('RGBA')
        w, h = img.size
        data = img.tobytes('raw', 'RGBA')
        qimg = QImage(data, w, h, QImage.Format.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg)
        label.setPixmap(pix.scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def _on_preview(self):
        folder = self.folder_card.contentLabel.text()
        if not folder:
            QMessageBox.warning(self, "Validation", "Please select a folder first.")
            return
        file, _ = QFileDialog.getOpenFileName(self, self.tr("Choose a texture"), folder,
                                              "Images (*.png *.jpg *.jpeg *.dds)")
        if not file:
            return
        self.current_preview_file = file
        try:
            pil = load_image(file)
            self._display_on_label(pil, self.original_label)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {e}")
            return
        # Run worker just for this file
        self.worker = UpscaleWorker(folder, output_dir=None, preview_only=file, include_subdirs=self._include_subdirs)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished_preview)
        self.worker.error.connect(self._on_error)
        self.btn_process.setEnabled(False)
        self.btn_preview.setEnabled(False)
        p = self.parent
        if p and hasattr(p, 'show_progress'): p.show_progress()
        self.worker.start()

    def _on_process(self):
        folder = self.folder_card.contentLabel.text()
        if not folder:
            QMessageBox.warning(self, "Validation", "Please select a folder first.")
            return
        out_dir = self.output_card.contentLabel.text() or None
        self.worker = UpscaleWorker(folder, output_dir=out_dir, include_subdirs=self._include_subdirs)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished_batch)
        self.worker.error.connect(self._on_error)
        self.btn_process.setEnabled(False)
        self.btn_preview.setEnabled(False)
        p = self.parent
        if p and hasattr(p, 'show_progress'): p.show_progress()
        self.worker.start()

    def _on_progress(self, i: int, total: int, name: str):
        self.setToolTip(f"Processing {i}/{total}: {name}")

    def _on_finished_preview(self, saved: int, skipped: int, failed: int):
        self.btn_process.setEnabled(True)
        self.btn_preview.setEnabled(True)
        p = self.parent
        if p and hasattr(p, 'complete_loader'): p.complete_loader()
        # Show upscaled preview if available
        if not self.current_preview_file:
            return
        # Expected output is '<stem>_upscaled.png' either next to the file or inside selected output dir
        stem = os.path.splitext(os.path.basename(self.current_preview_file))[0]
        out_path = os.path.join(os.path.dirname(self.current_preview_file), f"{stem}_upscaled.png")
        out_dir = self.output_card.contentLabel.text()
        if not os.path.exists(out_path) and out_dir:
            out_path = os.path.join(out_dir, f"{stem}_upscaled.png")
        if os.path.exists(out_path):
            try:
                img = Image.open(out_path)
                self._display_on_label(img, self.upscaled_label)
            except Exception:
                pass

    def _on_finished_batch(self, saved: int, skipped: int, failed: int):
        self.btn_process.setEnabled(True)
        self.btn_preview.setEnabled(True)
        p = self.parent
        if p and hasattr(p, 'complete_loader'): p.complete_loader()
        QMessageBox.information(self, "Done", f"Saved: {saved}\nSkipped: {skipped}\nFailed: {failed}")

    def _on_error(self, message: str):
        self.btn_process.setEnabled(True)
        self.btn_preview.setEnabled(True)
        p = self.parent
        if p and hasattr(p, 'complete_loader'): p.complete_loader()
        QMessageBox.critical(self, "Error", message)


