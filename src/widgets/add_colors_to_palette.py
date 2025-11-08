import os
from typing import List, Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QLabel, QFileDialog, QGridLayout, QMessageBox
)
from qfluentwidgets import (
    PushSettingCard,
    PrimaryPushButton,
    FluentIcon as FIF,
)

from src.palette.palette_engine import load_image, next_power_of_2, quantize_image
from src.settings.palette_settings import PaletteSettings
from src.utils.appconfig import cfg
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger


class AddColorsWorker(QThread):
    progress = Signal(int, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, base_palette_array: np.ndarray, palette_size: int, greyscale_array: np.ndarray, texture_paths: List[str], row_height: int):
        super().__init__()
        self.base_palette_array = base_palette_array
        self.palette_size = palette_size
        self.greyscale_array = greyscale_array
        self.texture_paths = texture_paths
        self.row_height = row_height

    def run(self):
        try:
            height_g, width_g = self.greyscale_array.shape
            # precompute positions for each grey index
            positions_by_g = {}
            for g in range(self.palette_size):
                positions = np.argwhere(self.greyscale_array == g)
                positions_by_g[g] = positions

            total = max(1, len(self.texture_paths))
            extra_arrays = []
            for i, path in enumerate(self.texture_paths, start=1):
                try:
                    pil_img = load_image(path, format='RGB')
                    ex_quantized, _ = quantize_image(pil_img, cfg.get(cfg.ci_default_quant_method))
                    ex_quant_rgb = ex_quantized.convert('RGB')
                    arr_rgb = np.array(ex_quant_rgb)
                    if arr_rgb.shape[:2] != (height_g, width_g):
                        try:
                            img = Image.fromarray(arr_rgb.astype('uint8'), 'RGB')
                            img = img.resize((width_g, height_g), Image.Resampling.NEAREST)
                            arr_rgb = np.array(img)
                        except Exception:
                            pass
                    palette_for_extra = np.zeros((self.palette_size, 3), dtype=np.uint8)
                    for g in range(self.palette_size):
                        positions = positions_by_g.get(g)
                        if positions is None or positions.size == 0:
                            palette_for_extra[g] = self.base_palette_array[g]
                            continue
                        colors = arr_rgb[positions[:, 0], positions[:, 1]]
                        if colors.size == 0:
                            palette_for_extra[g] = self.base_palette_array[g]
                            continue
                        tuples = [tuple(c) for c in colors]
                        if not tuples:
                            palette_for_extra[g] = self.base_palette_array[g]
                        else:
                            values, counts = np.unique(np.array(tuples), axis=0, return_counts=True)
                            idx = int(np.argmax(counts))
                            palette_for_extra[g] = values[idx]
                    extra_arrays.append(palette_for_extra)
                finally:
                    self.progress.emit(i, total)
            # compose palette image: base + extras
            num_blocks = 1 + len(extra_arrays)
            required_height = self.row_height * num_blocks
            palette_height = next_power_of_2(required_height)
            palette_width = self.palette_size

            palette_image_array = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
            all_blocks = [self.base_palette_array] + extra_arrays
            for block_idx, block_array in enumerate(all_blocks):
                start_row = block_idx * self.row_height
                end_row = start_row + self.row_height
                for row in range(start_row, end_row):
                    for col in range(palette_width):
                        if col < self.palette_size:
                            palette_image_array[row, col] = block_array[col]
                        else:
                            palette_image_array[row, col] = [0, 0, 0]
            # fill remaining rows with grey pattern
            filled_rows = self.row_height * num_blocks
            if filled_rows < palette_height:
                for row in range(filled_rows, palette_height):
                    for col in range(palette_width):
                        grey_value = int(col * (255 / (palette_width - 1))) if palette_width > 1 else 0
                        palette_image_array[row, col] = [grey_value, grey_value, grey_value]
            out_image = Image.fromarray(palette_image_array.astype('uint8'), 'RGB')
            self.finished.emit(out_image)
        except Exception as e:
            self.error.emit(str(e))


class AddColorsToPaletteWidget(BaseWidget):
    """
    UI: "Add Colors To Pallete"
    - Pick existing palette image
    - Pick black & white (index map) texture
    - Pick one or more color textures to sample from
    For each selected texture, create a new palette row using dominant color per grey index
    (fallback to original palette color when index doesn't exist in that texture at any pixel),
    mirroring logic used by palette_generator additional rows.
    """

    def __init__(self, parent: Optional[QWidget], text: str):
        super().__init__(text, parent, True)
        self.setObjectName('AddColorsToPaletteWidget')

        # inputs
        self.palette_path: Optional[str] = cfg.get(cfg.base_palette_cfg)
        self.greyscale_path: Optional[str] = None
        self.texture_paths: List[str] = []
        self.output_dir: Optional[str] = cfg.get(cfg.convert_output_dir_cfg)

        # data
        self.base_palette_array: Optional[np.ndarray] = None  # (palette_size, 3)
        self.palette_size: int = 256
        self.greyscale_array: Optional[np.ndarray] = None  # indices 0..palette_size-1
        self.generated_palette_image: Optional[Image.Image] = None

        # pickers
        self.palette_card = PushSettingCard(
            self.tr("Select Palette Texture"),
            CustomIcons.PALETTE.icon(),
            self.tr("Select Palette Texture"),
            self.palette_path or "",
        )
        self.palette_card.clicked.connect(self._on_pick_palette)

        self.greyscale_card = PushSettingCard(
            self.tr("Pick Greyscale Texture"),
            CustomIcons.GREYSCALE.icon(),
            self.tr("Pick Greyscale Texture. We use it for color coordinates to make the new palette rows."),
            self.greyscale_path or "",
        )
        self.greyscale_card.clicked.connect(self._on_pick_greyscale)

        self.textures_card = PushSettingCard(
            self.tr("Choose Source Texture(s) ..."),
            CustomIcons.IMAGEADD.icon(stroke=True),
            self.tr("Pick textures that MATCH the Greyscale. (eg: greyscale is receiver, this should be receiver just recolored)"),
            ", ".join(self.texture_paths) if self.texture_paths else "",
        )
        self.textures_card.clicked.connect(self._on_pick_textures)

        self.output_dir_card = PushSettingCard(
            self.tr("Output Folder (optional)"),
            FIF.SAVE.icon(),
            self.tr("Select output folder for generated palette (default: same folder as palette)"),
            self.output_dir or "",
        )
        self.output_dir_card.clicked.connect(self._on_pick_output)

        self.addToFrame(self.palette_card)
        self.addToFrame(self.greyscale_card)
        self.addToFrame(self.textures_card)
        self.addToFrame(self.output_dir_card)

        # previews
        self.preview_bw = QLabel(self.tr("Black & White (indices) Preview"))
        self.preview_bw.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_bw.setMinimumSize(360, 360)
        self.preview_bw.setStyleSheet("border: 1px dashed gray;")

        self.preview_palette = QLabel(self.tr("Generated Palette Preview"))
        self.preview_palette.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_palette.setMinimumSize(360, 360)
        self.preview_palette.setStyleSheet("border: 1px dashed gray;")

        grid = QGridLayout()
        grid.addWidget(QLabel(self.tr("B&W")), 0, 0)
        grid.addWidget(QLabel(self.tr("Palette")), 0, 1)
        grid.addWidget(self.preview_bw, 1, 0)
        grid.addWidget(self.preview_palette, 1, 1)
        grid_container = QWidget()
        grid_container.setLayout(grid)
        self.boxLayout.addStretch(1)
        self.addToFrame(grid_container)

        # actions
        self.btn_save = PrimaryPushButton(icon=FIF.RIGHT_ARROW, text=self.tr("Run"))
        self.btn_save.clicked.connect(self._on_build_and_save)

        self.addButtonBarToBottom( self.btn_save)

        # settings and help
        self.settings_widget = PaletteSettings(self)
        self.settings_drawer.addWidget(self.settings_widget)

    # region pickers
    def _on_pick_palette(self):
        file, _ = QFileDialog.getOpenFileName(self, self.tr("Select Palette Texture"),
                                              self.palette_path or "",
                                              "Images (*.png *.jpg *.jpeg *.dds)")
        if file:
            self.palette_path = file
            cfg.set(cfg.base_palette_cfg, file)
            self.palette_card.setContent(file)
            try:
                self.base_palette_array, self.palette_size = self._extract_base_palette(file)
            except Exception as e:
                logger.error(f"Failed to parse palette: {e}")
                QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to parse palette texture."))

    def _on_pick_greyscale(self):
        file, _ = QFileDialog.getOpenFileName(self, self.tr("Select Black & White Texture"),
                                              self.greyscale_path or "",
                                              "Images (*.png *.jpg *.jpeg *.dds)")
        if file:
            self.greyscale_path = file
            self.greyscale_card.setContent(file)
            try:
                self.greyscale_array = self._load_greyscale_indices(file)
                # show preview (stretch)
                img = Image.open(file).convert('RGB')
                self._display_on_label(img, self.preview_bw)
            except Exception as e:
                logger.error(f"Failed to load greyscale image: {e}")
                QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to load greyscale texture."))

    def _on_pick_textures(self):
        files, _ = QFileDialog.getOpenFileNames(self, self.tr("Select Texture(s)"),
                                                "",
                                                "Images (*.png *.jpg *.jpeg *.dds)")
        if files:
            self.texture_paths = files
            self.textures_card.setContent(", ".join(files))

    def _on_pick_output(self):
        dir_ = QFileDialog.getExistingDirectory(self, self.tr("Select Output Folder"), self.output_dir or "")
        if dir_:
            self.output_dir = dir_
            cfg.set(cfg.convert_output_dir_cfg, dir_)
            self.output_dir_card.setContent(dir_)
    # endregion

    def _on_build(self):
        if not self.palette_path or self.base_palette_array is None:
            QMessageBox.warning(self, self.tr("Missing"), self.tr("Please select a palette texture first."))
            return
        if self.greyscale_array is None:
            QMessageBox.warning(self, self.tr("Missing"), self.tr("Please select the black & white (index map) texture."))
            return
        if not self.texture_paths:
            QMessageBox.warning(self, self.tr("Missing"), self.tr("Please select one or more textures to sample from."))
            return

        # Show parent mask progress
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'show_progress'):
            try:
                p.show_progress()
            except Exception:
                pass
        try:
            self.btn_save.setEnabled(False)
            row_height = int(cfg.get(cfg.ci_palette_row_height))
            self._worker = AddColorsWorker(
                base_palette_array=self.base_palette_array,
                palette_size=self.palette_size,
                greyscale_array=self.greyscale_array,
                texture_paths=self.texture_paths,
                row_height=row_height,
            )
            self._worker.progress.connect(self._on_worker_progress)
            self._worker.finished.connect(self._on_worker_finished)
            self._worker.error.connect(self._on_worker_error)
            self._worker.start()
        except Exception as e:
            logger.error(f"Error building palette rows: {e}", exc_info=True)
            QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to build new palette rows. See log."))
            # Complete mask
            p = getattr(self, 'parent', None)
            if p and hasattr(p, 'complete_loader'):
                try:
                    p.complete_loader()
                except Exception:
                    pass

    def _on_worker_progress(self, i: int, total: int):
        try:
            if total:
                percent = int(max(0, min(100, round((i / total) * 100))))
                p = getattr(self, 'parent', None)
                if p and hasattr(p, 'update_progress'):
                    p.update_progress(percent)
        except Exception:
            pass

    def _on_worker_finished(self, image):
        try:
            self.generated_palette_image = image
            if self.generated_palette_image is not None:
                self._display_on_label(self.generated_palette_image, self.preview_palette)
        finally:
            self.btn_save.setEnabled(True)
            p = getattr(self, 'parent', None)
            if p and hasattr(p, 'complete_loader'):
                try:
                    p.complete_loader()
                except Exception:
                    pass

    def _on_worker_error(self, message: str):
        try:
            QMessageBox.critical(self, self.tr("Error"), self.tr(f"Failed to build palette rows: {message}"))
        finally:
            self.btn_save.setEnabled(True)
            p = getattr(self, 'parent', None)
            if p and hasattr(p, 'complete_loader'):
                try:
                    p.complete_loader()
                except Exception:
                    pass

    def _on_build_and_save(self):
        """Build the palette rows and save in one click."""
        # Preserve reference to detect whether build produced a new image
        prev_image = self.generated_palette_image
        self._on_build()
        # Only save if a new image was generated
        if self.generated_palette_image is not None and self.generated_palette_image is not prev_image:
            self._on_save()

    def _on_save(self):
        if self.generated_palette_image is None:
            QMessageBox.information(self, self.tr("Nothing to save"), self.tr("Please build the palette first."))
            return
        base_dir = self.output_dir or (os.path.dirname(self.palette_path) if self.palette_path else ".")
        base_name = os.path.splitext(os.path.basename(self.palette_path or 'palette.png'))[0]
        out_path = os.path.join(base_dir, f"{base_name}_with_added_rows.png")
        try:
            os.makedirs(base_dir, exist_ok=True)
            self.generated_palette_image.save(out_path)
            QMessageBox.information(self, self.tr("Saved"), self.tr(f"Saved: {out_path}"))
        except Exception as e:
            logger.error(f"Failed to save palette image: {e}")
            QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to save palette image."))

    # helpers
    def _extract_base_palette(self, path: str) -> tuple[np.ndarray, int]:
        """Parse the base palette colors from an existing palette texture.
        Strategy: locate the first non-greyscale row and read colors across width.
        """
        pil = load_image(path, format='RGB')
        arr = np.array(pil)
        h, w = arr.shape[:2]
        chosen_row = None
        for y in range(h):
            row = arr[y, :, :]
            # check if this row is mostly greyscale gradient: many pixels have R==G==B and values change smoothly
            eq = (row[:, 0] == row[:, 1]) & (row[:, 1] == row[:, 2])
            frac_grey = float(np.mean(eq))
            if frac_grey < 0.9:  # likely a color row
                chosen_row = y
                break
        if chosen_row is None:
            # fallback to middle row
            chosen_row = h // 2
        colors = arr[chosen_row, :w, :]
        palette_size = w
        base_palette = colors[:palette_size].astype(np.uint8)
        return base_palette, palette_size

    def _load_greyscale_indices(self, path: str) -> np.ndarray:
        """Load greyscale image and return 2D array of indices 0..palette_size-1 using the red channel."""
        pil = load_image(path, format='RGB')
        arr = np.array(pil)
        # use first channel as index, cap to palette_size
        indices = arr[:, :, 0].astype(np.int32)
        indices = np.clip(indices, 0, self.palette_size - 1)
        return indices

    def _display_on_label(self, pil_image: Image.Image, label: QLabel):
        # maintain aspect fit
        qimage = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, pil_image.width * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
