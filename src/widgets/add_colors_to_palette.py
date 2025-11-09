import os
import re
from typing import List, Optional, Tuple

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
    FluentIcon as FIF, PushButton, SwitchSettingCard,
)

from src.palette.palette_engine import load_image, next_power_of_2, quantize_image, convert_to_dds
from src.settings.palette_settings import PaletteSettings
from src.utils.appconfig import cfg
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger


class AddColorsWorker(QThread):
    progress = Signal(int, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, base_palette_array: np.ndarray, palette_size: int, pairs: List[Tuple[str, str]], row_height: int):
        super().__init__()
        self.base_palette_array = base_palette_array
        self.palette_size = palette_size
        self.pairs = pairs  # List of (greyscale_path, color_path)
        self.row_height = row_height

    def _load_greyscale_indices(self, path: str) -> np.ndarray:
        pil = load_image(path, format='RGB')
        arr = np.array(pil)
        indices = arr[:, :, 0].astype(np.int32)
        indices = np.clip(indices, 0, self.palette_size - 1)
        return indices

    def run(self):
        try:
            total = max(1, len(self.pairs))
            # combined palette row initialized as unset
            combined_palette = np.zeros((self.palette_size, 3), dtype=np.uint8)
            combined_set = np.zeros((self.palette_size,), dtype=bool)

            for i, (greyscale_path, color_path) in enumerate(self.pairs, start=1):
                try:
                    greyscale_array = self._load_greyscale_indices(greyscale_path)
                    height_g, width_g = greyscale_array.shape

                    # precompute positions for each grey index for this greyscale
                    positions_by_g = {}
                    for g in range(self.palette_size):
                        positions = np.argwhere(greyscale_array == g)
                        positions_by_g[g] = positions

                    # load and quantize the corresponding color image
                    pil_img = load_image(color_path, format='RGB')
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

                    # compute per-index dominant colors for this pair
                    for g in range(self.palette_size):
                        if combined_set[g]:
                            continue  # first wins on shared indices
                        positions = positions_by_g.get(g)
                        if positions is None or positions.size == 0:
                            continue
                        colors = arr_rgb[positions[:, 0], positions[:, 1]]
                        if colors.size == 0:
                            continue
                        tuples = [tuple(c) for c in colors]
                        if not tuples:
                            continue
                        values, counts = np.unique(np.array(tuples), axis=0, return_counts=True)
                        idx = int(np.argmax(counts))
                        combined_palette[g] = values[idx]
                        combined_set[g] = True
                finally:
                    self.progress.emit(i, total)

            # fill remaining indices from base palette
            for g in range(self.palette_size):
                if not combined_set[g]:
                    combined_palette[g] = self.base_palette_array[g]

            # compose palette image: base + single combined row
            num_blocks = 2  # base + combined
            required_height = self.row_height * num_blocks
            palette_height = next_power_of_2(required_height)
            palette_width = self.palette_size

            palette_image_array = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
            all_blocks = [self.base_palette_array, combined_palette]
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
        self.greyscale_paths: List[str] = []
        self.texture_paths: List[str] = []
        self.output_dir: Optional[str] = cfg.get(cfg.convert_output_dir_cfg)

        # data
        self.base_palette_array: Optional[np.ndarray] = None  # (palette_size, 3)
        self.palette_size: int = 256
        self.greyscale_array: Optional[np.ndarray] = None  # indices 0..palette_size-1
        self.generated_palette_image: Optional[Image.Image] = None
        self._save_after_build: bool = False
        self._worker_started: bool = False

        # pickers
        self.palette_card = PushSettingCard(
            self.tr("Select Palette Texture"),
            CustomIcons.PALETTE.icon(),
            self.tr("Select Palette Texture"),
            self.palette_path or "",
        )
        self.palette_card.clicked.connect(self._on_pick_palette)

        self.greyscale_card = PushSettingCard(
            self.tr("Pick Greyscale Texture(s)"),
            CustomIcons.GREYSCALE.icon(),
            self.tr("Pick one or more Greyscale images. We'll match each to a color image by filename (e.g., _greyscale/_grey suffix)."),
            ", ".join(self.greyscale_paths) if self.greyscale_paths else "",
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

        self.replace_existing = SwitchSettingCard(icon=CustomIcons.REPLACE.icon(),
                                                title=self.tr("Replace Existing Palette"),
                                                content=self.tr(
                                                    "Otherwise write to output folder."),
                                                configItem=cfg.ci_replace_existing)


        self.output_dir_card.clicked.connect(self._on_pick_output)

        self.addToFrame(self.palette_card)
        self.addToFrame(self.greyscale_card)
        self.addToFrame(self.textures_card)
        self.addToFrame(self.output_dir_card)
        self.addToFrame(self.replace_existing)

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

        self.clear_textures_card = PushButton(text=self.tr("Clear Textures"))
        self.clear_greyscale_card = PushButton(text=self.tr("Clear Greyscale"))

        # actions
        self.btn_save = PrimaryPushButton(icon=FIF.RIGHT_ARROW, text=self.tr("Run"))
        self.btn_save.clicked.connect(self._on_build_and_save)
        self.buttons_layout.addWidget(self.clear_textures_card)
        self.buttons_layout.addWidget(self.clear_greyscale_card)

        self.addButtonBarToBottom( self.btn_save)

        self.clear_textures_card.clicked.connect(self._on_clear_textures)
        self.clear_greyscale_card.clicked.connect(self._on_clear_greyscales)


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
        files, _ = QFileDialog.getOpenFileNames(self, self.tr("Select Black & White Texture(s)"),
                                                "",
                                                "Images (*.png *.jpg *.jpeg *.dds)")
        if files:
            self.greyscale_paths = files
            self.greyscale_card.setContent(", ".join(files))
            try:
                # show preview of the first greyscale (stretch)
                img = Image.open(files[0]).convert('RGB')
                self._display_on_label(img, self.preview_bw)
            except Exception as e:
                logger.error(f"Failed to load greyscale image: {e}")
                QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to load greyscale texture preview."))

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

    def _on_clear_greyscales(self):
        """Clear selected greyscale image(s) and reset preview label."""
        self.greyscale_paths = []
        self.greyscale_card.setContent("")
        try:
            self.preview_bw.clear()
            self.preview_bw.setText(self.tr("Black & White (indices) Preview"))
        except Exception:
            pass

    def _on_clear_textures(self):
        """Clear selected color texture(s)."""
        self.texture_paths = []
        self.textures_card.setContent("")
    # endregion

    def _on_build(self):
        if not self.palette_path or self.base_palette_array is None:
            QMessageBox.warning(self, self.tr("Missing"), self.tr("Please select a palette texture first."))
            return
        if not self.greyscale_paths:
            QMessageBox.warning(self, self.tr("Missing"), self.tr("Please select the black & white (index map) texture(s)."))
            return
        if not self.texture_paths:
            QMessageBox.warning(self, self.tr("Missing"), self.tr("Please select the color texture(s) to sample from."))
            return
        # enforce 1:1 pairing count
        if len(self.greyscale_paths) != len(self.texture_paths):
            QMessageBox.critical(self, self.tr("Pairing error"), self.tr("Greyscale and Color selections must be 1:1 (same count)."))
            return

        # Build pairs using filename matching
        try:
            pairs = self._pair_greyscale_and_textures(self.greyscale_paths, self.texture_paths)
        except ValueError as ve:
            QMessageBox.critical(self, self.tr("Pairing error"), str(ve))
            return
        except Exception as e:
            logger.error(f"Failed to pair files: {e}", exc_info=True)
            QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to pair greyscale and color images. See log."))
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
                pairs=pairs,
                row_height=row_height,
            )
            self._worker.progress.connect(self._on_worker_progress)
            self._worker.finished.connect(self._on_worker_finished)
            self._worker.error.connect(self._on_worker_error)
            self._worker_started = True
            self._worker.start()
        except Exception as e:
            logger.error(f"Error building palette rows: {e}", exc_info=True)
            QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to build new palette row. See log."))
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
                # If user clicked Run (build+save), save now that the image is ready
                if self._save_after_build:
                    try:
                        self._on_save()
                    finally:
                        self._save_after_build = False
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
            # Ensure we don't attempt to save after a failed build
            self._save_after_build = False
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
        # Defer saving until the async build finishes
        self._save_after_build = True
        # Reset start flag and kick off build
        self._worker_started = False
        self._on_build()
        # If the worker didn't start (due to validation/pairing errors), clear flag and save immediately if an image already exists
        if not self._worker_started:
            self._save_after_build = False
            if self.generated_palette_image is not None:
                self._on_save()

    def _on_save(self):
        if self.generated_palette_image is None:
            QMessageBox.information(self, self.tr("Nothing to save"), self.tr("Please build the palette first."))
            return

        # Determine if we should replace the existing palette instead of writing a new file
        replace_existing = bool(cfg.get(cfg.ci_replace_existing))
        palette_path = self.palette_path or ""
        base_name = os.path.splitext(os.path.basename(palette_path or 'palette.png'))[0]
        source_is_dds = bool(palette_path and palette_path.lower().endswith('.dds'))

        if replace_existing and palette_path:
            # Replace at the original palette path
            out_path = palette_path
            base_dir = os.path.dirname(palette_path)
            try:
                os.makedirs(base_dir, exist_ok=True)
                if source_is_dds:
                    # For DDS originals, generate a temp PNG with the SAME base name so texconv overwrites the DDS
                    temp_png = os.path.join(base_dir, f"{base_name}.png")
                    try:
                        self.generated_palette_image.save(temp_png)
                        logger.info(f"Replacing existing DDS palette via texconv: {temp_png} -> {out_path}")
                        convert_to_dds(
                            temp_png,
                            out_path,
                            is_palette=True,
                            palette_width=self.generated_palette_image.width,
                            palette_height=self.generated_palette_image.height,
                        )
                        logger.info("Replaced existing DDS palette successfully.")
                    finally:
                        try:
                            if os.path.exists(temp_png):
                                os.remove(temp_png)
                        except Exception as _cleanup_ex:
                            logger.warning(f"Failed to remove temp file {temp_png}: {_cleanup_ex}")
                else:
                    logger.info(f"Replacing existing palette PNG at: {out_path}")
                    self.generated_palette_image.save(out_path)
                    logger.info("Replaced existing PNG palette successfully.")
                QMessageBox.information(self, self.tr("Saved"), self.tr(f"Saved: {out_path}"))
            except Exception as e:
                logger.error(f"Failed to replace existing palette: {e}")
                QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to save palette image."))
            return

        # Default: write to output folder with suffix
        base_dir = self.output_dir or (os.path.dirname(palette_path) if palette_path else ".")
        # Decide output extension based on original palette extension
        output_extension = '.dds' if source_is_dds else '.png'
        out_path = os.path.join(base_dir, f"{base_name}_with_added_rows{output_extension}")
        try:
            os.makedirs(base_dir, exist_ok=True)
            if source_is_dds:
                # Save temp PNG then convert to DDS preserving final dimensions
                temp_png = os.path.join(base_dir, f"{base_name}_with_added_rows_temp.png")
                try:
                    self.generated_palette_image.save(temp_png)
                    logger.info(f"Converting generated palette to DDS: {temp_png} -> {out_path}")
                    convert_to_dds(
                        temp_png,
                        out_path,
                        is_palette=True,
                        palette_width=self.generated_palette_image.width,
                        palette_height=self.generated_palette_image.height,
                    )
                    logger.info("Saved generated palette successfully (DDS).")
                finally:
                    try:
                        if os.path.exists(temp_png):
                            os.remove(temp_png)
                    except Exception as _cleanup_ex:
                        logger.warning(f"Failed to remove temp file {temp_png}: {_cleanup_ex}")
            else:
                logger.info(f"Saving generated palette to: {out_path}")
                self.generated_palette_image.save(out_path)
                logger.info("Saved generated palette successfully.")
            QMessageBox.information(self, self.tr("Saved"), self.tr(f"Saved: {out_path}"))
        except Exception as e:
            logger.error(f"Failed to save palette image: {e}")
            QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to save palette image."))

    # helpers
    def _normalize_base_name(self, path: str) -> str:
        """Normalize filename to base key used for pairing.
        Removes suffixes like _greyscale/_grayscale/_grey/_gray/_bw/_mask at the end.
        Case-insensitive. Returns lowercase base name without extension.
        """
        name = os.path.splitext(os.path.basename(path))[0]
        base = name.lower()
        # strip common greyscale suffix patterns possibly repeated
        base = re.sub(r'(?:[\s_\-]?(?:greyscale|grayscale|grey|gray|bw|mask))+$', '', base, flags=re.IGNORECASE)
        # remove all non-alphanumeric to normalize naming like image1 vs image_1
        base = re.sub(r"[^a-z0-9]+", "", base)
        return base

    def _pair_greyscale_and_textures(self, greys: List[str], colors: List[str]) -> List[Tuple[str, str]]:
        """Pair greyscale and color images by normalized base name.
        Enforces 1:1 counts and first-wins policy is handled in the worker when merging.
        Raises ValueError with a helpful message if pairing fails.
        """
        if len(greys) != len(colors):
            raise ValueError(self.tr("Greyscale and Color selections must be 1:1 (same count)."))

        # Map colors by normalized name (ensure uniqueness)
        color_map = {}
        dup_colors = []
        for c in colors:
            key = self._normalize_base_name(c)
            if key in color_map:
                dup_colors.append(os.path.basename(c))
            else:
                color_map[key] = c
        if dup_colors:
            raise ValueError(self.tr("Duplicate color names after normalization: ") + ", ".join(dup_colors))

        pairs: List[Tuple[str, str]] = []
        missing = []
        for g in greys:
            key = self._normalize_base_name(g)
            c = color_map.get(key)
            if not c:
                missing.append(os.path.basename(g))
            else:
                pairs.append((g, c))
        if missing:
            raise ValueError(self.tr("No matching color for greyscale(s): ") + ", ".join(missing))

        return pairs

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
