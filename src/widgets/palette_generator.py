import os

import numpy as np
from PIL import Image
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QThread
from PySide6.QtCore import Signal
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (QFileDialog, QWidget, QMessageBox)
from qfluentwidgets import (
    PushSettingCard,
    ConfigItem,
    FluentIcon as FIF,
    PrimaryPushButton
)

from src.help.palette_help import PaletteHelp
from src.palette.palette_engine import perceptual_color_sort, \
    map_rgb_array_to_palette_indices, \
    compose_palette_image, adjacency_aware_color_sort_pmode
from src.settings.palette_settings import PaletteSettings
from src.utils.appconfig import cfg
from src.utils.dds_utils import save_image, load_image
from src.utils.filesystem_utils import get_app_root
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger
from src.utils.palette_utils import quantize_image, apply_palette_to_greyscale, get_palette


class SinglePaletteGenerationWorker(QThread):
    """Worker thread for generating a perfect Palette for a single image"""
    progress_updated = Signal(int, str)
    error_occurred = Signal(str)
    step_complete = Signal(str, object)  # step_name, result_data

    def __init__(self, parent, image_path, output_dir,
                 working_resolution=None,
                 palette_row_height=4):
        super().__init__()
        self.parent = parent
        self.image_path = image_path
        self.output_dir = output_dir
        # previous_data is deprecated; all intermediates are local to run()
        self.working_resolution = working_resolution  # None for Original, else max side target (e.g., 4096)
        self.palette_row_height = palette_row_height
        # sRGB compensation removed from UI/logic; keep attribute for backward-compatibility in save paths
        self.srgb_compensation = False


    def run(self):
        """Run the full Palette generation pipeline in one go, in a single method.
        This fully inlines the previous step-based logic to satisfy the requirement that
        all steps are executed inside one method without delegating to sub-steps.
        """
        try:
            logger.debug("Starting Palette generation (single-pass pipeline, fully inlined)")

            # =====================
            # 1) QUANTIZATION STEP
            # =====================
            self.progress_updated.emit(10, "Loading image...")

            def downscale_keep_aspect(img, max_side):
                if max_side is None:
                    return img
                w, h = img.size
                current_max = max(w, h)
                if current_max <= max_side:
                    return img
                scale = max_side / float(current_max)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            palette_size = cfg.get(cfg.ci_default_palette_size)
            img = load_image(self.image_path)
            base_original_img = img
            base_orig_w, base_orig_h = base_original_img.size
            # Apply working resolution downscale to base (no upscale)
            if self.working_resolution:
                original_img = downscale_keep_aspect(base_original_img, self.working_resolution)
            else:
                original_img = base_original_img

            original_array = np.array(original_img)
            height, width = original_array.shape[:2]

            logger.debug(f"Image loaded (processed): {width}x{height} (original {base_orig_w}x{base_orig_h})")

            self.progress_updated.emit(30, f"Quantizing using {cfg.get(cfg.ci_default_quant_method)}...")

            # Quantize to exactly Palette_SIZE colors using selected method
            quantized = quantize_image(original_img, cfg.get(cfg.ci_default_quant_method))
            # Work with palette indices and palette colors (avoid early RGB convert)
            quantized_indices = np.array(quantized, dtype=np.uint8)
            palette_colors = get_palette(quantized)
            used_colors = int(len(palette_colors))
            # For algorithms that expect RGB arrays, materialize an RGB view via palette lookup
            quantized_array = palette_colors[quantized_indices]
            logger.debug(f"Quantization complete: {used_colors} unique colors")

            # =====================
            # 2) GREYSCALE STEP
            # =====================
            self.progress_updated.emit(10, "Preparing greyscale conversion...")


            self.progress_updated.emit(20, "Sorting colors and assigning greyscale values...")

            color_tuples = [tuple(color) for color in palette_colors]
            try:
                sorted_colors = perceptual_color_sort(color_tuples)
            except Exception as _:
                sorted_colors = adjacency_aware_color_sort_pmode(quantized)

            color_to_grey = {}
            grey_to_color = {}
            for grey_value, color in enumerate(sorted_colors):
                color_to_grey[color] = grey_value
                grey_to_color[grey_value] = np.array(color, dtype=np.uint8)

            logger.debug("Color to greyscale mapping created")
            self.progress_updated.emit(60, "Creating greyscale image...")

            lut_exact = {tuple(map(int, c)): int(i) for i, c in enumerate(sorted_colors)}
            pal_int16 = np.array(sorted_colors, dtype=np.uint8).astype(np.int16)
            greyscale_array = map_rgb_array_to_palette_indices(quantized_array, lut_exact, pal_int16).astype(np.uint8)

            if used_colors > 1:
                scale = 255.0 / float(used_colors - 1)
            else:
                scale = 0.0
            disp = (greyscale_array.astype(np.float32) * scale).astype(np.uint8)
            # Always produce single-channel greyscale without transparency masking
            greyscale_rgb = Image.fromarray(disp, 'L')

            # Greyscale texture mapping removed
            self.progress_updated.emit(100, "Greyscale conversion complete!")

            # =====================
            # 3) PALETTE STEP
            # =====================
            self.progress_updated.emit(10, "Creating Palette...")

            # Helper to find the closest power of two to n
            def closest_power_of_two(n: int) -> int:
                if n <= 1:
                    return 1
                # lower and upper powers of two around n
                up = 1 << (int(n - 1).bit_length())
                low = up >> 1
                # choose the closest; if tie, prefer the upper power (commonly desired for textures)
                if (n - low) <= (up - n):
                    return low if low > 0 else 1
                return up

            # Resample (expand or shrink) a palette row to the target length using linear interpolation
            def resample_palette_row(colors_np: np.ndarray, target: int) -> np.ndarray:
                n = int(colors_np.shape[0])
                if n == target:
                    return colors_np.astype(np.uint8)
                if target <= 0:
                    return colors_np[:0].astype(np.uint8)
                if n == 0:
                    return np.zeros((target, 3), dtype=np.uint8)
                if n == 1:
                    # replicate the single color
                    return np.tile(colors_np[:1], (target, 1)).astype(np.uint8)
                x = np.linspace(0, n - 1, num=n)
                xi = np.linspace(0, n - 1, num=target)
                out = np.stack([
                    np.clip(np.interp(xi, x, colors_np[:, c]), 0, 255) for c in range(3)
                ], axis=1)
                return np.rint(out).astype(np.uint8)

            # Build base palette row from actually used/sorted colors
            base_palette_array = np.array(sorted_colors, dtype=np.uint8)

            # If the quantized palette has fewer colors than requested, expand to the nearest power of two of the actual used size
            target_palette_size = int(palette_size)
            if used_colors < palette_size:
                target_palette_size = closest_power_of_two(palette_size)
                logger.info(
                    f"Quantizer returned {used_colors} < requested {palette_size}; expanding linearly to closest power-of-two of requested: {target_palette_size}")
            # Resample to target size if needed (will also downsample when used_colors > target)
            base_palette_array = resample_palette_row(base_palette_array, target_palette_size)

            # Compose palette image
            palette_image = compose_palette_image(
                rows=[base_palette_array],
                row_height=self.palette_row_height,
                palette_size=target_palette_size,
                pad_mode='gradient'
            )

            self.progress_updated.emit(50, "Applying Palette to greyscale for preview...")
            preview_image = apply_palette_to_greyscale(palette_image, greyscale_rgb)

            # Color report generation removed

            self.progress_updated.emit(100, "Palette creation complete!")

            # =====================
            # 4) SAVE RESULTS
            # =====================
            # Inline save_all_results logic using local variables
            try:
                logger.debug("Saving all results")
                self.progress_updated.emit(90, "Saving files...")

                # Simplified base name (no method/size suffixes)
                base_name = os.path.splitext(os.path.basename(self.image_path))[0]

                # Determine output format based on input format
                source_is_dds = self.image_path.lower().endswith('.dds')
                output_extension = ".dds" if source_is_dds else ".png"

                output_files = {}

                # Save core outputs

                quantized_path = os.path.join(self.output_dir, f"{base_name}_quantized{output_extension}")
                save_image(quantized.convert("RGB"), quantized_path)

                greyscale_path = os.path.join(self.output_dir, f"{base_name}_greyscale{output_extension}")
                save_image(greyscale_rgb, greyscale_path)
                output_files['greyscale'] = greyscale_path
                logger.debug(f"Saved greyscale image: {greyscale_path}")

                palette_path = os.path.join(self.output_dir, f"{base_name}_palette{output_extension}")
                save_image(palette_image, palette_path)
                output_files['palette'] = palette_path
                logger.debug(f"Saved Palette: {palette_path}")

                # Saving of additional greyscale-conversion textures removed

                # Color report saving removed

                pixmap = self.parent.pil_to_pixmap(quantized.convert("RGB"))
                self.parent.update_preview(pixmap, self.parent.quantized_preview_label)

                pixmap = self.parent.pil_to_pixmap(greyscale_rgb)
                self.parent.update_preview(pixmap, self.parent.greyscale_preview_label)

                pixmap = self.parent.pil_to_pixmap(preview_image)
                self.parent.update_preview(pixmap, self.parent.preview_label)

                default_grey_path = os.path.normpath(os.path.join(get_app_root(), 'resource', 'grayscale_4k_cutout.png'))
                pixmap = self.parent.pil_to_pixmap(apply_palette_to_greyscale(palette_image, load_image(default_grey_path, f='L')))
                self.parent.update_preview(pixmap, self.parent.debug_preview_label)

                self.progress_updated.emit(100, "Complete!")
                logger.debug("All results saved successfully")
            except Exception as e:
                logger.error(f"Error saving results: {str(e)}", exc_info=True)
                raise

        except Exception as e:
            logger.error(f"Error in Palette generation: {str(e)}", exc_info=True)
            self.error_occurred.emit(str(e))


class PaletteGenerator(BaseWidget):
    log_signal = Signal(str)

    def __init__(self, parent, text):
        super().__init__(parent=parent, text=text, vertical=True)
        self.main_widget = QWidget()
        # Persistent settings via qfluentwidgets ConfigItem
        self.ci_last_image_dir = ConfigItem("palette", "last_image_dir", "")
        self.ci_last_output_dir = ConfigItem("palette", "last_output_dir", "")
        self.current_image_path = None
        self.output_dir = None
        self.current_results = None
        # Per-step cached data removed: pipeline runs and completes in one worker call
        self.original_preview_label = None
        self.greyscale_preview_label = None
        self.debug_preview_label = None
        self.quantized_preview_label = None
        self.preview_label = None
        self.init_ui()
        self.addButtonBarToBottom(self.generate_all_button)
        self.settings_widget = PaletteSettings(self)
        self.settings_drawer.addWidget(self.settings_widget)
        self.help_widget = PaletteHelp(self)
        self.help_drawer.addWidget(self.help_widget)


    def init_ui(self):
        # --- Image Selection Cards ---
        self.base_image_card = PushSettingCard(
            self.tr("Base Image"),
            CustomIcons.IMAGE.icon(stroke=True),
            self.tr("Source Image for Greyscale Template / Palette"),
            "No image selected"
        )
        self.base_image_card.clicked.connect(self.on_base_image_card)
        self.addToFrame(self.base_image_card)

        self.output_dir_card = PushSettingCard(
            self.tr("Output Directory"),
            CustomIcons.FOLDERRIGHT.icon(),
            self.tr("Where generated files will be written. Defaults to the image folder."),
            self.tr("Will use image directory")
        )
        self.output_dir_card.clicked.connect(self.on_output_dir_card)
        self.addToFrame(self.output_dir_card)

        # Generate button
        self.generate_all_button = PrimaryPushButton(icon=FIF.RIGHT_ARROW, text="Generate")
        self.generate_all_button.clicked.connect(self.generate_step)
        self.generate_all_button.setEnabled(False)

        # Preview area (4 previews horizontally)
        preview_splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)

        def make_preview_group(title, attr_name):
            group = QtWidgets.QGroupBox(title)
            group_layout = QtWidgets.QVBoxLayout(group)
            label = QtWidgets.QLabel(f"{title}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(300, 300)
            group_layout.addWidget(label)
            setattr(self, attr_name, label)
            return group

        # Updated previews: Quantized, Greyscale, Preview on Greyscale, Preview on Debug Texture
        preview_splitter.addWidget(make_preview_group("Quantized", "quantized_preview_label"))
        preview_splitter.addWidget(make_preview_group("Greyscale (Color → Grey Mapping)", "greyscale_preview_label"))
        preview_splitter.addWidget(make_preview_group("Preview (Palette Applied to Greyscale)", "preview_label"))
        preview_splitter.addWidget(make_preview_group("Preview (Palette Applied to Debug Texture)", "debug_preview_label"))

        preview_splitter.setSizes([400, 400, 400, 400])
        self.addToFrame(preview_splitter)

    def on_base_image_card(self):
        last_dir = str(self.ci_last_image_dir.value or "")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Image for Palette Generation"),
            last_dir,
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.dds *.tga);;All Files (*)"
        )
        if file_path:
            logger.debug(f"Image selected: {file_path}")
            self.current_image_path = file_path
            try:
                self.ci_last_image_dir.value = os.path.dirname(file_path)
            except Exception:
                pass
            # Default output to image directory if not explicitly set or previously unset
            self.output_dir = os.path.dirname(file_path)
            try:
                self.base_image_card.setContent(os.path.basename(file_path))
            except Exception:
                pass
            try:
                # If user hasn't chosen a custom output yet, show default
                self.output_dir_card.setContent(f"{self.tr('Using')}: {self.output_dir}")
            except Exception:
                pass

            self.update_button_states()

    def on_output_dir_card(self):
        last_dir = str(self.ci_last_output_dir.value or (self.output_dir or ""))
        directory = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Output Directory"),
            last_dir,
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.output_dir = directory
            try:
                self.ci_last_output_dir.value = directory
            except Exception:
                pass
            try:
                self.output_dir_card.setContent(directory)
            except Exception:
                pass

    def load_and_display_image(self, file_path, label, description):
        try:
            logger.debug(f"Loading image for display: {file_path}")
            img = load_image(file_path)
            pixmap = self.pil_to_pixmap(img)

            scaled_pixmap = pixmap.scaled(
                label.width() - 20,
                label.height() - 20,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
            label.setText("")
            logger.debug(f"Image displayed successfully: {description}")

        except Exception as e:
            logger.error(f"Error loading {description}: {str(e)}")
            label.setText(f"Error loading {description}: {e}")

    def pil_to_pixmap(self, pil_image):
        """Convert PIL Image to QPixmap"""
        try:
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            data = pil_image.tobytes("raw", "RGB")
            qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(qimage)
        except Exception as e:
            logger.error(f"Error converting PIL to QPixmap: {str(e)}")
            # Return a blank pixmap on error
            return QPixmap(100, 100)

    def generate_step(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "Please select an image first.")
            return

        if not self.output_dir:
            self.output_dir = os.path.dirname(self.current_image_path)
            try:
                self.output_dir_card.setContent(self.output_dir)
            except Exception:
                pass

        # Get working resolution from ConfigItem
        wr_val = cfg.get(cfg.ci_default_working_res).value
        if isinstance(wr_val, str) and str(wr_val).lower().startswith('original'):
            working_resolution = None
        else:
            try:
                working_resolution = int(wr_val)
            except Exception:
                working_resolution = None

        # Disable buttons during processing and use parent mask progress
        self.set_buttons_enabled(False)
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'show_progress'):
            try:
                p.show_progress()
            except Exception:
                pass

        self.worker = SinglePaletteGenerationWorker(
            self,
            self.current_image_path,
            self.output_dir,
            working_resolution=working_resolution,
            palette_row_height=int(cfg.get(cfg.ci_palette_row_height) or 4)
        )

        self.worker.progress_updated.connect(self.update_progress)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()

    def update_progress(self, value, message):
        p = getattr(self, 'parent', None)
        if value == 100:
            self.set_buttons_enabled(True)
            if p and hasattr(p, 'complete_loader'):
                try:
                    p.complete_loader()
                except Exception:
                    pass
        else:
            if p and hasattr(p, 'update_progress'):
                try:
                    p.update_progress(int(value))
                except Exception:
                    pass
        logger.info(f"Progress: {value}% - {message}")

    def handle_error(self, error_message):
        self.set_buttons_enabled(True)
        logger.error(f"Worker error: {error_message}")
        QMessageBox.critical(self, "Error", f"Palette generation failed: {error_message}")
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'complete_loader'):
            try:
                p.complete_loader()
            except Exception:
                pass

    def update_preview(self, pixmap, label):
        scaled_pixmap = pixmap.scaled(
            label.width() - 20,
            label.height() - 20,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)
        label.setText("")

    def set_buttons_enabled(self, enabled):
        """Enable or disable interactive controls during processing"""
        # Main action
        self.generate_all_button.setEnabled(enabled and self.current_image_path is not None)
        # Setting cards and combos
        try:
            self.base_image_card.setEnabled(enabled)
            self.output_dir_card.setEnabled(enabled)
        except Exception:
            pass

    def update_button_states(self):
        """Update button states based on available data (single-button workflow)"""
        self.generate_all_button.setEnabled(self.current_image_path is not None)
