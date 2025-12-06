import os
from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QWidget, QLabel, QFileDialog
from qfluentwidgets import PushSettingCard, RangeSettingCard
from qfluentwidgets import RangeConfigItem, RangeValidator

from src.utils.dds_utils import load_image
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger
from src.utils.filesystem_utils import get_app_root


class PaletteApplier(BaseWidget):
    """
    A small UI that allows a user to select a palette texture and a greyscale image.
    Shows a single preview of the greyscale image with the selected palette row applied.

    Spinner lets the user pick the pixel row (0..palette_height-1). The range is updated after reading the palette.
    """

    def __init__(self, parent: QWidget | None, text: str = "Palette Applier"):
        super().__init__(parent=parent, text=text, vertical=True)

        # State
        self.palette_path: Optional[str] = None
        # User-selected greyscale image (left preview)
        self.greyscale_path: Optional[str] = None
        self.palette_img: Optional[Image.Image] = None  # RGB
        self.greyscale_img: Optional[Image.Image] = None  # L (8-bit)
        # Fixed reference greyscale (always grayscale_4k_cutout, right preview)
        self.greyscale_ref_path: Optional[str] = None
        self.greyscale_ref_img: Optional[Image.Image] = None  # L (8-bit)

        # Cards
        self.palette_card = PushSettingCard(
            self.tr("Palette Texture"),
            CustomIcons.PALETTE.icon() if hasattr(CustomIcons, 'PALETTE') else CustomIcons.IMAGE.icon(),
            self.tr("Select a color palette image (width = palette size; row applies to greyscale)"),
            self.tr("No palette selected")
        )
        self.palette_card.clicked.connect(self.on_select_palette)

        self.greyscale_card = PushSettingCard(
            self.tr("Greyscale Image"),
            CustomIcons.GREYSCALE.icon() if hasattr(CustomIcons, 'GREYSCALE') else CustomIcons.IMAGE.icon(),
            self.tr("Select a greyscale image to colorize using the palette row"),
            self.tr("No greyscale selected")
        )
        self.greyscale_card.clicked.connect(self.on_select_greyscale)

        # Spinner for pixel row, dynamic range
        self.row_index_cfg = RangeConfigItem("palette_applier", "row_index", 0, RangeValidator(0, 256))
        self.row_card = RangeSettingCard(
            self.row_index_cfg,
            CustomIcons.HEIGHT.icon() if hasattr(CustomIcons, 'HEIGHT') else CustomIcons.SPARK.icon(),
            self.tr("Palette Pixel Row"),
            self.tr("Where to get the colors"),
        )

        self.row_card.valueChanged.connect(self.update_preview)

        # Previews: left = selected greyscale, right = fixed grayscale_4k_cutout reference
        self.preview_left_label = QLabel(self.tr("Selected texture preview"))
        self.preview_left_label.setAlignment(Qt.AlignCenter)
        self.preview_left_label.setMinimumSize(400, 400)

        self.preview_right_label = QLabel(self.tr("grayscale_4k_cutout reference"))
        self.preview_right_label.setAlignment(Qt.AlignCenter)
        self.preview_right_label.setMinimumSize(400, 400)


        # Layout
        self.addToFrame(self.palette_card)
        self.addToFrame(self.greyscale_card)
        self.addToFrame(self.row_card)

        # Two previews side-by-side
        from PySide6.QtWidgets import QHBoxLayout, QWidget as QtWidget

        previews_container = QtWidget(self)
        previews_layout = QHBoxLayout(previews_container)
        previews_layout.setContentsMargins(0, 0, 0, 0)
        previews_layout.setSpacing(12)
        previews_layout.addWidget(self.preview_left_label)
        previews_layout.addWidget(self.preview_right_label)

        self.addToFrame(previews_container)

        # Ensure the fixed reference greyscale is loaded up-front
        self._ensure_default_greyscale_loaded()


    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Rescale previews when the widget is resized
        if (self.preview_left_label.pixmap() is not None) or (self.preview_right_label.pixmap() is not None):
            self.update_preview()

    # -------------- Events --------------
    def on_select_palette(self):
        path, _ = QFileDialog.getOpenFileName(self, self.tr("Select Palette"), "", self.tr("Images (*.png *.jpg *.jpeg *.bmp *.tga *.webp *.dds)"))
        if not path:
            return
        try:
            img = load_image(path)
        except Exception as e:
            logger.exception("Failed to open palette image: %s", e)
            self.palette_card.setContent(self.tr("Failed to open palette"))
            return
        self.palette_path = path
        self.palette_img = img
        w, h = img.size
        self.palette_card.setContent(self.tr(f"{os.path.basename(path)} | {w}x{h}"))
        # Make sure the fixed reference greyscale is available
        self._ensure_default_greyscale_loaded()
        self.update_preview()

    def on_select_greyscale(self):
        path, _ = QFileDialog.getOpenFileName(self, self.tr("Select Greyscale Image"), "", self.tr("Images (*.png *.jpg *.jpeg *.bmp *.tga *.webp *.dds)"))
        if not path:
            return
        try:
            # Load (supports DDS) and convert to L (8-bit)
            img = load_image(path, f='L')
        except Exception as e:
            logger.exception("Failed to open greyscale image: %s", e)
            self.greyscale_card.setContent(self.tr("Failed to open greyscale image"))
            return
        self.greyscale_path = path
        self.greyscale_img = img
        w, h = img.size
        self.greyscale_card.setContent(self.tr(f"{os.path.basename(path)} | {w}x{h} (L)"))
        self.update_preview()
        # Automatically analyze greyscale values and log the report
        try:
            self.on_analyze_greyscale()
        except Exception:
            logger.exception("Greyscale analysis post-select failed")

    # -------------- Helpers --------------
    def _ensure_default_greyscale_loaded(self):
        """Ensure the fixed reference greyscale (grayscale_4k_cutout) is loaded for the right preview."""
        if self.greyscale_ref_img is not None:
            return
        try:
            default_grey_path = os.path.normpath(os.path.join(get_app_root(), 'resource', 'grayscale_4k_cutout.png'))
            if os.path.isfile(default_grey_path):
                img_g = load_image(default_grey_path, f='L')
                self.greyscale_ref_path = default_grey_path
                self.greyscale_ref_img = img_g
                logger.info("Loaded fixed reference greyscale image for palette applier: %s", default_grey_path)
            else:
                logger.warning("Fixed reference greyscale not found at: %s", default_grey_path)
        except Exception as e:
            logger.exception("Failed to load fixed reference greyscale image: %s", e)

    # -------------- Core logic --------------
    def update_preview(self):
        """Update both left (selected greyscale) and right (fixed reference) previews."""
        if self.palette_img is None:
            return

        try:
            row = self.get_selected_palette_row()

            # Left: user-selected greyscale image, if any
            if self.greyscale_img is not None:
                try:
                    colored_left = self.apply_row_to_greyscale(row, self.greyscale_img)
                    self.update_preview_label(self.preview_left_label, colored_left)
                except Exception:
                    logger.exception("Failed to update left preview")

            # Right: fixed grayscale_4k_cutout reference
            if self.greyscale_ref_img is not None:
                try:
                    colored_right = self.apply_row_to_greyscale(row, self.greyscale_ref_img)
                    self.update_preview_label(self.preview_right_label, colored_right)
                except Exception:
                    logger.exception("Failed to update right preview")
        except Exception as e:
            logger.exception("Failed to update preview: %s", e)

    def on_analyze_greyscale(self):
        """Analyze which greyscale values 0-255 are present in the selected greyscale image and log the report."""
        # Analyze the currently selected greyscale image (left preview source)
        if self.greyscale_img is None:
            logger.warning("Greyscale analysis skipped: no greyscale image selected.")
            return
        try:
            g = np.array(self.greyscale_img, dtype=np.uint8)
            flat = g.flatten()
            values, counts = np.unique(flat, return_counts=True)
            total = int(flat.size)

            present_count = int(values.size)
            coverage = present_count / 256.0 * 100.0
            min_v = int(values.min()) if present_count > 0 else 0
            max_v = int(values.max()) if present_count > 0 else 0

            all_vals = np.arange(256, dtype=np.uint16)
            missing = np.setdiff1d(all_vals, values)

            present_ranges = self._format_ranges(values.astype(int)) if present_count > 0 else "(none)"
            missing_ranges = self._format_ranges(missing.astype(int)) if missing.size > 0 else "(none)"

            # Build a concise summary plus a histogram section
            header = [
                f"Greyscale analysis for: {os.path.basename(self.greyscale_path) if self.greyscale_path else '(in-memory)'}",
                f"Pixels: {total:,}",
                f"Unique values: {present_count} / 256 ({coverage:.2f}%)",
                f"Min: {min_v}",
                f"Max: {max_v}",
                f"Present ranges: {present_ranges}",
                f"Missing ranges: {missing_ranges}",
            ]

            # Histogram lines (value,count) for present values only
            hist_lines = ["value,count"] + [f"{int(v)},{int(c)}" for v, c in zip(values, counts)]
            text = "\n".join(header) + "\n\n" + "\n".join(hist_lines)

            logger.info("\n" + ("-" * 80) + "\n" + text + "\n" + ("-" * 80))
        except Exception as e:
            logger.exception("Failed to analyze greyscale values: %s", e)

    @staticmethod
    def _format_ranges(arr: np.ndarray) -> str:
        """Format a sorted array of ints as compact ranges like '0-3,5,7-10'."""
        if arr.size == 0:
            return ""
        arr = np.sort(arr)
        ranges = []
        start = int(arr[0])
        prev = start
        for x in arr[1:]:
            x = int(x)
            if x == prev + 1:
                prev = x
                continue
            # close previous range
            if start == prev:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{prev}")
            # start new
            start = prev = x
        # close last range
        if start == prev:
            ranges.append(f"{start}")
        else:
            ranges.append(f"{start}-{prev}")
        return ", ".join(ranges)

    def get_selected_palette_row(self) -> np.ndarray:
        assert self.palette_img is not None
        w, h = self.palette_img.size
        y = 0
        if self.row_card:
            y = int(self.row_card.configItem.value)
        else:
            # Fallback to config value
            try:
                y = int(self.row_index_cfg.value)
            except Exception:
                y = 0
        y = max(0, min(h - 1, y))
        # Extract row as numpy array shape (w, 3)
        row_pixels = np.array(self.palette_img)[y, :, :3]
        if row_pixels.ndim == 1:
            row_pixels = np.expand_dims(row_pixels, axis=0)
        return row_pixels.astype(np.uint8)

    @staticmethod
    def apply_row_to_greyscale(palette_row: np.ndarray, grey_img: Image.Image) -> Image.Image:
        # Map grey 0..255 to indices 0..(palette_width-1)
        pw = palette_row.shape[0]
        # Build lookup of 256 x 3
        if pw == 256:
            lut = palette_row
        else:
            # Interpolate along the row to 256 entries
            x = np.linspace(0, pw - 1, num=pw)
            xi = np.linspace(0, pw - 1, num=256)
            # Interpolate each channel
            lut = np.stack([
                np.interp(xi, x, palette_row[:, c]).astype(np.uint8) for c in range(3)
            ], axis=1)
        # Apply LUT
        g = np.array(grey_img, dtype=np.uint8)
        colored = lut[g]  # shape (H, W, 3)
        return Image.fromarray(colored, mode='RGB')

    def update_preview_label(self, label: QLabel, pil_image: Image.Image):
        """Fit image into the given label while keeping aspect ratio."""
        if label is None:
            return
        qimg = self.pil_to_qimage(pil_image)
        pix = QPixmap.fromImage(qimg)
        # Scale to label size
        target_w = max(1, label.width())
        target_h = max(1, label.height())
        pix = pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pix)

    @staticmethod
    def pil_to_qimage(img: Image.Image) -> QImage:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        arr = np.array(img)
        h, w, _ = arr.shape
        return QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
