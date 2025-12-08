import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QImage, QPainter, QPixmap, QColor, QMouseEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QSplitter,
    QCheckBox, QColorDialog, QMessageBox
)
from qfluentwidgets import (
    PushSettingCard, PrimaryPushButton, PushButton, FluentIcon as FIF,
    RangeSettingCard, RangeConfigItem, RangeValidator
)

from src.settings.palette_settings import PaletteSettings
from src.utils.appconfig import cfg
from src.utils.dds_utils import load_image, save_image
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger
from src.utils.palette_utils import apply_palette_to_greyscale, get_palette_row


@dataclass
class AdjustmentState:
    hue: int = 0            # -180..180
    saturation: int = 0     # -100..100 (percent)
    value: int = 0          # -100..100 (percent)
    brightness: int = 0     # -100..100 (percent)
    contrast: int = 0       # -100..100 (percent)


class PaletteAdjustCanvas(QLabel):
    """Simple canvas to display a palette and allow rectangular selection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 200)
        self.setStyleSheet("background-color: #1f1f1f; border: 1px solid #444;")
        self._image_array: Optional[np.ndarray] = None
        self._selection_mask: Optional[np.ndarray] = None
        self._selection_rect: Optional[QRect] = None
        self._dragging = False
        self._scale_factor = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._cached_scaled_size: Tuple[int, int] | None = None
        self.setMouseTracking(True)

    # ---------- Image / selection helpers ----------
    def set_image(self, img_array: np.ndarray):
        if img_array is None or img_array.ndim != 3:
            raise ValueError("Expected HxWxC image array")
        self._image_array = img_array.copy()
        self._selection_mask = np.zeros(self._image_array.shape[:2], dtype=bool)
        self._selection_rect = None
        self._update_pixmap(self._image_array)

    def image_array(self) -> Optional[np.ndarray]:
        return self._image_array

    def selection_mask(self) -> Optional[np.ndarray]:
        return self._selection_mask

    def clear_selection(self):
        if self._selection_mask is not None:
            self._selection_mask.fill(False)
        self._selection_rect = None
        if self._image_array is not None:
            self._update_pixmap(self._image_array)

    def select_row(self, row_idx: int, row_height: int = 4):
        if self._image_array is None:
            return
        h, w, _ = self._image_array.shape
        row_idx = max(0, min(h - 1, row_idx))
        y0 = row_idx
        y1 = min(h, y0 + max(1, row_height))
        mask = np.zeros((h, w), dtype=bool)
        mask[y0:y1, :] = True
        self._selection_mask = mask
        self._selection_rect = QRect(0, y0, w, y1 - y0)
        self._update_pixmap(self._image_array)

    def set_adjusted_preview(self, preview_array: np.ndarray):
        """Update display with adjusted preview (non-destructive)."""
        if preview_array is None:
            return
        self._update_pixmap(preview_array)

    def _update_pixmap(self, src_array: np.ndarray):
        h, w, c = src_array.shape
        if c == 3:
            fmt = QImage.Format.Format_RGB888
        else:
            fmt = QImage.Format.Format_RGBA8888
        qimg = QImage(src_array.data, w, h, src_array.strides[0], fmt)

        # Fit-to-window scale similar to palette_creator
        ww, wh = self.width(), self.height()
        base_scale = min(ww / max(1, w), wh / max(1, h))
        self._scale_factor = base_scale
        scaled_w = max(1, int(w * base_scale))
        scaled_h = max(1, int(h * base_scale))
        scaled_pix = QPixmap.fromImage(qimg).scaled(scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cached_scaled_size = (scaled_w, scaled_h)

        # Centered offsets
        self._offset_x = (ww - scaled_w) // 2
        self._offset_y = (wh - scaled_h) // 2

        viewport = QPixmap(ww, wh)
        viewport.fill(QColor(0, 0, 0, 0))
        painter = QPainter(viewport)
        painter.drawPixmap(self._offset_x, self._offset_y, scaled_pix)

        # Draw selection overlay if any
        if self._selection_mask is not None and self._selection_mask.any():
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            overlay[self._selection_mask] = [0, 170, 255, 110]
            q_overlay = QImage(overlay.data, w, h, overlay.strides[0], QImage.Format.Format_RGBA8888)
            overlay_pix = QPixmap.fromImage(q_overlay).scaled(scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(self._offset_x, self._offset_y, overlay_pix)

        painter.end()
        self.setPixmap(viewport)

    # ---------- Mouse events ----------
    def mousePressEvent(self, event: QMouseEvent):
        if self._image_array is None:
            return
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._selection_rect = QRect(event.position().toPoint(), event.position().toPoint())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging and self._selection_rect is not None:
            self._selection_rect.setBottomRight(event.position().toPoint())
            self._update_selection_from_rect()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self._update_selection_from_rect()
        super().mouseReleaseEvent(event)

    def _update_selection_from_rect(self):
        if self._image_array is None or self._selection_rect is None or self._cached_scaled_size is None:
            return
        h, w, _ = self._image_array.shape
        scaled_w, scaled_h = self._cached_scaled_size

        # Map widget coords back to image coords
        rect = self._selection_rect.normalized()
        x0 = max(0, rect.left() - self._offset_x)
        y0 = max(0, rect.top() - self._offset_y)
        x1 = max(0, rect.right() - self._offset_x)
        y1 = max(0, rect.bottom() - self._offset_y)

        # Scale to image space
        if scaled_w == 0 or scaled_h == 0:
            return
        img_x0 = int(x0 * w / scaled_w)
        img_y0 = int(y0 * h / scaled_h)
        img_x1 = int(x1 * w / scaled_w)
        img_y1 = int(y1 * h / scaled_h)

        x0_i = max(0, min(w - 1, min(img_x0, img_x1)))
        y0_i = max(0, min(h - 1, min(img_y0, img_y1)))
        x1_i = max(0, min(w, max(img_x0, img_x1) + 1))
        y1_i = max(0, min(h, max(img_y0, img_y1) + 1))

        mask = np.zeros((h, w), dtype=bool)
        mask[y0_i:y1_i, x0_i:x1_i] = True
        self._selection_mask = mask
        self._update_pixmap(self._image_array)


class PaletteAdjuster(BaseWidget):
    """Widget to tweak palette rows and append new variants."""

    def __init__(self, parent: Optional[QWidget], text: str = "Palette Adjuster"):
        super().__init__(parent=parent, text=text, vertical=True)

        # State
        self.palette_path: Optional[str] = None
        self.palette_img: Optional[Image.Image] = None
        self.palette_array: Optional[np.ndarray] = None
        self.original_array: Optional[np.ndarray] = None
        self.greyscale_img: Optional[Image.Image] = None
        self.adjust_state = AdjustmentState()
        self.start_color = QColor(0, 0, 0)
        self.end_color = QColor(255, 255, 255)
        self.row_height = int(cfg.get(cfg.ci_palette_row_height)) if hasattr(cfg, 'ci_palette_row_height') else 4

        # Settings / help drawers
        self.settings_widget = PaletteSettings(self)
        self.settings_drawer.addWidget(self.settings_widget)

        # Top file pickers
        self.palette_card = PushSettingCard(
            self.tr("Palette Texture"),
            CustomIcons.PALETTE.icon(),
            self.tr("Select an existing palette image"),
            self.tr("No palette selected"),
        )
        self.palette_card.clicked.connect(self.on_select_palette)

        self.greyscale_card = PushSettingCard(
            self.tr("Greyscale Preview (optional)"),
            CustomIcons.GREYSCALE.icon(),
            self.tr("Pick a greyscale image to preview the palette"),
            self.tr("No greyscale selected"),
        )
        self.greyscale_card.clicked.connect(self.on_select_greyscale)

        self.addToFrame(self.palette_card)
        self.addToFrame(self.greyscale_card)

        # Splitter layout (left: previews, right: controls)
        splitter = QSplitter(Qt.Horizontal)

        # Left side
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.canvas = PaletteAdjustCanvas()
        left_layout.addWidget(self.canvas, 3)

        # Greyscale preview
        preview_label = QLabel(self.tr("Greyscale Preview"))
        preview_label.setAlignment(Qt.AlignCenter)
        self.preview_img_label = QLabel()
        self.preview_img_label.setAlignment(Qt.AlignCenter)
        self.preview_img_label.setMinimumSize(300, 420)
        self.preview_img_label.setStyleSheet("background-color: #111; border: 1px solid #333;")

        left_layout.addWidget(preview_label)
        left_layout.addWidget(self.preview_img_label, 2)

        splitter.addWidget(left_panel)

        # Right side controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Selection controls
        sel_row_layout = QVBoxLayout()

        # Row selector as RangeSettingCard (1-based)
        self.row_cfg = RangeConfigItem("palette_adjuster", "row_index", 1, RangeValidator(1, 256))
        self.row_card = RangeSettingCard(
            self.row_cfg,
            CustomIcons.HEIGHT.icon(stroke=True) if hasattr(CustomIcons, 'HEIGHT') else FIF.BRUSH,
            self.tr("Row"),
            self.tr("Select palette row (1-based)")
        )
        self._updating_row_card = False
        self.row_card.valueChanged.connect(self.on_row_card_changed)
        sel_row_layout.addWidget(self.row_card)

        row_btns = QHBoxLayout()
        self.select_row_btn = PushButton(self.tr("Select Row"))
        self.select_row_btn.clicked.connect(self.on_select_row)
        row_btns.addWidget(self.select_row_btn)

        self.clear_sel_btn = PushButton(self.tr("Clear Selection"))
        self.clear_sel_btn.clicked.connect(self.canvas.clear_selection)
        row_btns.addWidget(self.clear_sel_btn)

        sel_row_layout.addLayout(row_btns)

        right_layout.addLayout(sel_row_layout)

        # Apply scope
        self.apply_selection_only = QCheckBox(self.tr("Adjust selection only (otherwise entire palette)"))
        self.apply_selection_only.setChecked(True)
        right_layout.addWidget(self.apply_selection_only)

        # Adjustment cards
        self.hue_cfg = RangeConfigItem("palette_adjuster", "hue", 0, RangeValidator(-180, 180))
        self.sat_cfg = RangeConfigItem("palette_adjuster", "saturation", 0, RangeValidator(-100, 100))
        self.val_cfg = RangeConfigItem("palette_adjuster", "value", 0, RangeValidator(-100, 100))
        self.bright_cfg = RangeConfigItem("palette_adjuster", "brightness", 0, RangeValidator(-100, 100))
        self.contrast_cfg = RangeConfigItem("palette_adjuster", "contrast", 0, RangeValidator(-100, 100))

        right_layout.addWidget(QLabel("<b>HSV Adjustments</b>"))
        self.hue_card = RangeSettingCard(self.hue_cfg, CustomIcons.SINE.icon(stroke=True) if hasattr(CustomIcons, 'SINE') else FIF.BRUSH,
                                         self.tr("Hue"), self.tr("-180 to 180"))
        self.sat_card = RangeSettingCard(self.sat_cfg, CustomIcons.SATURATION.icon(stroke=False),
                                         self.tr("Saturation"), self.tr("-100 to 100"))
        self.val_card = RangeSettingCard(self.val_cfg, CustomIcons.SPARK.icon(),
                                         self.tr("Value"), self.tr("-100 to 100"))

        right_layout.addWidget(self.hue_card)
        right_layout.addWidget(self.sat_card)
        right_layout.addWidget(self.val_card)

        right_layout.addWidget(QLabel("<b>Brightness / Contrast</b>"))
        self.brightness_card = RangeSettingCard(self.bright_cfg, FIF.BRIGHTNESS,
                                                self.tr("Brightness"), self.tr("-100 to 100"))
        self.contrast_card = RangeSettingCard(self.contrast_cfg, CustomIcons.CONTRAST.icon(stroke=True),
                                              self.tr("Contrast"), self.tr("-100 to 100"))

        right_layout.addWidget(self.brightness_card)
        right_layout.addWidget(self.contrast_card)

        for card in (self.hue_card, self.sat_card, self.val_card, self.brightness_card, self.contrast_card):
            card.valueChanged.connect(self.on_adjustments_changed)
            card.valueChanged.connect(self.apply_preview_adjustment)

        btn_row = QHBoxLayout()
        self.preview_btn = PushButton(self.tr("Preview Adjustment"))
        self.preview_btn.clicked.connect(self.apply_preview_adjustment)
        btn_row.addWidget(self.preview_btn)

        self.apply_btn = PrimaryPushButton(self.tr("Apply Adjustment"))
        self.apply_btn.clicked.connect(self.commit_adjustment)
        btn_row.addWidget(self.apply_btn)
        right_layout.addLayout(btn_row)

        # Gradient / fill
        right_layout.addWidget(QLabel("<b>Gradient / Fill</b>"))
        grad_row = QHBoxLayout()
        self.pick_start_btn = PushButton(self.tr("Start Color"))
        self.pick_start_btn.clicked.connect(lambda: self.pick_color(is_start=True))
        grad_row.addWidget(self.pick_start_btn)
        self.pick_end_btn = PushButton(self.tr("End Color"))
        self.pick_end_btn.clicked.connect(lambda: self.pick_color(is_start=False))
        grad_row.addWidget(self.pick_end_btn)
        right_layout.addLayout(grad_row)

        grad_apply_row = QHBoxLayout()
        self.gradient_h_btn = PushButton(self.tr("Apply Gradient H"))
        self.gradient_h_btn.clicked.connect(lambda: self.apply_gradient(horizontal=True))
        grad_apply_row.addWidget(self.gradient_h_btn)
        self.gradient_v_btn = PushButton(self.tr("Apply Gradient V"))
        self.gradient_v_btn.clicked.connect(lambda: self.apply_gradient(horizontal=False))
        grad_apply_row.addWidget(self.gradient_v_btn)
        right_layout.addLayout(grad_apply_row)

        fill_row = QHBoxLayout()
        self.fill_btn = PushButton(self.tr("Fill Selection"))
        self.fill_btn.clicked.connect(self.apply_fill)
        fill_row.addWidget(self.fill_btn)
        right_layout.addLayout(fill_row)

        # Palette operations
        right_layout.addWidget(QLabel("<b>Palette Operations</b>"))
        pal_row = QHBoxLayout()
        self.reload_btn = PushButton(self.tr("Reload Palette"))
        self.reload_btn.clicked.connect(self.reload_palette)
        pal_row.addWidget(self.reload_btn)
        self.reset_btn = PushButton(self.tr("Reset Adjustments"))
        self.reset_btn.clicked.connect(self.reset_to_original)
        pal_row.addWidget(self.reset_btn)
        right_layout.addLayout(pal_row)

        self.add_row_btn = PrimaryPushButton(self.tr("Add Adjusted Row To Palette"))
        self.add_row_btn.clicked.connect(self.append_row_to_palette)
        right_layout.addWidget(self.add_row_btn)

        right_layout.addStretch(1)

        splitter.addWidget(right_panel)
        splitter.setSizes([900, 420])
        self.addToFrame(splitter)

        self._update_color_buttons()

    # ---------- File pickers ----------
    def on_select_palette(self):
        path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Select Palette"), "", self.tr("Images (*.png *.jpg *.jpeg *.bmp *.tga *.webp *.dds)"))
        if not path:
            return
        try:
            img = load_image(path)
            self.palette_img = img.convert("RGBA")
            self.palette_array = np.array(self.palette_img, dtype=np.uint8)
            self.original_array = self.palette_array.copy()
            h, w = self.palette_array.shape[:2]
            self._set_row_range_and_value(max(1, h), 1)
            self.palette_path = path
            self.palette_card.setContent(f"{os.path.basename(path)} | {w}x{h}")
            self.canvas.set_image(self.palette_array)
            self.update_preview()
        except Exception as e:
            logger.exception("Failed to open palette image: %s", e)
            self.palette_card.setContent(self.tr("Failed to open palette"))

    def on_select_greyscale(self):
        path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Select Greyscale"), "", self.tr("Images (*.png *.jpg *.jpeg *.bmp *.tga *.webp *.dds)"))
        if not path:
            return
        try:
            img = load_image(path, f='L')
            self.greyscale_img = img
            self.greyscale_card.setContent(f"{os.path.basename(path)} | {img.size[0]}x{img.size[1]} (L)")
            self.update_preview()
        except Exception as e:
            logger.exception("Failed to open greyscale image: %s", e)
            self.greyscale_card.setContent(self.tr("Failed to open greyscale"))

    # ---------- Selection ----------
    def on_row_card_changed(self, val: int):
        if self._updating_row_card:
            return
        self.canvas.clear_selection()
        self._refresh_canvas_row_view()
        self.update_preview()

    def on_select_row(self):
        if self.palette_array is None:
            return
        if self.canvas.image_array() is None:
            return
        self.canvas.select_row(0, self.canvas.image_array().shape[0])

    # ---------- Adjustment handling ----------
    def on_adjustments_changed(self, *_args):
        self.adjust_state.hue = int(self.hue_cfg.value)
        self.adjust_state.saturation = int(self.sat_cfg.value)
        self.adjust_state.value = int(self.val_cfg.value)
        self.adjust_state.brightness = int(self.bright_cfg.value)
        self.adjust_state.contrast = int(self.contrast_cfg.value)

    def apply_preview_adjustment(self):
        if self.palette_array is None:
            return
        adjusted = self._apply_adjustment(self.palette_array, preview_only=True)
        if adjusted is not None:
            row_preview = self._extract_row_slice(adjusted)
            self.canvas.set_adjusted_preview(self._build_row_display_from_array(row_preview))
            self.update_preview(adjusted)

    def commit_adjustment(self):
        if self.palette_array is None:
            return
        adjusted = self._apply_adjustment(self.palette_array, preview_only=False)
        if adjusted is not None:
            self.palette_array = adjusted
            self.palette_img = Image.fromarray(self.palette_array, mode="RGBA" if self.palette_array.shape[2] == 4 else "RGB")
            self._refresh_canvas_row_view()
            self.update_preview()

    def _apply_adjustment(self, src: np.ndarray, preview_only: bool) -> Optional[np.ndarray]:
        if src is None:
            return None
        if cv2 is None:
            if not preview_only:
                QMessageBox.warning(self, self.tr("Missing Dependency"), self.tr("OpenCV is not available; HSV adjustments are disabled."))
            return None
        palette_mask = self._build_palette_mask_from_canvas_selection() if self.apply_selection_only.isChecked() else None
        if not self.apply_selection_only.isChecked():
            palette_mask = np.ones(src.shape[:2], dtype=bool)
        elif palette_mask is None:
            # Default to current row
            row_idx = self._get_row_value(max_row=src.shape[0]) - 1
            row_end = min(src.shape[0], row_idx + self.row_height)
            palette_mask = np.zeros(src.shape[:2], dtype=bool)
            palette_mask[row_idx:row_end, :] = True

        adj = src.copy()
        try:
            rgb = adj[:, :, :3].astype(np.float32)

            # OpenCV expects 0-255 uint8 in RGB order
            hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

            # Hue shift (-180..180) maps to OpenCV 0..180 scale (H is 0-179)
            h_shift = self.adjust_state.hue / 2.0
            hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180

            s_scale = 1.0 + self.adjust_state.saturation / 100.0
            v_scale = 1.0 + self.adjust_state.value / 100.0
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)

            rgb_new = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

            # Brightness / contrast on 0..255
            c_scale = 1.0 + self.adjust_state.contrast / 100.0
            b_add = self.adjust_state.brightness * 255.0 / 100.0
            rgb_new = np.clip((rgb_new - 127.5) * c_scale + 127.5 + b_add, 0, 255)

            # Blend only on masked area
            rgb_out = rgb.copy()
            rgb_out[palette_mask] = rgb_new[palette_mask]

            adj[:, :, :3] = rgb_out.astype(np.uint8)
            return adj
        except Exception:
            logger.exception("Failed to apply adjustments")
            if not preview_only:
                QMessageBox.warning(self, self.tr("Adjustment Failed"), self.tr("Could not apply adjustments."))
            return None

    # ---------- Gradient / Fill ----------
    def pick_color(self, is_start: bool):
        color = QColorDialog.getColor(self.start_color if is_start else self.end_color, self)
        if color.isValid():
            if is_start:
                self.start_color = color
            else:
                self.end_color = color
            self._update_color_buttons()

    def apply_gradient(self, horizontal: bool):
        if self.palette_array is None:
            return
        mask = self.canvas.selection_mask()
        apply_selection = self.apply_selection_only.isChecked()
        if mask is None or not mask.any() or not apply_selection:
            mask = np.ones(self.palette_array.shape[:2], dtype=bool)

        y_idx, x_idx = np.nonzero(mask)
        if y_idx.size == 0:
            return
        y0, y1 = y_idx.min(), y_idx.max()
        x0, x1 = x_idx.min(), x_idx.max()

        arr = self.palette_array.copy()
        start = np.array([self.start_color.red(), self.start_color.green(), self.start_color.blue()], dtype=np.float32)
        end = np.array([self.end_color.red(), self.end_color.green(), self.end_color.blue()], dtype=np.float32)

        if horizontal:
            span = max(1, x1 - x0)
            ramp = (np.arange(x0, x1 + 1) - x0) / span
            grad = start[None, :] * (1.0 - ramp[:, None]) + end[None, :] * ramp[:, None]
            for xi in range(x0, x1 + 1):
                arr[y0:y1 + 1, xi, :3][mask[y0:y1 + 1, xi]] = np.clip(grad[xi - x0], 0, 255).astype(np.uint8)
        else:
            span = max(1, y1 - y0)
            ramp = (np.arange(y0, y1 + 1) - y0) / span
            grad = start[None, :] * (1.0 - ramp[:, None]) + end[None, :] * ramp[:, None]
            for yi in range(y0, y1 + 1):
                arr[yi, x0:x1 + 1, :3][mask[yi, x0:x1 + 1]] = np.clip(grad[yi - y0], 0, 255).astype(np.uint8)

        self.palette_array = arr
        self.palette_img = Image.fromarray(arr, mode="RGBA" if arr.shape[2] == 4 else "RGB")
        self._refresh_canvas_row_view()
        self.update_preview()

    def apply_fill(self):
        if self.palette_array is None:
            return
        mask = self.canvas.selection_mask()
        apply_selection = self.apply_selection_only.isChecked()
        if mask is None or not mask.any() or not apply_selection:
            mask = np.ones(self.palette_array.shape[:2], dtype=bool)

        color = self.start_color
        arr = self.palette_array.copy()
        arr[mask, 0] = color.red()
        arr[mask, 1] = color.green()
        arr[mask, 2] = color.blue()

        self.palette_array = arr
        self.palette_img = Image.fromarray(arr, mode="RGBA" if arr.shape[2] == 4 else "RGB")
        self._refresh_canvas_row_view()
        self.update_preview()

    # ---------- Palette operations ----------
    def reload_palette(self):
        if not self.palette_path:
            return
        try:
            img = load_image(self.palette_path)
            self.palette_img = img.convert("RGBA")
            self.palette_array = np.array(self.palette_img, dtype=np.uint8)
            self.original_array = self.palette_array.copy()
            h, w = self.palette_array.shape[:2]
            self._set_row_range_and_value(max(1, h), max(1, min(self._get_row_value(), h)))
            self._refresh_canvas_row_view()
            self.update_preview()
        except Exception:
            logger.exception("Failed to reload palette")

    def reset_to_original(self):
        if self.original_array is None:
            return
        self.palette_array = self.original_array.copy()
        self.palette_img = Image.fromarray(self.palette_array, mode="RGBA" if self.palette_array.shape[2] == 4 else "RGB")
        self._refresh_canvas_row_view()
        self.update_preview()

    def append_row_to_palette(self):
        if self.palette_array is None or self.palette_path is None:
            return
        h, w, c = self.palette_array.shape
        sel_row = self._get_row_value(max_row=h)
        row_idx = sel_row - 1
        try:
            # Build new block from selected row (repeat row_height)
            row_data = self.palette_array[row_idx:row_idx + 1, :, :3]
            new_block = np.repeat(row_data, self.row_height, axis=0)

            pal_arr = self.palette_array
            if pal_arr.shape[2] == 3:
                pal_new = np.concatenate([pal_arr, new_block], axis=0)
                mode = "RGB"
            else:
                row_rgba = np.zeros((new_block.shape[0], w, 4), dtype=np.uint8)
                row_rgba[:, :, :3] = new_block
                row_rgba[:, :, 3] = 255
                pal_new = np.concatenate([pal_arr, row_rgba], axis=0)
                mode = "RGBA"

            new_img = Image.fromarray(pal_new, mode=mode)
            save_image(new_img, self.palette_path, True)
            self.palette_img = new_img
            self.palette_array = np.array(new_img, dtype=np.uint8)
            self.original_array = self.palette_array.copy()
            self._set_row_range_and_value(max(1, self.palette_array.shape[0]), sel_row)
            self._refresh_canvas_row_view()
            self.update_preview()
            QMessageBox.information(self, self.tr("Palette Updated"), self.tr("Row appended and palette reloaded."))
        except Exception:
            logger.exception("Failed to append row")
            QMessageBox.warning(self, self.tr("Append Failed"), self.tr("Unable to append row."))

    # ---------- Preview ----------
    def update_preview(self, preview_palette: Optional[np.ndarray] = None):
        if self.palette_img is None and preview_palette is None:
            return
        try:
            palette_img = self.palette_img
            if preview_palette is not None:
                palette_img = Image.fromarray(preview_palette[:, :, :3], mode="RGB")

            row_slice = self._extract_row_slice(np.array(palette_img))
            display = self._build_row_display_from_array(row_slice)
            self._set_preview(self.preview_img_label, display)
        except Exception:
            logger.exception("Failed to update preview")

    def _set_preview(self, label: QLabel, row_array: np.ndarray):
        if row_array is None or row_array.size == 0:
            return
        arr = row_array
        if arr.shape[2] == 3:
            fmt = QImage.Format.Format_RGB888
        else:
            fmt = QImage.Format.Format_RGBA8888
        qimg = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], fmt)
        label.setPixmap(QPixmap.fromImage(qimg))

    # ---------- Row view helpers ----------
    def _refresh_canvas_row_view(self):
        if self.palette_array is None:
            return
        h = self.palette_array.shape[0]
        sel_row = self._get_row_value(max_row=h)
        row_idx = sel_row - 1
        self._updating_row_card = True
        self.row_card.setValue(sel_row)
        self._updating_row_card = False
        row_slice = self.palette_array[row_idx: min(h, row_idx + self.row_height), :, :]
        display = self._build_row_display_from_array(row_slice)
        self.canvas.set_image(display)

    def _build_row_display_from_array(self, arr: np.ndarray) -> np.ndarray:
        if arr is None or arr.size == 0:
            return arr
        h = arr.shape[0]
        target_h = max(24, h * 6)
        repeat = max(1, math.ceil(target_h / h))
        tiled = np.repeat(arr, repeat, axis=0)
        return tiled[:target_h]

    def _extract_row_slice(self, array: np.ndarray) -> np.ndarray:
        if array is None or array.size == 0:
            return array
        h = array.shape[0]
        sel_row = self._get_row_value(max_row=h)
        row_idx = max(0, min(h - 1, sel_row - 1))
        return array[row_idx: row_idx + 1, :, :]

    def _build_palette_mask_from_canvas_selection(self) -> Optional[np.ndarray]:
        if self.palette_array is None:
            return None
        mask_view = self.canvas.selection_mask()
        if mask_view is None or not mask_view.any():
            return None
        col_mask = mask_view.any(axis=0)
        if not col_mask.any():
            return None
        h, w = self.palette_array.shape[:2]
        sel_row = self._get_row_value(max_row=h)
        row_idx = sel_row - 1
        row_end = min(h, row_idx + self.row_height)
        palette_mask = np.zeros((h, w), dtype=bool)
        palette_mask[row_idx:row_end, :] = col_mask[None, :]
        return palette_mask

    # ---------- Row helpers ----------
    def _set_row_range_and_value(self, max_row: int, value: int):
        max_row = max(1, max_row)
        self.row_card.slider.setRange(1, max_row)
        self.row_cfg.range = (1, max_row)
        self._updating_row_card = True
        self.row_card.setValue(max(1, min(max_row, value)))
        self._updating_row_card = False

    def _get_row_value(self, max_row: Optional[int] = None) -> int:
        val = int(self.row_cfg.value)
        if max_row is not None:
            val = max(1, min(max_row, val))
        return val

    # ---------- Color UI helpers ----------
    def _update_color_buttons(self):
        def _text_color(qc: QColor) -> str:
            # Simple luminance check
            lum = (0.299 * qc.red() + 0.587 * qc.green() + 0.114 * qc.blue()) / 255
            return "#000" if lum > 0.6 else "#fff"

        start_css = f"background-color: rgb({self.start_color.red()}, {self.start_color.green()}, {self.start_color.blue()});"
        end_css = f"background-color: rgb({self.end_color.red()}, {self.end_color.green()}, {self.end_color.blue()});"
        grad_css = (
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            f" stop:0 rgb({self.start_color.red()}, {self.start_color.green()}, {self.start_color.blue()}),"
            f" stop:1 rgb({self.end_color.red()}, {self.end_color.green()}, {self.end_color.blue()}));"
        )
        text_start = _text_color(self.start_color)
        text_end = _text_color(self.end_color)
        self.pick_start_btn.setStyleSheet(start_css + f" color: {text_start};")
        self.pick_end_btn.setStyleSheet(end_css + f" color: {text_end};")
        self.gradient_h_btn.setStyleSheet(grad_css + " color: #fff;")
        self.gradient_v_btn.setStyleSheet(grad_css + " color: #fff;")
        self.fill_btn.setStyleSheet(start_css + f" color: {text_start};")


# Fallback import for cv2 if missing at module import time
try:
    import cv2
except Exception as _cv2_import_error:  # pragma: no cover - optional dependency handled at runtime
    cv2 = None
    logger = logger if 'logger' in globals() else None
    if logger:
        logger.warning("OpenCV not available; HSV adjustments will be skipped.")