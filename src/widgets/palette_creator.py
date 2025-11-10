import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QCursor
from PySide6.QtWidgets import (QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QFileDialog,
                               QListWidget, QMessageBox, QSplitter)
from qfluentwidgets import PushSettingCard, PushButton, ScrollArea, PrimaryPushButton, \
    FluentIcon as FIF, ConfigItem, RangeSettingCard
from scipy import ndimage

from help.palette_help import PaletteHelp
from settings.palette_settings import PaletteSettings
from src.utils.appconfig import cfg
from src.utils.cards import ComboBoxSettingsCard, RadioSettingCard
from src.utils.dds_utils import save_image
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger
from src.utils.nifutils import build_uv_entries_for_nif, build_mask_from_nif
from src.utils.palette_utils import quantize_image, apply_palette_to_greyscale


class ImageCanvas(QLabel):
    """Interactive canvas for selecting image regions"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("background-color: #2b2b2b; border: 2px solid #555;")

        self.original_image = None
        self.display_pixmap = None
        self.current_mask = None
        self.all_masks = {}  # island_name -> mask array
        self.current_island = None
        self.selection_color = QColor(255, 0, 0, 128)
        self.edit_locked = False
        self._edit_lock_reason = None

        # View transforms
        self.zoom_factor = 1.0
        self.pan_x_percent = 0  # -100..100 maps to left..right
        self.pan_y_percent = 0  # -100..100 maps to top..bottom

        self.setMouseTracking(True)

    def set_image_array(self, img_array: np.ndarray, quantized_pil, quantized_rgb):
        """Set the canvas image from an RGBA numpy array (quantizes RGB like load_image)."""
        if img_array is None or img_array.ndim != 3 or img_array.shape[2] < 4:
            raise ValueError("Expected RGBA array")

        alpha_channel = img_array[:, :, 3]

        self.original_image = np.dstack([quantized_rgb, alpha_channel])

        # Convert to QPixmap for display
        height, width, channel = self.original_image.shape
        bytes_per_line = 4 * width
        q_image = QImage(self.original_image.data, width, height,
                         bytes_per_line, QImage.Format.Format_RGBA8888)
        self.display_pixmap = QPixmap.fromImage(q_image)
        self.reset_view()
        return True

    def update_display(self):
        """Update the display with current image and masks"""
        if self.display_pixmap is None:
            return

        # Create a copy to draw on
        display = self.display_pixmap.copy()
        painter = QPainter(display)

        # Draw all existing masks
        for island_name, mask in self.all_masks.items():
            if island_name != self.current_island:
                self.draw_mask_overlay(painter, mask, QColor(100, 100, 255, 80))

        # Draw current mask being edited
        if self.current_island and self.current_island in self.all_masks:
            mask = self.all_masks[self.current_island]
            self.draw_mask_overlay(painter, mask, self.selection_color)

        painter.end()

        # Fit-to-window base scale
        ww, wh = self.width(), self.height()
        img_w, img_h = display.width(), display.height()
        if img_w == 0 or img_h == 0 or ww == 0 or wh == 0:
            return

        base_scale = min(ww / img_w, wh / img_h)
        final_scale = base_scale * self.zoom_factor
        scaled_w = max(1, int(img_w * final_scale))
        scaled_h = max(1, int(img_h * final_scale))

        scaled_pix = display.scaled(scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Determine pan offsets within the available range
        min_offset_x = min(0, ww - scaled_pix.width())
        max_offset_x = 0
        min_offset_y = min(0, wh - scaled_pix.height())
        max_offset_y = 0

        t_x = (self.pan_x_percent + 100) / 200.0
        t_y = (self.pan_y_percent + 100) / 200.0
        offset_x = max_offset_x + (min_offset_x - max_offset_x) * t_x
        offset_y = max_offset_y + (min_offset_y - max_offset_y) * t_y

        # Draw into a viewport-sized pixmap
        viewport = QPixmap(ww, wh)
        viewport.fill(QColor(0, 0, 0, 0))
        vp_painter = QPainter(viewport)
        vp_painter.drawPixmap(int(offset_x), int(offset_y), scaled_pix)
        vp_painter.end()

        self._cached_scaled_size = (scaled_pix.width(), scaled_pix.height())
        self._cached_offsets = (offset_x, offset_y)
        self.setPixmap(viewport)

    def draw_mask_overlay(self, painter, mask, color):
        """Draw a mask overlay on the painter"""
        if mask is None or self.original_image is None:
            return

        height, width = mask.shape
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        overlay[mask] = [color.red(), color.green(), color.blue(), color.alpha()]

        bytes_per_line = 4 * width
        q_image = QImage(overlay.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
        overlay_pixmap = QPixmap.fromImage(q_image)

        painter.drawPixmap(0, 0, overlay_pixmap)

    def set_current_island(self, island_name):
        """Set the currently active island for selection"""
        self.current_island = island_name
        if island_name not in self.all_masks and self.original_image is not None:
            # Create empty mask
            self.all_masks[island_name] = np.zeros(
                (self.original_image.shape[0], self.original_image.shape[1]),
                dtype=bool
            )
        self.update_display()

    def map_to_image_coords(self, widget_point):
        """Map widget coordinates to original image coordinates"""
        if self.pixmap() is None or self.original_image is None:
            return None

        ww, wh = self.width(), self.height()
        img_h, img_w = self.original_image.shape[:2]
        base_scale = min(ww / img_w, wh / img_h)
        final_scale = base_scale * self.zoom_factor
        scaled_w = max(1, int(img_w * final_scale))
        scaled_h = max(1, int(img_h * final_scale))

        min_offset_x = min(0, ww - scaled_w)
        max_offset_x = 0
        min_offset_y = min(0, wh - scaled_h)
        max_offset_y = 0
        t_x = (self.pan_x_percent + 100) / 200.0
        t_y = (self.pan_y_percent + 100) / 200.0
        offset_x = max_offset_x + (min_offset_x - max_offset_x) * t_x
        offset_y = max_offset_y + (min_offset_y - max_offset_y) * t_y

        adj_x = widget_point.x() - offset_x
        adj_y = widget_point.y() - offset_y

        if adj_x < 0 or adj_y < 0 or adj_x >= scaled_w or adj_y >= scaled_h:
            return None

        scale_x = img_w / scaled_w
        scale_y = img_h / scaled_h

        img_x = int(adj_x * scale_x)
        img_y = int(adj_y * scale_y)

        return QPoint(img_x, img_y)

    def set_zoom(self, zoom_factor: float):
        self.zoom_factor = max(0.1, min(5.0, zoom_factor))
        self.update_display()

    def set_pan(self, pan_x_percent: int = None, pan_y_percent: int = None):
        if pan_x_percent is not None:
            self.pan_x_percent = max(-100, min(100, pan_x_percent))
        if pan_y_percent is not None:
            self.pan_y_percent = max(-100, min(100, pan_y_percent))
        self.update_display()

    def reset_view(self):
        self.zoom_factor = 1.0
        self.pan_x_percent = 0
        self.pan_y_percent = 0
        self.update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()

    def mousePressEvent(self, event):
        if self.edit_locked:
            return
        if event.button() == Qt.LeftButton and self.original_image is not None and self.current_island:
            # Get clicked pixel in image coordinates
            click_pos = self.map_to_image_coords(event.pos())
            if click_pos:
                self.flood_fill_selection(click_pos.x(), click_pos.y())

    def mouseMoveEvent(self, event):
        # Show crosshair cursor when over image
        if self.original_image is not None:
            pos = self.map_to_image_coords(event.pos())
            if pos:
                self.setCursor(QCursor(Qt.CrossCursor))
            else:
                self.setCursor(QCursor(Qt.ArrowCursor))

    def mouseReleaseEvent(self, event):
        pass  # Not needed for click selection

    def flood_fill_selection(self, start_x, start_y):
        """Flood fill to select all connected non-transparent pixels"""
        if self.original_image is None or self.current_island is None:
            return

        height, width = self.original_image.shape[:2]

        # Check if start pixel is valid
        if start_x < 0 or start_x >= width or start_y < 0 or start_y >= height:
            return

        # Check if start pixel is transparent
        alpha = self.original_image[start_y, start_x, 3]
        if alpha == 0:
            QMessageBox.warning(self.parent(), "Warning",
                                "Clicked on transparent pixel. Please click on a non-transparent area.")
            return

        # Get the alpha channel
        alpha_channel = self.original_image[:, :, 3]

        # Create a mask for non-transparent pixels
        non_transparent = alpha_channel > 0

        # Perform flood fill to find connected region
        visited = np.zeros((height, width), dtype=bool)
        region_mask = np.zeros((height, width), dtype=bool)

        # Stack-based flood fill (4-connected)
        stack = [(start_x, start_y)]
        visited[start_y, start_x] = True

        while stack:
            x, y = stack.pop()
            region_mask[y, x] = True

            # Check 4 neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy

                # Check bounds
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue

                # Skip if already visited
                if visited[ny, nx]:
                    continue

                # Skip if transparent
                if not non_transparent[ny, nx]:
                    continue

                # Add to stack
                visited[ny, nx] = True
                stack.append((nx, ny))

        # Check if the clicked pixel is already selected
        mask = self.all_masks[self.current_island]
        is_already_selected = mask[start_y, start_x]

        if is_already_selected:
            # Remove region from current mask
            mask &= ~region_mask  # AND with NOT operation to remove from selection
            pixel_count = np.sum(region_mask)
            print(f"Removed {pixel_count} connected pixels from {self.current_island}")
        else:
            # Add region to current mask
            mask |= region_mask  # OR operation to add to existing selection
            pixel_count = np.sum(region_mask)
            print(f"Selected {pixel_count} connected pixels for {self.current_island}")

        self.update_display()

    def clear_current_mask(self):
        """Clear the current island's mask"""
        if self.current_island and self.current_island in self.all_masks:
            self.all_masks[self.current_island].fill(False)
            self.update_display()

    def get_all_masks(self):
        """Return all masks"""
        return self.all_masks


class PaletteLUTGenerator(BaseWidget):
    def __init__(self, parent, text):
        super().__init__(parent=parent, text=text, vertical=True)
        self.image_path = None
        self.pending_image_path = None
        self.ci_last_image_dir = ConfigItem("palette", "last_image_dir", "")
        self.ci_last_output_dir = ConfigItem("palette", "last_output_dir", "")
        self.model_path: Path | None = None
        self.uv_entries = []
        self.selected_uv_index = 0
        self._base_image_array = None  # quantized RGBA before UV masking
        self._quantized_image_array = None
        self._quantized_image = None
        self.islands = []  # List of (name, gray_start, gray_end)
        self.current_island_size = 16  # Default size
        self.island_edit_locked = False
        self._edit_lock_reason = None
        self._nontransparent_source = False
        self.settings_widget = PaletteSettings(self)
        self.settings_drawer.addWidget(self.settings_widget)
        self.help_widget = PaletteHelp(self)
        self.help_drawer.addWidget(self.help_widget)

        self.select_buttons = QHBoxLayout()

        self._palette_size_previous = cfg.get(cfg.ci_default_palette_size)
        self.base_image_card = PushSettingCard(
            self.tr("Select Texture"),
            CustomIcons.IMAGE.icon(stroke=True),
            self.tr("Source Image for Grayscale and Palette"),
            "No image selected"
        )
        self.base_image_card.clicked.connect(self.load_image)

        self.card_pick_model = PushSettingCard(
            title=self.tr("Pick NIF"),
            icon=CustomIcons.CUBE.icon(stroke=True),
            text=self.tr("Select a .nif"),
            content="Only If Not Transparent"
        )
        self.card_pick_model.clicked.connect(self.on_pick_model)

        # UV set chooser (populated after model load)
        self.card_uv_set = ComboBoxSettingsCard(
            icon=FIF.GLOBE,
            title=self.tr("UV Set"),
            content=self.tr("If the model has multiple UV sets, choose which to use."),
        )
        self.card_uv_set.combox.currentIndexChanged.connect(self.on_uv_changed)

        self.select_buttons.addWidget(self.base_image_card)
        self.select_buttons.addWidget(self.card_pick_model)
        self.select_buttons.addWidget(self.card_uv_set)

        self.addToLayout(self.select_buttons)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Canvas
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Canvas
        scroll = ScrollArea()
        scroll.setWidgetResizable(True)
        self.canvas = ImageCanvas()
        scroll.setWidget(self.canvas)
        left_layout.addWidget(scroll)

        splitter.addWidget(left_panel)

        # Right panel - Controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Island management
        right_layout.addWidget(QLabel("<b>Palette Islands:</b>"))

        island_controls = QHBoxLayout()

        self.add_island_btn = PushButton("Add Islands")
        self.add_island_btn.clicked.connect(self.add_island)
        island_controls.addWidget(self.add_island_btn)

        self.remove_island_btn = PushButton("Remove Islands")
        self.remove_island_btn.clicked.connect(self.remove_island)
        island_controls.addWidget(self.remove_island_btn)

        right_layout.addLayout(island_controls)

        self.auto_create_btn = PushButton("Auto Create Islands")
        self.auto_create_btn = PushButton("Auto Create Islands")
        self.auto_create_btn.clicked.connect(self.auto_create_islands)
        right_layout.addWidget(self.auto_create_btn)

        # Grouping sensitivity slider for auto grouping
        self.row_card = RangeSettingCard(
            cfg.ci_grouping_threshold,
            CustomIcons.SCALE.icon(),
            self.tr("Auto Grouping Sensitivity"),
            self.tr("Higher allows more colors to be selected"),
        )

        # Palette size selector
        self.palette_size_card = RadioSettingCard(
            cfg.ci_default_palette_size,
            CustomIcons.WIDTH.icon(),
            self.tr("Palette Size"),
            self.tr("Palette Size"),
            texts=["128", "64", "32"],
            parent=self
        )

        self.palette_size_card.optionChanged.connect(self.on_palette_size_changed)

        right_layout.addWidget(self.palette_size_card)
        right_layout.addWidget(self.row_card)

        self.island_list = QListWidget()
        self.island_list.currentItemChanged.connect(self.on_island_selected)
        right_layout.addWidget(self.island_list)

        # Gray range controls with buttons
        gray_range_layout = QVBoxLayout()

        size_label_layout = QHBoxLayout()
        size_label_layout.addWidget(QLabel("Island Size (power of 2):"))
        self.gray_range_label = QLabel("[0-15]")
        size_label_layout.addWidget(self.gray_range_label)
        size_label_layout.addStretch()
        gray_range_layout.addLayout(size_label_layout)


        size_buttons_layout = QHBoxLayout()

        self.size_8_btn = PushButton("8")
        self.size_8_btn.setCheckable(True)
        self.size_8_btn.setChecked(True)
        self.size_8_btn.clicked.connect(lambda: self.set_island_size(8))
        size_buttons_layout.addWidget(self.size_8_btn)

        self.size_16_btn = PushButton("16")
        self.size_16_btn.setCheckable(True)
        self.size_16_btn.setChecked(True)
        self.size_16_btn.clicked.connect(lambda: self.set_island_size(16))
        size_buttons_layout.addWidget(self.size_16_btn)

        self.size_32_btn = PushButton("32")
        self.size_32_btn.setCheckable(True)
        self.size_32_btn.clicked.connect(lambda: self.set_island_size(32))
        size_buttons_layout.addWidget(self.size_32_btn)

        self.size_64_btn = PushButton("64")
        self.size_64_btn.setCheckable(True)
        self.size_64_btn.clicked.connect(lambda: self.set_island_size(64))
        size_buttons_layout.addWidget(self.size_64_btn)

        self.size_128_btn = PushButton("128")
        self.size_128_btn.setCheckable(True)
        self.size_128_btn.clicked.connect(lambda: self.set_island_size(128))
        size_buttons_layout.addWidget(self.size_128_btn)

        gray_range_layout.addLayout(size_buttons_layout)

        self.available_space_label = QLabel(f"Available: {cfg.get(cfg.ci_default_palette_size)} Color Values")
        gray_range_layout.addWidget(self.available_space_label)

        right_layout.addLayout(gray_range_layout)

        # Selection controls
        right_layout.addWidget(QLabel("<b>Selection Tools:</b>"))

        self.clear_selection_btn = PushButton("Clear Current Island Selection")
        self.clear_selection_btn.clicked.connect(self.clear_current_selection)
        right_layout.addWidget(self.clear_selection_btn)

        self.magic_wand_btn = PushButton("Magic Wand (Select Similar)")
        self.magic_wand_btn.clicked.connect(self.magic_wand_select)
        right_layout.addWidget(self.magic_wand_btn)

        self.add_remaining_btn = PushButton("Add Remaining Islands to Island")
        self.add_remaining_btn.clicked.connect(self.add_remaining_to_current)
        right_layout.addWidget(self.add_remaining_btn)

        right_layout.addStretch()

        # Generation controls
        right_layout.addWidget(QLabel("<b>Generate:</b>"))

        self.generate_both_btn = PrimaryPushButton("Generate Grayscale And Palette")
        self.generate_both_btn.clicked.connect(self.generate_both)

        self.save_state_btn = PushButton("Save Islands")
        self.save_state_btn.clicked.connect(self.save_island_state)

        self.load_state_btn = PushButton("Load Islands")
        self.load_state_btn.clicked.connect(self.load_island_state)

        self.buttons_layout.addWidget(self.generate_both_btn, stretch=1)
        self.buttons_layout.addWidget(self.save_state_btn)
        self.buttons_layout.addWidget(self.load_state_btn)
        self.buttons_layout.addWidget(self.settings_button)
        self.buttons_layout.addWidget(self.help_button)
        right_layout.addLayout(self.buttons_layout)

        splitter.addWidget(right_panel)

        # Set initial splitter sizes
        splitter.setSizes([800, 400])

        self.addToFrame(splitter)

        # Ensure controls reflect default unlocked state
        self._set_edit_lock(False)

    def _set_edit_lock(self, locked: bool, reason: str | None = None):
        """Enable/disable editing of islands and selection tools."""
        self.island_edit_locked = bool(locked)
        self._edit_lock_reason = reason
        self.canvas.edit_locked = locked
        self.canvas._edit_lock_reason = reason

        controls = [
            self.add_island_btn,
            self.remove_island_btn,
            self.auto_create_btn,
            self.clear_selection_btn,
            self.magic_wand_btn,
            self.add_remaining_btn,
        ]

        for ctrl in controls:
            try:
                ctrl.setEnabled(not locked)
            except Exception:
                pass

        if locked and reason:
            self.gray_range_label.setText(reason)
        elif self.islands:
            name, gs, ge = self.islands[0]
            self.gray_range_label.setText(f"[{gs}-{ge}]")
        else:
            self.gray_range_label.setText("[0-15]")

    def _ensure_editable(self) -> bool:
        if not self.island_edit_locked:
            return True
        msg = self._edit_lock_reason or "Island editing is disabled for this image."
        QMessageBox.information(self, "Editing Locked", msg)
        return False

    def load_image(self):
        """Load a transparent PNG image"""
        last_dir = str(self.ci_last_image_dir.value or "")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Image for Palette Generation"),
            last_dir,
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.dds *.tga);;All Files (*)"
        )

        if file_path:
            try:
                pil_img = Image.open(file_path).convert("RGBA")
                img_array = np.array(pil_img)
                alpha_channel = img_array[:, :, 3]
                rgb_array = img_array[:, :, :3]

                # Quantize RGB similar to canvas.load_image
                pil_rgb = Image.fromarray(rgb_array, mode='RGB')
                self._quantized_image = quantize_image(pil_rgb, cfg.get(cfg.ci_default_quant_method))
                self._quantized_image_array = np.array(self._quantized_image.convert('RGB'))
                base_rgba = np.dstack([self._quantized_image_array, alpha_channel])
                self._base_image_array = base_rgba
                self.pending_image_path = None
                self.base_image_card.setContent(file_path)

                # Treat image as truly transparent only if it has any non-opaque pixels.
                # Merely having an RGBA channel is not enough.
                is_transparent = self._has_transparent_pixels(base_rgba)

                if is_transparent:
                    # Load directly
                    self.canvas.original_image = None  # ensure reset before set
                    self.canvas.set_image_array(base_rgba, self._quantized_image, self._quantized_image_array)
                    self.image_path = file_path
                    self._nontransparent_source = False
                    self._set_edit_lock(False)
                    self._reset_islands_and_masks()
                else:
                    # Opaque image: allow loading but lock editing; Islands must come from saved NPZ.
                    self.canvas.original_image = None  # ensure reset before set
                    self.canvas.set_image_array(base_rgba, self._quantized_image, self._quantized_image_array)
                    self.image_path = file_path
                    self.pending_image_path = file_path
                    self._nontransparent_source = True
                    self._set_edit_lock(True, "Loaded image has no transparency; island editing is disabled. Load saved Islands to continue.")
                    self._reset_islands_and_masks()
                    QMessageBox.information(
                        self,
                        "Editing Locked",
                        "Image has no transparent pixels. Island editing is disabled. Load a saved palette Islands NPZ to use existing islands."
                    )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def _build_island_state(self):
        """Package current island state and masks for serialization."""
        if self.canvas.original_image is None:
            raise ValueError("No image loaded")

        height, width = self.canvas.original_image.shape[:2]
        islands = [
            {
                "name": name,
                "gray_start": int(gray_start),
                "gray_end": int(gray_end),
            }
            for name, gray_start, gray_end in self.islands
        ]

        mask_stack = []
        for entry in islands:
            island_name = entry["name"]
            mask = self.canvas.all_masks.get(island_name)
            if mask is None:
                mask = np.zeros((height, width), dtype=bool)
            else:
                mask = mask.astype(bool, copy=False)
            mask_stack.append(mask)

        mask_stack = np.stack(mask_stack, axis=0) if mask_stack else np.zeros((0, height, width), dtype=bool)

        metadata = {
            "version": 1,
            "image_path": self.image_path,
            "islands": islands,
            "width": int(width),
            "height": int(height),
            "current_island": self.canvas.current_island,
            "model_path": str(self.model_path) if self.model_path else None,
            "selected_uv_index": int(self.selected_uv_index),
            "quant_method": str(cfg.get(cfg.ci_default_quant_method)),
        }

        return metadata, mask_stack

    def save_island_state(self):
        """Save current islands, ranges, and masks to a fast-loading NPZ."""
        if self.canvas.original_image is None or not self.image_path:
            QMessageBox.warning(self, "Warning", "Load an image and create Islands before saving.")
            return

        try:
            metadata, mask_stack = self._build_island_state()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not build Island state: {str(e)}")
            return

        base_path, _ = os.path.splitext(self.image_path)
        suggested = f"{base_path}_palette_state.npz"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save Palette Islands"),
            suggested,
            "Palette Islands (*.npz);;All Files (*)",
        )

        if not file_path:
            return

        try:
            np.savez_compressed(file_path, metadata=json.dumps(metadata), masks=mask_stack)
            QMessageBox.information(self, "Saved", f"Islands saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save Islands:\n{str(e)}")

    def _auto_save_island_state(self):
        """Automatically persist island state next to the source image without prompting the user."""
        if self.canvas.original_image is None or not self.image_path:
            return None

        try:
            metadata, mask_stack = self._build_island_state()
        except Exception as e:
            logger.warning("Auto-save skipped: could not build state: %s", e, exc_info=True)
            return None

        base_path, _ = os.path.splitext(self.image_path)
        file_path = f"{base_path}_palette_state.npz"
        try:
            np.savez_compressed(file_path, metadata=json.dumps(metadata), masks=mask_stack)
            logger.info("Auto-saved palette Islands to %s", file_path)
            return file_path
        except Exception:
            logger.warning("Auto-save failed", exc_info=True)
            return None

    def load_island_state(self):
        """Load islands, ranges, and masks from a saved NPZ state."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Load Palette Islands"),
            "",
            "Palette Islands (*.npz);;All Files (*)",
        )

        if not file_path:
            return

        try:
            data = np.load(file_path, allow_pickle=False)
            raw_meta = data.get("metadata")
            if raw_meta is None:
                raise ValueError("Missing metadata")
            if hasattr(raw_meta, "item"):
                raw_meta = raw_meta.item()
            metadata = json.loads(str(raw_meta))
            mask_stack = data.get("masks")
            if mask_stack is None:
                raise ValueError("Missing masks")
            mask_stack = mask_stack.astype(bool)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load Islands:\n{str(e)}")
            return

        if self.canvas.original_image is None:
            QMessageBox.warning(
                self,
                "Image Required",
                "Load the original image before loading Islands so sizes match.",
            )
            return

        height, width = self.canvas.original_image.shape[:2]
        if metadata.get("width") != width or metadata.get("height") != height:
            QMessageBox.critical(
                self,
                "Size Mismatch",
                "Saved Islands were created for a different image size. Load the matching image first.",
            )
            return

        # Rebuild islands and masks
        self.islands = []
        self.island_list.clear()
        self.canvas.all_masks.clear()

        islands = metadata.get("islands", [])
        if mask_stack.shape[0] != len(islands):
            QMessageBox.critical(self, "Error", "Mask count does not match saved islands.")
            return

        for idx, entry in enumerate(islands):
            name = entry.get("name", f"Island_{idx+1}")
            gs = int(entry.get("gray_start", 0))
            ge = int(entry.get("gray_end", 0))
            self.islands.append((name, gs, ge))
            self.island_list.addItem(f"{name} [{gs}-{ge}] ({ge - gs + 1} colors)")
            self.canvas.all_masks[name] = mask_stack[idx]

        # Restore current island selection
        current_name = metadata.get("current_island") or (self.islands[0][0] if self.islands else None)
        if current_name and current_name in self.canvas.all_masks:
            target_index = next((i for i, (n, _, _) in enumerate(self.islands) if n == current_name), 0)
            self.island_list.setCurrentRow(target_index)
            self.canvas.set_current_island(current_name)
            _, gs, ge = self.islands[target_index]
            self.set_island_size(ge - gs + 1)
            self.gray_range_label.setText(f"[{gs}-{ge}]")
        else:
            self.canvas.set_current_island(None)
            self.island_list.setCurrentRow(-1)

        self.update_available_space()
        self.canvas.update_display()

    @staticmethod
    def _has_transparent_pixels(img_array: np.ndarray) -> bool:
        """Return True only if the image actually contains *fully* transparent pixels.

        For this tool we only care about true cutout holes (alpha == 0). Semi-transparent
        pixels (0 < alpha < 255) should still be treated as opaque here so that textures
        with minor edge anti-aliasing are handled via the NIF+UV path when desired.
        """
        if img_array is None or img_array.ndim != 3 or img_array.shape[2] < 4:
            return False
        alpha = img_array[:, :, 3]

        # Normalize dtype to avoid surprises and compute min/max for diagnostics
        try:
            alpha_view = alpha.astype(np.uint16, copy=False)
        except Exception:
            alpha_view = alpha

        alpha_min = int(np.min(alpha_view))
        alpha_max = int(np.max(alpha_view))

        # Debug-level logging to help diagnose unexpected transparency detection
        try:
            logger.debug(
                "PaletteLUTGenerator._has_transparent_pixels: alpha_min=%d alpha_max=%d shape=%s dtype=%s",
                alpha_min,
                alpha_max,
                alpha.shape,
                str(alpha.dtype),
            )
        except Exception:
            # Logging is best-effort only; never break transparency checks.
            pass

        # Treat as transparent only if at least one pixel is actually == 0.
        # Using the precomputed minimum avoids per-pixel boolean scans.
        return alpha_min == 0

    def set_island_size(self, size):
        """Set the current island size"""
        self.current_island_size = size

        # Update button states
        self.size_16_btn.setChecked(size == 16)
        self.size_32_btn.setChecked(size == 32)
        self.size_64_btn.setChecked(size == 64)
        self.size_128_btn.setChecked(size == 128)

        # Update available space display
        self.update_available_space()

    def update_available_space(self):
        """Update the available space label and button states"""
        # Calculate used space
        used_space = sum(gray_end - gray_start + 1 for _, gray_start, gray_end in self.islands)
        available_space = cfg.get(cfg.ci_default_palette_size) - used_space

        self.available_space_label.setText(f"Available: {available_space} gray values")

        # Enable/disable size buttons based on available space
        self.size_16_btn.setEnabled(available_space >= 16)
        self.size_32_btn.setEnabled(available_space >= 32)
        self.size_64_btn.setEnabled(available_space >= 64)
        self.size_128_btn.setEnabled(available_space >= 128)

        # If current selection is too large, select largest available
        if self.current_island_size > available_space:
            for size in [128, 64, 32, 16]:
                if size <= available_space:
                    self.set_island_size(size)
                    break

    def add_island(self):
        """Add a new island/Island"""
        if not self._ensure_editable():
            return
        island_count = len(self.islands) + 1
        island_name = f"Island_{island_count}"

        # Get current island size setting
        island_size = self.current_island_size

        # Calculate used space
        used_space = sum(gray_end - gray_start + 1 for _, gray_start, gray_end in self.islands)
        available_space = cfg.get(cfg.ci_default_palette_size) - used_space

        # Check if there's enough space
        if island_size > available_space:
            QMessageBox.warning(self, "Warning",
                                f"Not enough space! Requested: {island_size}, Available: {available_space}")
            return

        # Calculate default gray range based on existing islands
        if self.islands:
            last_island_end = self.islands[-1][2]
            gray_start = last_island_end + 1
        else:
            gray_start = 0

        gray_end = gray_start + island_size - 1

        if gray_end > cfg.get(cfg.ci_default_palette_size) - 1:
            QMessageBox.warning(self, "Warning", f"Maximum gray value range (0-{cfg.get(cfg.ci_default_palette_size) - 1}) reached!")
            return

        self.islands.append((island_name, gray_start, gray_end))
        self.island_list.addItem(f"{island_name} [{gray_start}-{gray_end}] ({island_size} colors)")

        # Update display
        self.gray_range_label.setText(f"[{gray_start}-{gray_end}]")

        # Update available space
        self.update_available_space()

        # Select the new island
        self.island_list.setCurrentRow(len(self.islands) - 1)

    def on_pick_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select NIF", "", "NIF (*.nif)"
        )
        if not file_path:
            return

        self.model_path = Path(file_path)
        self.card_pick_model.setContent(str(self.model_path))
        self.uv_entries = build_uv_entries_for_nif(self.model_path)
        self.card_uv_set.combox.clear()
        if self.uv_entries:
            items = [label for (_, _, label) in self.uv_entries]
            self.card_uv_set.combox.addItems(items)
        else:
            self.card_uv_set.combox.addItems(["UV 0"])
        self.card_uv_set.combox.setCurrentIndex(0)
        self.selected_uv_index = 0

        # If an opaque image was pending, try to apply the mask now
        if self.pending_image_path or (self._base_image_array is not None and self.image_path is None):
            self._apply_nif_mask_and_load()

    def on_uv_changed(self, idx: int):
        self.selected_uv_index = idx
        # Clear all islands and masks when UV selection changes
        self._reset_islands_and_masks()
        # Reapply nif mask if we have a pending/opaque base
        if self.image_path is None and (self.pending_image_path or self._base_image_array is not None):
            self._apply_nif_mask_and_load()

    def _apply_nif_mask_and_load(self):
        """Apply UV mask from selected NIF to build transparency for an opaque image."""
        if self._base_image_array is None and self.pending_image_path:
            try:
                pil_img = Image.open(self.pending_image_path).convert("RGBA")
                self._base_image_array = np.array(pil_img)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load pending image: {e}")
                return

        if self._base_image_array is None:
            return

        if not self.model_path:
            QMessageBox.information(self, "NIF Required", "Select a NIF to derive transparency.")
            return

        h, w = self._base_image_array.shape[:2]
        mask_img = build_mask_from_nif(
            self.model_path,
            self.uv_entries,
            self.selected_uv_index,
            w,
            h,
            scale_uvs=False,
            wrap=True,
        )

        if mask_img is None:
            QMessageBox.warning(self, "Mask Failed", "Could not build UV mask from the selected NIF.")
            return

        mask_arr = np.array(mask_img, dtype=np.uint8)
        base_alpha = self._base_image_array[:, :, 3]
        alpha = np.where(mask_arr > 0, base_alpha, 0).astype(np.uint8)
        rgba = self._base_image_array.copy()
        rgba[:, :, 3] = alpha

        # Apply to canvas
        self.canvas.set_image_array(rgba, self._quantized_image, self._quantized_image_array)
        self.image_path = self.pending_image_path or (self.model_path.name + "_uv.png")
        self.pending_image_path = None
        self._nontransparent_source = False
        self._set_edit_lock(False)
        # Reset selections
        self._reset_islands_and_masks()

    def _reset_islands_and_masks(self):
        self.islands = []
        self.island_list.clear()
        self.canvas.all_masks.clear()
        self.canvas.current_island = None
        self.gray_range_label.setText("[0-15]")
        self.update_available_space()
        self.canvas.update_display()

    def remove_island(self):
        """Remove the selected island"""
        if not self._ensure_editable():
            return
        current_row = self.island_list.currentRow()
        if current_row >= 0:
            island_name = self.islands[current_row][0]
            self.islands.pop(current_row)
            self.island_list.takeItem(current_row)

            # Remove from canvas
            if island_name in self.canvas.all_masks:
                del self.canvas.all_masks[island_name]

            # Update available space
            self.update_available_space()

            self.canvas.update_display()

    def on_island_selected(self, current, previous):
        """Handle island selection change"""
        if current:
            row = self.island_list.row(current)
            if row >= 0 and row < len(self.islands):
                island_name, gray_start, gray_end = self.islands[row]
                self.canvas.set_current_island(island_name)
                island_size = gray_end - gray_start + 1
                self.set_island_size(island_size)
                self.gray_range_label.setText(f"[{gray_start}-{gray_end}]")

    def clear_current_selection(self):
        """Clear the current island's selection"""
        if not self._ensure_editable():
            return
        self.canvas.clear_current_mask()

    def on_palette_size_changed(self, text):
        """Handle palette size selection from dropdown."""
        try:
            new_size = int(text.value)
        except ValueError:
            return

        if new_size not in (128, 64, 32):
            return

        used_space = sum(gray_end - gray_start + 1 for _, gray_start, gray_end in self.islands)
        max_gray_end = max((gray_end for _, _, gray_end in self.islands), default=-1)

        # Prevent shrinking below existing allocations
        if used_space > new_size or max_gray_end >= new_size:
            self._reset_islands_and_masks()
            return

        self._palette_size_previous = new_size
        self.update_available_space()

    # --- Shared Lab utilities for region similarity ---
    def _lab_image(self, rgb_image: np.ndarray) -> np.ndarray:
        """Convert an RGB image array to Lab (float32)."""
        return np.array(Image.fromarray(rgb_image, mode='RGB').convert('LAB'), dtype=np.float32)

    def _lab_histogram(self, lab_pixels: np.ndarray) -> np.ndarray:
        """Compute 8x8x8 (512-bin) Lab histogram normalized to 1."""
        l_bins = np.clip((lab_pixels[:, 0] / 100.0 * 8).astype(np.int32), 0, 7)
        a_bins = np.clip(((lab_pixels[:, 1] + 128.0) / 255.0 * 8).astype(np.int32), 0, 7)
        b_bins = np.clip(((lab_pixels[:, 2] + 128.0) / 255.0 * 8).astype(np.int32), 0, 7)
        idx = l_bins * 64 + a_bins * 8 + b_bins
        hist = np.bincount(idx, minlength=512).astype(np.float32)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum
        return hist

    def _histogram_intersection_distance(self, h1: np.ndarray, h2: np.ndarray) -> float:
        """Histogram intersection distance (0 same, 1 disjoint)."""
        return 1.0 - float(np.minimum(h1, h2).sum())

    def _mean_lab_distance(self, m1: np.ndarray, m2: np.ndarray) -> float:
        """Perceptual Lab distance normalized to ~0..1."""
        return float(np.linalg.norm(m1 - m2) / 100.0)

    def _lab_bin_center(self, bin_index: int) -> np.ndarray:
        l_bin = bin_index // 64
        rem = bin_index % 64
        a_bin = rem // 8
        b_bin = rem % 8
        l_center = (l_bin + 0.5) * (100.0 / 8.0)
        a_center = (a_bin + 0.5) * (255.0 / 8.0) - 128.0
        b_center = (b_bin + 0.5) * (255.0 / 8.0) - 128.0
        return np.array([l_center, a_center, b_center], dtype=np.float32)

    def _dominant_bin_guard(self, comp_hist: np.ndarray, grp_hist: np.ndarray, share_gap_max: float = 0.25,
                            center_tol: float = 15.0) -> bool:
        """Check dominant-bin compatibility between two histograms in Lab bin space."""
        top_bin_comp = int(comp_hist.argmax())
        top_share_comp = float(comp_hist[top_bin_comp])
        top_bin_grp = int(grp_hist.argmax())
        top_share_grp = float(grp_hist[top_bin_grp])

        if top_bin_comp != top_bin_grp:
            return False

        share_gap = abs(top_share_comp - top_share_grp)
        if share_gap > share_gap_max:
            return False

        comp_center = self._lab_bin_center(top_bin_comp)
        grp_center = self._lab_bin_center(top_bin_grp)
        return float(np.linalg.norm(comp_center - grp_center)) <= center_tol

    def magic_wand_select(self):
        """Find regions similar to current island and add them if unassigned."""
        if self.canvas.original_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return

        if not self.canvas.current_island:
            QMessageBox.warning(self, "Warning", "Please select an island first!")
            return

        current_island = self.canvas.current_island
        if current_island not in self.canvas.all_masks:
            QMessageBox.warning(self, "Warning", "Current island has no mask yet!")
            return

        base_mask = self.canvas.all_masks[current_island]
        if not base_mask.any():
            QMessageBox.warning(self, "Warning", "Current island has no selected pixels to match against.")
            return

        alpha = self.canvas.original_image[:, :, 3]
        rgb = self.canvas.original_image[:, :, :3]
        # Convert once to Lab (perceptual) for comparisons
        lab_image = self._lab_image(rgb)
        non_transparent = alpha > 0

        # Union of already assigned pixels across all islands
        assigned_mask = np.zeros(non_transparent.shape, dtype=bool)
        for name, mask in self.canvas.all_masks.items():
            if mask is not None:
                assigned_mask |= mask

        # Build reference descriptors from the current island selection (Lab)
        ref_lab = lab_image[base_mask]
        ref_hist = self._lab_histogram(ref_lab)
        ref_mean_lab = ref_lab.mean(axis=0)

        # Connected components over non-transparent pixels (4-connected)
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
        labels, num = ndimage.label(non_transparent, structure=structure)
        slices = ndimage.find_objects(labels)

        hist_weight = 0.75
        dist_threshold = (cfg.get(cfg.grouping_threshold) / 100.0)

        added_pixels = 0

        for lbl in range(1, num + 1):
            sl = slices[lbl - 1]
            if sl is None:
                continue
            lbl_region = labels[sl]
            region_mask = lbl_region == lbl

            # Skip if any pixel is already assigned to an island
            if assigned_mask[sl][region_mask].any():
                continue

            pixel_count = int(region_mask.sum())
            if pixel_count < 8:
                continue

            region_lab = lab_image[sl][region_mask]
            if region_lab.size == 0:
                continue

            hist = self._lab_histogram(region_lab)
            mean_lab = region_lab.mean(axis=0)

            d_hist = self._histogram_intersection_distance(hist, ref_hist)
            d_mean = self._mean_lab_distance(mean_lab, ref_mean_lab)
            score = hist_weight * d_hist + (1.0 - hist_weight) * d_mean

            if score <= dist_threshold and self._dominant_bin_guard(hist, ref_hist):
                # Accept this region into the current island
                self.canvas.all_masks[current_island][sl][region_mask] = True
                added_pixels += pixel_count

        if added_pixels == 0:
            QMessageBox.information(self, "Magic Wand", "No additional matching regions found.")
        else:
            QMessageBox.information(self, "Magic Wand", f"Added {added_pixels} pixels to {current_island}.")
            self.canvas.update_display()

    def add_remaining_to_current(self):
        """Add all unassigned non-transparent pixels to the current island."""
        if self.canvas.original_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return

        if not self.canvas.current_island:
            QMessageBox.warning(self, "Warning", "Please select an island first!")
            return

        alpha = self.canvas.original_image[:, :, 3]
        non_transparent = alpha > 0
        if not non_transparent.any():
            QMessageBox.information(self, "Add Remaining", "Image has no opaque pixels to add.")
            return

        # Build assigned mask across all islands
        assigned_mask = np.zeros(non_transparent.shape, dtype=bool)
        for mask in self.canvas.all_masks.values():
            if mask is not None:
                assigned_mask |= mask

        remaining = non_transparent & ~assigned_mask
        if not remaining.any():
            QMessageBox.information(self, "Add Remaining", "No remaining unassigned pixels found.")
            return

        current_name = self.canvas.current_island
        if current_name not in self.canvas.all_masks:
            self.canvas.all_masks[current_name] = np.zeros(non_transparent.shape, dtype=bool)

        self.canvas.all_masks[current_name] |= remaining
        added_pixels = int(remaining.sum())
        QMessageBox.information(self, "Add Remaining", f"Added {added_pixels} pixels to {current_name}.")
        self.canvas.update_display()

    def auto_create_islands(self):
        """Automatically detect non-transparent regions and group by color similarity."""
        if self.canvas.original_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return

        alpha = self.canvas.original_image[:, :, 3]
        rgb = self.canvas.original_image[:, :, :3]
        # Convert once to Lab (perceptual) for grouping comparisons
        lab_image = self._lab_image(rgb)
        non_transparent = alpha > 0

        if not non_transparent.any():
            QMessageBox.warning(self, "Warning", "Image has no opaque pixels.")
            return

        # Connected components over non-transparent pixels (4-connected)
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
        labels, num = ndimage.label(non_transparent, structure=structure)

        if num == 0:
            QMessageBox.warning(self, "Warning", "No regions detected.")
            return

        # Prepare slices to avoid full-image per-component scans
        slices = ndimage.find_objects(labels)

        components = []
        min_pixels = 8  # ignore extremely small specs

        for lbl in range(1, num + 1):
            sl = slices[lbl - 1]
            if sl is None:
                continue
            lbl_region = labels[sl]
            region_mask = lbl_region == lbl
            pixel_count = int(region_mask.sum())
            if pixel_count < min_pixels:
                continue

            region_rgb = rgb[sl][region_mask]
            region_lab = lab_image[sl][region_mask]
            if region_rgb.size == 0 or region_lab.size == 0:
                continue

            unique_colors = np.unique(region_rgb.reshape(-1, 3), axis=0)

            # Lab histogram (8x8x8 = 512 bins) captures proportion in perceptual space
            hist = self._lab_histogram(region_lab)

            mean_lab = region_lab.mean(axis=0)

            # Store local mask to avoid full-image re-labeling later
            components.append({
                "slice": sl,
                "mask": region_mask,
                "hist": hist,
                "mean_lab": mean_lab,
                "pixels": pixel_count,
                "unique_colors": set(map(tuple, unique_colors.tolist())),
                # Track regions uniformly so later assignment can iterate safely
                "regions": [(sl, region_mask)],
            })

        if not components:
            QMessageBox.warning(self, "Warning", "No sufficiently large regions found.")
            return

        # Reset existing islands and masks
        self.islands = []
        self.island_list.clear()
        self.canvas.all_masks.clear()

        # Determine four islands by dividing the palette into quarters
        desired_islands = 4
        if cfg.get(cfg.ci_default_palette_size) <= 0:
            QMessageBox.warning(self, "Warning", "Palette size must be greater than zero to auto create islands.")
            return

        base_size = cfg.get(cfg.ci_default_palette_size) // desired_islands
        remainder = cfg.get(cfg.ci_default_palette_size) % desired_islands

        island_specs = []
        current_start = 0
        for i in range(desired_islands):
            size = base_size + (1 if i < remainder else 0)
            if size <= 0:
                continue
            gray_start = current_start
            gray_end = current_start + size - 1
            island_specs.append({
                "gray_start": gray_start,
                "gray_end": gray_end,
                "capacity": size
            })
            current_start += size

        if not island_specs:
            QMessageBox.warning(self, "Warning",
                                "Unable to divide the palette into islands with the current palette size.")
            return

        # --- Color-first clustering into up to four distinct groups ---
        hist_weight = 0.75

        def _comp_group_score(comp: dict, ref_hist: np.ndarray, ref_mean: np.ndarray) -> float:
            """Combined histogram/mean distance with a penalty if dominant bins disagree."""
            d_hist = self._histogram_intersection_distance(comp["hist"], ref_hist)
            d_mean = self._mean_lab_distance(comp["mean_lab"], ref_mean)
            base = hist_weight * d_hist + (1.0 - hist_weight) * d_mean
            if not self._dominant_bin_guard(comp["hist"], ref_hist):
                base += 0.25  # discourage mixing different dominant colors
            return base

        # Seed groups with the largest component, then pick farthest colors to maximize diversity
        components_sorted = sorted(components, key=lambda c: c["pixels"], reverse=True)
        seeds: list[dict] = []
        if components_sorted:
            seeds.append(components_sorted[0])
            remaining = components_sorted[1:]
            while len(seeds) < min(desired_islands, len(components_sorted)) and remaining:
                far_idx = None
                far_score = -1.0
                for idx, cand in enumerate(remaining):
                    min_dist = min(_comp_group_score(cand, s["hist"], s["mean_lab"]) for s in seeds)
                    if min_dist > far_score:
                        far_score = min_dist
                        far_idx = idx
                seeds.append(remaining.pop(far_idx))

        # Initialize groups from seeds
        groups: list[dict] = []
        for seed in seeds:
            groups.append({
                "hist_centroid": seed["hist"],
                "mean_lab_centroid": seed["mean_lab"],
                "pixel_total": seed["pixels"],
                "unique_colors": set(seed["unique_colors"]),
                "regions": [(sl, m) for sl, m in seed.get("regions", [(seed["slice"], seed["mask"])])]
            })

        # Assign remaining components to the closest color group
        for comp in components_sorted:
            if comp in seeds:
                continue

            best_idx = None
            best_score = math.inf

            for idx, grp in enumerate(groups):
                score = _comp_group_score(comp, grp["hist_centroid"], grp["mean_lab_centroid"])
                if score < best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                continue

            grp = groups[best_idx]
            total_pixels = grp["pixel_total"] + comp["pixels"]
            grp["hist_centroid"] = (grp["hist_centroid"] * grp["pixel_total"] + comp["hist"] * comp[
                "pixels"]) / total_pixels
            grp["mean_lab_centroid"] = (grp["mean_lab_centroid"] * grp["pixel_total"] + comp["mean_lab"] * comp[
                "pixels"]) / total_pixels
            grp["pixel_total"] = total_pixels
            grp["unique_colors"].update(comp["unique_colors"])
            grp["regions"].extend(comp.get("regions", [(comp["slice"], comp["mask"])]))

        # If we have fewer groups than desired islands, pad with empty groups
        while len(groups) < desired_islands:
            groups.append({
                "hist_centroid": np.zeros(512, dtype=np.float32),
                "mean_lab_centroid": np.zeros(3, dtype=np.float32),
                "pixel_total": 0,
                "unique_colors": set(),
                "regions": []
            })

        # Map color groups to island slots by matching largest color needs to largest capacities
        island_data: list[dict | None] = [None] * len(island_specs)
        groups_sorted_for_capacity = sorted(enumerate(groups), key=lambda t: len(t[1]["unique_colors"]), reverse=True)
        specs_sorted_by_capacity = sorted(enumerate(island_specs), key=lambda t: t[1]["capacity"], reverse=True)

        overflow_flag = False

        for (grp_idx, grp), (spec_idx, spec) in zip(groups_sorted_for_capacity, specs_sorted_by_capacity):
            mask = np.zeros(non_transparent.shape, dtype=bool)
            for sl, m in grp.get("regions", []):
                mask[sl][m] = True

            island_data[spec_idx] = {
                "gray_start": spec["gray_start"],
                "gray_end": spec["gray_end"],
                "capacity": spec["capacity"],
                "unique_colors": set(grp["unique_colors"]),
                "mask": mask,
                "pixel_total": grp["pixel_total"],
            }

            if len(grp["unique_colors"]) > spec["capacity"]:
                overflow_flag = True

        # Fill any remaining island slots with empty masks
        for idx, spec in enumerate(island_specs):
            if island_data[idx] is None:
                island_data[idx] = {
                    "gray_start": spec["gray_start"],
                    "gray_end": spec["gray_end"],
                    "capacity": spec["capacity"],
                    "unique_colors": set(),
                    "mask": np.zeros(non_transparent.shape, dtype=bool),
                    "pixel_total": 0,
                }

        # Ensure all non-transparent pixels end up in some island mask
        combined_mask = np.zeros(non_transparent.shape, dtype=bool)
        for isl in island_data:
            combined_mask |= isl["mask"]

        leftovers = non_transparent & ~combined_mask
        if leftovers.any():
            # Assign leftovers to the island with the most remaining capacity (tie -> later island)
            def remaining_capacity(idx: int) -> tuple[int, int]:
                isl = island_data[idx]
                rem = isl["capacity"] - len(isl["unique_colors"])
                return rem, idx

            target_idx = max(range(len(island_data)), key=remaining_capacity)
            target = island_data[target_idx]
            target["mask"] |= leftovers

            leftover_colors = set(map(tuple, rgb[leftovers].reshape(-1, 3)))
            target["unique_colors"].update(leftover_colors)
            if len(target["unique_colors"]) > target["capacity"]:
                overflow_flag = True

        # Build final islands and UI list
        self.islands = []
        self.island_list.clear()

        for idx, isl in enumerate(island_data, start=1):
            island_name = f"AutoIsland_{idx}"
            gray_start, gray_end = isl["gray_start"], isl["gray_end"]
            self.islands.append((island_name, gray_start, gray_end))
            self.canvas.all_masks[island_name] = isl["mask"]
            self.island_list.addItem(f"{island_name} [{gray_start}-{gray_end}] ({gray_end - gray_start + 1} colors)")

        if not self.islands:
            QMessageBox.warning(self, "Warning", "Could not allocate any islands within available grayscale range.")
            return

        if overflow_flag:
            QMessageBox.warning(self, "Warning",
                                "Some islands exceeded their ideal unique-color capacity; colors were consolidated to fit all pixels.")

        # Select first island and refresh UI
        self.island_list.setCurrentRow(0)
        self.canvas.set_current_island(self.islands[0][0])
        first_island_size = self.islands[0][2] - self.islands[0][1] + 1
        self.set_island_size(first_island_size)
        self.update_available_space()
        self.canvas.update_display()

    def generate_both(self):
        """Generate both grayscale atlas and LUT palette"""
        if self.canvas.original_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return

        if not self.image_path:
            QMessageBox.warning(self, "Warning", "No image path available!")
            return

        try:
            # Get base path from original image
            import os

            source_is_dds = self.image_path.lower().endswith('.dds')
            output_extension = ".dds" if source_is_dds else ".png"

            directory = os.path.dirname(self.image_path)
            base_path  = os.path.splitext(os.path.basename(self.image_path))[0]

            grayscale_path = os.path.join(
                directory, f"{base_path }_grayscale{output_extension}"
            )
            palette_path = os.path.join(
                directory, f"{base_path }_palette{output_extension}"
            )
            applied_path = os.path.join(
                directory, f"{base_path }_applied{output_extension}"
            )

            # Game interpolates palettes <256 up to 256 entries. Keep the palette width
            # at the chosen size (typically 128) but scale grayscale indices to 0-255 so
            # they line up with the interpolated palette the game will use.
            if cfg.get(cfg.ci_default_palette_size) <= 1:
                palette_to_game_scale = 1.0
            else:
                palette_to_game_scale = 255.0 / float(cfg.get(cfg.ci_default_palette_size) - 1)

            def _scale_range_to_game(start: int, end: int) -> tuple[int, int]:
                """Map a palette-range [start,end] to the game-expected 0-255 range."""
                gs = int(round(start * palette_to_game_scale))
                ge = int(round(end * palette_to_game_scale))
                gs = max(0, min(255, gs))
                ge = max(0, min(255, ge))
                if ge < gs:
                    ge = gs
                return gs, ge

            # Get image dimensions
            height, width = self.canvas.original_image.shape[:2]
            alpha_channel = self.canvas.original_image[:, :, 3]
            non_transparent = alpha_channel > 0

            # Build assigned mask across all islands
            assigned_mask = np.zeros((height, width), dtype=bool)
            for mask in self.canvas.all_masks.values():
                if mask is not None:
                    assigned_mask |= mask

            remaining_pixels = non_transparent & ~assigned_mask

            # Determine remaining palette space
            used_space = sum(gray_end - gray_start + 1 for _, gray_start, gray_end in self.islands)
            available_space = cfg.get(cfg.ci_default_palette_size) - used_space

            added_remaining_island = None

            if available_space > 0:
                # Add a real island that spans the remaining palette range
                gray_start = self.islands[-1][2] + 1 if self.islands else 0
                gray_end = cfg.get(cfg.ci_default_palette_size) - 1

                # Ensure unique island name
                base_idx = len(self.islands) + 1
                existing_names = {name for name, _, _ in self.islands}
                island_name = f"Island_{base_idx}"
                while island_name in existing_names:
                    base_idx += 1
                    island_name = f"Island_{base_idx}"

                self.islands.append((island_name, gray_start, gray_end))
                self.island_list.addItem(
                    f"{island_name} [{gray_start}-{gray_end}] ({gray_end - gray_start + 1} colors)")
                self.canvas.all_masks.setdefault(island_name, np.zeros(non_transparent.shape, dtype=bool))
                if remaining_pixels.any():
                    self.canvas.all_masks[island_name] |= remaining_pixels
                added_remaining_island = island_name
                self.update_available_space()
            else:
                # No palette space left: merge remaining pixels into the largest (tie -> latest) island
                if remaining_pixels.any() and self.islands:
                    target_idx = -1
                    target_size = -1
                    for idx, (_, gs, ge) in enumerate(self.islands):
                        size = ge - gs + 1
                        if size > target_size or (size == target_size and idx > target_idx):
                            target_size = size
                            target_idx = idx

                    if target_idx >= 0:
                        target_name = self.islands[target_idx][0]
                        if target_name not in self.canvas.all_masks:
                            self.canvas.all_masks[target_name] = np.zeros(non_transparent.shape, dtype=bool)
                        self.canvas.all_masks[target_name] |= remaining_pixels

            # Recompute assigned/unassigned after potential allocation
            all_selected = np.zeros((height, width), dtype=bool)
            for mask in self.canvas.all_masks.values():
                if mask is not None:
                    all_selected |= mask

            unselected_pixels = non_transparent & ~all_selected

            # Generate grayscale atlas
            grayscale_output = np.zeros((height, width), dtype=np.uint8)
            rgb_array = self.canvas.original_image[:, :, :3]

            # Calculate luminosity once
            luminosity = (0.299 * rgb_array[:, :, 0] +
                          0.587 * rgb_array[:, :, 1] +
                          0.114 * rgb_array[:, :, 2])

            # Dictionary to store original colors for each island
            island_colors = {}

            # Process each island (including the newly added remainder island if any)
            for island_name, gray_start, gray_end in self.islands:
                if island_name not in self.canvas.all_masks:
                    continue

                mask = self.canvas.all_masks[island_name]
                if not mask.any():
                    print(f"Warning: {island_name} has no selection")
                    continue

                # Get luminosity values only for this island
                island_luminosity = luminosity[mask]

                if len(island_luminosity) == 0:
                    continue

                # Normalize to gray range
                lum_min = island_luminosity.min()
                lum_max = island_luminosity.max()

                if lum_max - lum_min < 1:
                    lum_max = lum_min + 1

                normalized = (luminosity[mask] - lum_min) / (lum_max - lum_min)
                remapped_palette_space = gray_start + normalized * (gray_end - gray_start)
                # Scale to game (0-255) index space
                remapped = remapped_palette_space * palette_to_game_scale

                grayscale_output[mask] = remapped.astype(np.uint8)

                # Store original RGB colors mapped to their grayscale values
                island_rgb = rgb_array[mask]
                island_gray = remapped_palette_space.astype(np.uint8)

                # For each gray value in this island's range, collect all colors
                color_map = {}
                for gray_val in range(gray_start, gray_end + 1):
                    color_map[gray_val] = []

                for rgb, gray in zip(island_rgb, island_gray):
                    color_map[gray].append(rgb)

                # Average colors for each gray value
                averaged_colors = {}
                for gray_val, colors in color_map.items():
                    if colors:
                        averaged_colors[gray_val] = np.mean(colors, axis=0).astype(np.uint8)
                    else:
                        # No color at this gray value, interpolate
                        averaged_colors[gray_val] = None

                island_colors[island_name] = averaged_colors

            # Process any still-unassigned pixels (should be rare)
            if unselected_pixels.any():
                # If we added a remainder island, its mask should be updated already; otherwise, merge into target island
                target_mask_name = added_remaining_island
                if target_mask_name is None:
                    # Choose largest/last island again
                    target_idx = -1
                    target_size = -1
                    for idx, (_, gs, ge) in enumerate(self.islands):
                        size = ge - gs + 1
                        if size > target_size or (size == target_size and idx > target_idx):
                            target_size = size
                            target_idx = idx
                    target_mask_name = self.islands[target_idx][0] if target_idx >= 0 else None

                if target_mask_name:
                    if target_mask_name not in self.canvas.all_masks:
                        self.canvas.all_masks[target_mask_name] = np.zeros(non_transparent.shape, dtype=bool)
                    self.canvas.all_masks[target_mask_name] |= unselected_pixels
                    mask = self.canvas.all_masks[target_mask_name]

                    # Recompute luminosity mapping for these pixels using the target island's gray range
                    target_gray_start, target_gray_end = next(
                        (gs, ge) for name, gs, ge in self.islands if name == target_mask_name)
                    unselected_luminosity = luminosity[unselected_pixels]
                    lum_min = unselected_luminosity.min()
                    lum_max = unselected_luminosity.max()

                    if lum_max - lum_min < 1:
                        lum_max = lum_min + 1

                    normalized = (luminosity[unselected_pixels] - lum_min) / (lum_max - lum_min)
                    remapped_palette_space = target_gray_start + normalized * (target_gray_end - target_gray_start)
                    remapped = remapped_palette_space * palette_to_game_scale
                    grayscale_output[unselected_pixels] = remapped.astype(np.uint8)

                    # Store averaged colors for palette generation
                    unselected_rgb = rgb_array[unselected_pixels]
                    unselected_gray = remapped_palette_space.astype(np.uint8)

                    color_map = {gray_val: [] for gray_val in range(target_gray_start, target_gray_end + 1)}
                    for rgb, gray in zip(unselected_rgb, unselected_gray):
                        color_map[gray].append(rgb)

                    averaged_colors = {}
                    for gray_val, colors in color_map.items():
                        if colors:
                            averaged_colors[gray_val] = np.mean(colors, axis=0).astype(np.uint8)
                        else:
                            averaged_colors[gray_val] = None

                    island_colors.setdefault(target_mask_name, {}).update(averaged_colors)

            # Save grayscale atlas
            def _fill_transparent_with_nearest(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
                """Fill transparent pixels by copying the nearest non-transparent value (fast EDT-based)."""
                if mask is None or img.shape != mask.shape:
                    return img
                if not mask.any():
                    return img

                transparent = ~mask
                if not transparent.any():
                    return img

                # Distance transform: for each pixel, get indices of the nearest non-transparent pixel
                # ndimage.distance_transform_edt returns (distance, indices); indices has shape (ndim, h, w)
                _, nearest_indices = ndimage.distance_transform_edt(transparent, return_indices=True)
                nearest_y, nearest_x = nearest_indices

                filled = img.copy()
                filled[transparent] = img[nearest_y[transparent], nearest_x[transparent]]
                return filled

            # Fill transparent areas by dilating nearest non-transparent values, then BC7-smooth blocks
            grayscale_filled = _fill_transparent_with_nearest(grayscale_output, non_transparent)

            save_image(Image.fromarray(grayscale_filled, mode='L'), grayscale_path)

            # Generate LUT palette from actual colors
            palette_height = 4
            palette_width = cfg.get(cfg.ci_default_palette_size)
            palette = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)

            # First row uses actual averaged colors
            for island_name, gray_start, gray_end in self.islands:
                if island_name not in island_colors:
                    continue

                colors = island_colors[island_name]

                # Fill in the palette with actual colors, interpolating gaps
                for gray_val in range(gray_start, min(gray_end + 1, palette_width)):
                    if colors.get(gray_val) is not None:
                        palette[0, gray_val] = colors[gray_val]
                    else:
                        # Interpolate from nearest known colors
                        # Find previous and next known colors
                        prev_val, next_val = None, None
                        for g in range(gray_val - 1, gray_start - 1, -1):
                            if colors.get(g) is not None:
                                prev_val = g
                                break
                        for g in range(gray_val + 1, gray_end + 1):
                            if colors.get(g) is not None:
                                next_val = g
                                break

                        if prev_val is not None and next_val is not None:
                            # Interpolate
                            t = (gray_val - prev_val) / (next_val - prev_val)
                            color = (1 - t) * colors[prev_val] + t * colors[next_val]
                            palette[0, gray_val] = color.astype(np.uint8)
                        elif prev_val is not None:
                            palette[0, gray_val] = colors[prev_val]
                        elif next_val is not None:
                            palette[0, gray_val] = colors[next_val]

            # Copy actual averaged colors into all rows
            for theme_row in range(1, palette_height):
                palette[theme_row, :] = palette[0, :]

            # Save palette
            palette_img = Image.fromarray(palette, mode='RGB')
            save_image(palette_img, palette_path, True)

            # Save image with palette applied to grayscale (preserving alpha)
            alpha_img = np.where(non_transparent, 255, 0).astype(np.uint8)
            grey_with_alpha = np.dstack([grayscale_filled, alpha_img])
            grey_pil = Image.fromarray(grey_with_alpha, mode='LA')
            applied_img = apply_palette_to_greyscale(palette_img, grey_pil)
            save_image(applied_img, applied_path, True)

            QMessageBox.information(self, "Success",
                                    f"Files generated successfully!\n\n"
                                    f"Grayscale: {grayscale_path}\n"
                                    f"Palette: {palette_path}\n"
                                    f"Palette applied preview: {applied_path}\n\n"
                                    f"Total Islands Processed: {len(self.islands)}\n"
                                    f"Unselected pixels: {'Yes' if unselected_pixels.any() else 'None'}\n\n")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate files:\n{str(e)}")
            import traceback
            traceback.print_exc()
