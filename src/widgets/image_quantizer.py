import os
from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QLabel, QFileDialog, QGridLayout, QMessageBox, QVBoxLayout
)
from qfluentwidgets import (
    PushSettingCard,
    PrimaryPushButton,
    InfoBar, ConfigItem,
    FluentIcon as FIF, SwitchSettingCard,

)

from help.quantize_help import QuantizeHelp
from settings.quant_settings import QuantSettings
from src.palette.palette_engine import load_image, quantize_image, reduce_colors_lab_de00_with_hue_balance, remap_rgb_array_to_representatives
from src.utils.appconfig import cfg
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger
from utils.cards import RadioSettingCard


class ImageQuantizerWidget(BaseWidget):
    """
    Simple UI to load an image, quantize it using configured method & palette size,
    preview original vs. quantized side-by-side, and save the quantized image.
    """

    def __init__(self, parent: Optional[QWidget], text: str = "Image Quantizer"):
        super().__init__(text=text, parent=parent, vertical=True)
        self.setObjectName("Image-Quantizer")

        # State
        self.current_image_path: Optional[str] = None
        self.original_pil: Optional[Image.Image] = None
        self.quantized_pil: Optional[Image.Image] = None
        self.src_cfg = ConfigItem("quant", "image", "")

        self.pick_image_card = PushSettingCard(
            self.tr("Select Image"),
            CustomIcons.IMAGE.icon(stroke=True),
            self.tr("Select Image"),
            self.src_cfg.value or ""
        )

        self.palette_size_card = RadioSettingCard(
            cfg.ci_default_palette_size,
            CustomIcons.WIDTH.icon(),
            self.tr("Colors"),
            self.tr("Number of colors to quantize to"),
            texts=["256", "128", "64", "32"],
            parent=self
        )

        self.method_card = RadioSettingCard(
            cfg.ci_default_quant_method,
            CustomIcons.QUANT.icon(),
            self.tr("Quantization Method"),
            self.tr("How do we reduce the color palette of the base images"),
            texts=[
                "median_cut",
                "max_coverage",
                "fast_octree",
                "libimagequant",
                "kmeans_adaptive",
                "uniform"
            ],
            parent=self
        )

        # self.advanced_quant = SwitchSettingCard(
        #     icon=FIF.CUT,
        #     title=self.tr("Oversample Colors - Uses LAB Reduction to keep more unique colors"),
        #     configItem=cfg.ci_advanced_quant
        # )

        self.addToFrame(self.palette_size_card)
        self.addToFrame(self.method_card)
        # self.addToFrame(self.advanced_quant)

        self.pick_image_card.clicked.connect(self._on_pick_image)
        self.addToFrame(self.pick_image_card)

        # Two preview labels inside a grid
        self.original_label = QLabel(self.tr("Original preview"))
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setMinimumSize(450, 450)
        self.original_label.setStyleSheet("border: 1px dashed gray;")

        self.quantized_label = QLabel(self.tr("Quantized preview"))
        self.quantized_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.quantized_label.setMinimumSize(450, 450)
        self.quantized_label.setStyleSheet("border: 1px dashed gray;")

        grid = QGridLayout()
        # grid.setContentsMargins(8, 8, 8, 8)
        # grid.setHorizontalSpacing(16)
        # grid.setVerticalSpacing(8)
        grid.addWidget(QLabel(self.tr("Original")), 0, 0)
        grid.addWidget(QLabel(self.tr("Quantized")), 0, 1)
        grid.addWidget(self.original_label, 1, 0)
        grid.addWidget(self.quantized_label, 1, 1)

        container = QWidget()
        container.setLayout(grid)

        self.addToFrame(container)

        self.btn_quantize = PrimaryPushButton(icon=FIF.RIGHT_ARROW, text=self.tr("Quantize"))
        self.btn_quantize.clicked.connect(self._on_quantize)

        self.btn_save = PrimaryPushButton(icon=FIF.SAVE, text=self.tr("Save"))
        self.btn_save.clicked.connect(self._on_save)
        self.btn_save.setEnabled(False)

        self.buttons_layout.addWidget(self.btn_quantize, stretch=1)
        self.addButtonBarToBottom(self.btn_save)

        self.settings_widget = QuantSettings(self)
        self.settings_drawer.addWidget(self.settings_widget)
        self.help_widget = QuantizeHelp(self)
        self.help_drawer.addWidget(self.help_widget)


    # ----------------------------- HELPERS ----------------------------- #
    def _wrap_in_group(self, title: str, layout: QVBoxLayout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        w.setProperty("groupTitle", title)
        return w

    @staticmethod
    def _pil_to_pixmap(pil_image: Image.Image) -> QPixmap:
        try:
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            data = pil_image.tobytes("raw", "RGB")
            qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(qimage)
        except Exception as e:
            logger.error(f"Error converting PIL to QPixmap: {e}")
            return QPixmap(100, 100)

    def _display_on_label(self, img: Image.Image, label: QLabel):
        pix = self._pil_to_pixmap(img)
        scaled = pix.scaled(
            label.width() - 20,
            label.height() - 20,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled)
        label.setText("")

    # ----------------------------- SLOTS ----------------------------- #
    def _on_pick_image(self):
        last_dir = os.path.dirname(self.current_image_path) if self.current_image_path else os.path.expanduser("")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Image"),
            last_dir,
            self.tr("Images (*.png *.jpg *.jpeg *.dds *.bmp *.tga);;All Files (*.*)")
        )
        if not file_path:
            return
        self.current_image_path = file_path
        try:
            self.src_cfg.value = file_path
            self.pick_image_card.setContent(file_path)
            self.original_pil = load_image(file_path, cfg.get(cfg.texconv_file))
            self._display_on_label(self.original_pil, self.original_label)
            self.quantized_pil = None
            self.quantized_label.setPixmap(QPixmap())
            self.quantized_label.setText(self.tr("Quantized preview"))
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            QMessageBox.critical(self, self.tr("Error"), self.tr(f"Failed to load image: {e}"))

    def _on_quantize(self):
        if not self.original_pil:
            QMessageBox.warning(self, self.tr("Warning"), self.tr("Please select an image first."))
            return
        # Use method from cfg
        method = cfg.get(cfg.ci_default_quant_method).value if hasattr(cfg.get(cfg.ci_default_quant_method), 'value') else cfg.get(cfg.ci_default_quant_method)

        # Show parent mask progress
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'show_progress'):
            try:
                p.show_progress()
            except Exception:
                pass
        try:
            q_img, info = quantize_image(self.original_pil, method)
            # Mid-progress update
            try:
                p = getattr(self, 'parent', None)
                if p and hasattr(p, 'update_progress'):
                    p.update_progress(50)
            except Exception:
                pass
            # Ensure we have an RGB to show as preview
            rgb = q_img.convert('RGB')
            arr = np.array(rgb)
            # Compute unique colors and reduce if exceeding configured palette size
            unique, counts = np.unique(arr.reshape(-1, 3), axis=0, return_counts=True)
            target = int(cfg.get(cfg.ci_default_palette_size))
            before = len(unique)
            if before > target:
                logger.debug(f"ImageQuantizer: reducing {before}→{target} with LAB/ΔE00 + hue balancing")
                kept_reps, color_map, pad_candidates = reduce_colors_lab_de00_with_hue_balance(unique, counts, target)
                arr = remap_rgb_array_to_representatives(arr, color_map)
                rgb = Image.fromarray(arr.astype('uint8'), 'RGB')
                after = len(np.unique(arr.reshape(-1, 3), axis=0))
                logger.debug(f"ImageQuantizer: post-reduction unique colors = {after}")
            else:
                after = before
            self.quantized_pil = rgb
            self._display_on_label(self.quantized_pil, self.quantized_label)
            self.btn_save.setEnabled(True)
            InfoBar.success(
                title=self.tr("Quantization complete"),
                content=self.tr(f"{info.get('description', method)}; colors: {before}→{after} (target {target})"),
                duration=3000,
                parent=self,
            )
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            QMessageBox.critical(self, self.tr("Error"), self.tr(f"Quantization failed: {e}"))
        finally:
            p = getattr(self, 'parent', None)
            if p and hasattr(p, 'complete_loader'):
                try:
                    p.complete_loader()
                except Exception:
                    pass

    def _on_save(self):
        if not self.quantized_pil:
            QMessageBox.information(self, self.tr("Info"), self.tr("No quantized image to save."))
            return
        base_dir = os.path.dirname(self.current_image_path) if self.current_image_path else os.path.expanduser("~")
        base_name = os.path.splitext(os.path.basename(self.current_image_path or "image"))[0]
        default_name = f"{base_name}_quantized.png"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save Quantized Image"),
            os.path.join(base_dir, default_name),
            self.tr("PNG Image (*.png)")
        )
        if not save_path:
            return
        try:
            # Ensure PNG extension
            if not save_path.lower().endswith('.png'):
                save_path += '.png'
            self.quantized_pil.save(save_path, format='PNG')
            InfoBar.success(
                title=self.tr("Saved"),
                content=self.tr(f"Saved to: {save_path}"),
                duration=3000,
                parent=self,
            )
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            QMessageBox.critical(self, self.tr("Error"), self.tr(f"Failed to save image: {e}"))
