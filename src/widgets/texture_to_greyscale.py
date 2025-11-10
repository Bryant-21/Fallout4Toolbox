import os
from typing import List, Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QLabel, QFileDialog, QGridLayout, QMessageBox, QVBoxLayout
)
from qfluentwidgets import (
    PushSettingCard,
    PrimaryPushButton,
    InfoBar,
    FluentIcon as FIF,
    PushButton
)

from help.convertpalette_help import PaletteConvertHelp
from settings.palette_settings import PaletteSettings
from src.utils.helpers import BaseWidget
from src.utils.logging_utils import logger
from src.utils.appconfig import cfg
from src.utils.icons import CustomIcons


class ConvertToPaletteWorker(QThread):
    progress = Signal(int, int)
    info = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, palette_path: str, texture_paths: List[str], output_dir: Optional[str]):
        super().__init__()
        self.palette_path = palette_path
        self.texture_paths = texture_paths
        self.output_dir = output_dir

    def run(self):
        try:
            if not self.texture_paths:
                self.finished.emit({'processed': 0, 'skipped': 0, 'failed': 0})
                return
            # Load palette image once
            palette_image = load_image(self.palette_path, format='RGB')
            palette_p_image = palette_image.convert('P', palette=Image.Palette.ADAPTIVE)
            processed = skipped = failed = 0
            first_bw = None
            first_pal = None
            total = len(self.texture_paths)
            for i, tex_path in enumerate(self.texture_paths, start=1):
                try:
                    # Load texture
                    img = load_image(tex_path, format='RGB')
                    # Quantize using palette
                    q_img = img.quantize(palette=palette_p_image, dither=Image.Dither.NONE)
                    q_rgb = q_img.convert('RGB')
                    # Build greyscale
                    idx = np.array(q_img)
                    max_idx = int(np.max(idx)) if idx.size else 0
                    scale = 255.0 / max(1, max_idx)
                    disp = (idx.astype(np.float32) * scale).astype(np.uint8)
                    gs_img = Image.fromarray(disp, 'L').convert('RGB')
                    # Save outputs
                    base_dir = self.output_dir or os.path.dirname(tex_path)
                    base_name, src_ext = os.path.splitext(os.path.basename(tex_path))
                    src_ext = src_ext.lower()
                    os.makedirs(base_dir, exist_ok=True)
                    if src_ext == '.dds':
                        tmp_quant = os.path.join(base_dir, f"{base_name}_quant_temp.png")
                        tmp_bw = os.path.join(base_dir, f"{base_name}_bw_temp.png")
                        out_quant_dds = os.path.join(base_dir, f"{base_name}_quant.dds")
                        out_bw_dds = os.path.join(base_dir, f"{base_name}_bw.dds")
                        try:
                            q_rgb.save(tmp_quant, format='PNG')
                            gs_img.save(tmp_bw, format='PNG')
                            convert_to_dds(tmp_quant, out_quant_dds)
                            convert_to_dds(tmp_bw, out_bw_dds)
                        finally:
                            for p in (tmp_quant, tmp_bw):
                                try:
                                    if os.path.exists(p):
                                        os.remove(p)
                                except Exception:
                                    pass
                    else:
                        q_rgb.save(os.path.join(base_dir, f"{base_name}_quant.png"), format='PNG')
                        gs_img.save(os.path.join(base_dir, f"{base_name}_bw.png"), format='PNG')
                    if first_bw is None:
                        first_bw = gs_img
                    if first_pal is None:
                        first_pal = q_rgb
                    processed += 1
                except Exception as e:
                    logger.warning(f"Failed processing {tex_path}: {e}")
                    failed += 1
                self.progress.emit(i, total)
                if i % 10 == 0 or i == total:
                    self.info.emit(f"Processed {i}/{total} (ok={processed}, failed={failed})")
            self.finished.emit({'processed': processed, 'failed': failed, 'skipped': skipped, 'first_bw': first_bw, 'first_pal': first_pal})
        except Exception as e:
            self.error.emit(str(e))


class ConvertToPaletteWidget(BaseWidget):
    """
    UI: "convert to pallete" (as requested)
    - Pick a palette texture (image)
    - Pick multiple texture files to process
    - Quantize each image using Image.quantize(palette=selected palette texture)
    - Convert quantized images to black & white indices according to the palette order (no resorting — assume palette is already ordered)
    - Export results to selected directory (or same directory when not set)
    """

    def __init__(self, parent: Optional[QWidget], text: str):
        super().__init__(text, parent, True)
        self.setObjectName('ConvertToPaletteWidget')

        self.palette_path: Optional[str] = None
        self.texture_paths: List[str] = []
        self.output_dir: Optional[str] = cfg.get(cfg.convert_output_dir_cfg)

        self.palette_image: Optional[Image.Image] = None
        self.palette_p_image: Optional[Image.Image] = None  # P-mode image to be used as quantize palette

        self.palette_card = PushSettingCard(
            self.tr("Select Palette Texture"),
            CustomIcons.PALETTE.icon(),
            self.tr("Select Palette Texture"),
            "",
        )
        self.palette_card.clicked.connect(self._on_pick_palette)

        self.textures_dir_card = PushSettingCard(
            self.tr("Choose Textures..."),
            CustomIcons.IMAGE.icon(stroke=True),
            self.tr("Choose Textures..."),
            cfg.get(cfg.textures),
        )
        self.textures_dir_card.clicked.connect(self._on_pick_textures)

        self.output_dir_card = PushSettingCard(
            self.tr("Output Folder (optional)"),
            FIF.SAVE.icon(),
            self.tr("Select output folder for PNGs (default: same as texture)"),
            cfg.get(cfg.convert_output_dir_cfg),
        )

        self.output_dir_card.clicked.connect(self._on_pick_output)

        self.addToFrame(self.palette_card)
        self.addToFrame(self.textures_dir_card)
        self.addToFrame(self.output_dir_card)

        # Results preview info
        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        self.addToFrame(self.lbl_info)

        # Previews
        self.preview_bw = QLabel(self.tr("Black And White Preview"))
        self.preview_bw.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_bw.setMinimumSize(450, 450)
        self.preview_bw.setStyleSheet("border: 1px dashed gray;")

        self.preview_pal = QLabel(self.tr("Applied Palette"))
        self.preview_pal.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_pal.setMinimumSize(450, 450)
        self.preview_pal.setStyleSheet("border: 1px dashed gray;")

        grid = QGridLayout()
        grid.addWidget(QLabel(self.tr("B&W")), 0, 0)
        grid.addWidget(QLabel(self.tr("Applied Palette")), 0, 1)
        grid.addWidget(self.preview_bw, 1, 0)
        grid.addWidget(self.preview_pal, 1, 1)
        container = QWidget()
        container.setLayout(grid)
        self.addToFrame(container)

        # Action buttons
        self.btn_start = PrimaryPushButton(icon=FIF.RIGHT_ARROW, text=self.tr("Run"))
        self.btn_start.clicked.connect(self._on_start)

        self.addButtonBarToBottom(self.btn_start)

        self.settings_widget = PaletteSettings(self)
        self.settings_drawer.addWidget(self.settings_widget)

        self.help_widget = PaletteConvertHelp(self)
        self.help_drawer.addWidget(self.help_widget)



    # -------------- Actions ---------------
    def _on_pick_palette(self):
        path, _ = QFileDialog.getOpenFileName(self, self.tr("Select Palette Texture"), cfg.get(cfg.base_palette_cfg),
                                              self.tr("Image Files (*.png *.jpg *.jpeg *.bmp *.tga *.dds)"))
        if not path:
            return
        try:
            # use loader to support DDS and other formats
            self.palette_image = load_image(path, format='RGB')
            cfg.set(cfg.base_palette_cfg, path)
            self.palette_card.setContent(path)
            self.palette_path = path
            # Build a P-mode palette using PIL adaptive palette (keeps original order reasonably while ensuring a valid palette)
            self.palette_p_image = self.palette_image.convert('P', palette=Image.Palette.ADAPTIVE)
        except Exception as e:
            logger.error(f"Failed to load palette: {e}")
            QMessageBox.critical(self, self.tr("Error"), self.tr(f"Failed to load palette: {e}"))

    def _on_pick_textures(self):
        paths, _ = QFileDialog.getOpenFileNames(self, self.tr("Select Texture Images"), cfg.get(cfg.base_palette_cfg),
                                                self.tr("Image Files (*.png *.jpg *.jpeg *.bmp *.tga *.dds)"))
        if not paths:
            return
        self.texture_paths = paths
        short = ", ".join(os.path.basename(p) for p in paths[:5])
        extra = "" if len(paths) <= 5 else f" +{len(paths) - 5} more"


        self.textures_dir_card.setContent(self.tr(f"{len(paths)} file(s): {short}{extra}"))

    def _on_pick_output(self):
        path = QFileDialog.getExistingDirectory(self, self.tr("Select Output Directory"), os.path.expanduser("~"))
        if not path:
            return
        self.output_dir = path

        self.output_dir_card.setContent(path)
        cfg.set(cfg.convert_output_dir_cfg, path)

    def _on_start(self):
        if not self.palette_image and not self.palette_path:
            QMessageBox.warning(self, self.tr("Warning"), self.tr("Please select a palette texture first."))
            return
        if not self.texture_paths:
            QMessageBox.warning(self, self.tr("Warning"), self.tr("Please select one or more textures to process."))
            return
        try:
            self.btn_start.setEnabled(False)
            p = getattr(self, 'parent', None)
            if p and hasattr(p, 'show_progress'):
                try:
                    p.show_progress()
                except Exception:
                    pass
            # Launch worker
            palette_path = self.palette_path or cfg.get(cfg.base_palette_cfg)
            self.worker = ConvertToPaletteWorker(palette_path, self.texture_paths, self.output_dir)
            self.worker.progress.connect(self._on_worker_progress)
            self.worker.info.connect(self._on_worker_info)
            self.worker.finished.connect(self._on_worker_finished)
            self.worker.error.connect(self._on_worker_error)
            self.worker.start()
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            QMessageBox.critical(self, self.tr("Error"), self.tr(f"Conversion failed: {e}"))
            self.btn_start.setEnabled(True)
            p = getattr(self, 'parent', None)
            if p and hasattr(p, 'complete_loader'):
                try:
                    p.complete_loader()
                except Exception:
                    pass

    def _on_worker_progress(self, i: int, total: int):
        try:
            p = getattr(self, 'parent', None)
            if p and hasattr(p, 'update_progress') and total:
                percent = int(max(0, min(100, round((i / total) * 100))))
                p.update_progress(percent)
        except Exception:
            pass

    def _on_worker_info(self, text: str):
        try:
            self.lbl_info.setText(self.tr(text))
        except Exception:
            pass

    def _on_worker_finished(self, data: object):
        try:
            self.btn_start.setEnabled(True)
            first_bw = data.get('first_bw') if isinstance(data, dict) else None
            first_pal = data.get('first_pal') if isinstance(data, dict) else None
            if first_bw is not None:
                self._display_on_label(first_bw, self.preview_bw)
            if first_pal is not None:
                self._display_on_label(first_pal, self.preview_pal)
            InfoBar.success(title=self.tr("Done"), content=self.tr("Conversion finished."), duration=3000, parent=self)
        finally:
            p = getattr(self, 'parent', None)
            if p and hasattr(p, 'complete_loader'):
                try:
                    p.complete_loader()
                except Exception:
                    pass

    def _on_worker_error(self, message: str):
        try:
            self.btn_start.setEnabled(True)
            QMessageBox.critical(self, self.tr("Error"), self.tr(f"Conversion failed: {message}"))
        finally:
            p = getattr(self, 'parent', None)
            if p and hasattr(p, 'complete_loader'):
                try:
                    p.complete_loader()
                except Exception:
                    pass

    # -------------- Core processing ---------------
    def _process_single(self, tex_path: str):
        # Load texture
        img = load_image(tex_path, format='RGB')
        # Quantize using the provided palette image per PIL docs
        pal_src = self._ensure_palette_image()
        q_img = img.quantize(palette=pal_src, dither=Image.Dither.NONE)
        q_rgb = q_img.convert('RGB')

        # Convert to greyscale by scaling palette indices from the quantized P image
        gs_img = self._map_to_palette_greyscale(q_img)

        # Save outputs (DDS if original was DDS)
        base_dir = self.output_dir or os.path.dirname(tex_path)
        base_name, src_ext = os.path.splitext(os.path.basename(tex_path))
        src_ext = src_ext.lower()
        os.makedirs(base_dir, exist_ok=True)

        source_is_dds = src_ext == '.dds'
        if source_is_dds:
            # write temp PNGs then convert
            tmp_quant = os.path.join(base_dir, f"{base_name}_quant_temp.png")
            tmp_bw = os.path.join(base_dir, f"{base_name}_bw_temp.png")
            out_quant_dds = os.path.join(base_dir, f"{base_name}_quant.dds")
            out_bw_dds = os.path.join(base_dir, f"{base_name}_bw.dds")
            try:
                q_rgb.save(tmp_quant, format='PNG')
                gs_img.save(tmp_bw, format='PNG')
                convert_to_dds(tmp_quant, out_quant_dds)
                convert_to_dds(tmp_bw, out_bw_dds)
            finally:
                # cleanup temp files
                for p in (tmp_quant, tmp_bw):
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except Exception as ce:
                        logger.warning(f"Failed to remove temp file {p}: {ce}")
        else:
            out_quant = os.path.join(base_dir, f"{base_name}_quant.png")
            out_bw = os.path.join(base_dir, f"{base_name}_bw.png")
            q_rgb.save(out_quant, format='PNG')
            gs_img.save(out_bw, format='PNG')

        return q_rgb, gs_img

    def _ensure_palette_image(self) -> Image.Image:
        """Ensure and return a P-mode palette image built per PIL docs using ADAPTIVE palette."""
        if self.palette_p_image is not None and self.palette_p_image.mode == 'P':
            return self.palette_p_image
        if self.palette_image is None:
            raise ValueError("Palette image is not loaded")
        # Per Pillow docs: build a palette image using ADAPTIVE
        self.palette_p_image = self.palette_image.convert('P', palette=Image.Palette.ADAPTIVE)
        return self.palette_p_image

    def _map_to_palette_greyscale(self, q_or_rgb_img: Image.Image) -> Image.Image:
        """Create a displayable greyscale image by scaling quantized palette indices to 0..255.
        If input is not mode 'P', it will be quantized using the selected palette first.
        """
        img = q_or_rgb_img
        if img.mode != 'P':
            pal = self._ensure_palette_image()
            img = img.convert('RGB').quantize(palette=pal, dither=Image.Dither.NONE)
        idx = np.array(img)
        max_idx = int(np.max(idx)) if idx.size else 0
        scale = 255.0 / max(1, max_idx)
        disp = (idx.astype(np.float32) * scale).astype(np.uint8)
        return Image.fromarray(disp, 'L').convert('RGB')

    # -------------- Helpers ---------------
    def _display_on_label(self, pil_image: Image.Image, label: QLabel, description: str = ""):
        if not pil_image:
            return
        # downscale to fit label
        max_w, max_h = 256, 128
        img = pil_image.copy()
        img.thumbnail((max_w, max_h))
        data = img.convert('RGBA').tobytes('raw', 'RGBA')
        qimg = QImage(data, img.width, img.height, QImage.Format.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg)
        label.setPixmap(pix)
        if description:
            label.setText(description)
