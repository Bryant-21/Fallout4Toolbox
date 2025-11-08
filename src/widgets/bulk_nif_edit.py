import os
import traceback
from pathlib import Path
from typing import Optional

from PIL import Image
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QLabel, QFileDialog, QMessageBox, QGridLayout
from qfluentwidgets import (
    PushSettingCard,
    PrimaryPushButton,
    InfoBar,
    FluentIcon as FIF, SwitchSettingCard, PushButton,
)

from help.unnif_help import NifUVHelp
from palette.palette_engine import load_image
from settings.basic_settings import BasicSettings
from src.utils.appconfig import cfg
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.imageutils import dilation_fill_static
from src.utils.logging_utils import logger
from src.utils.mipflooding import _apply_mip_flooding_to_png
from src.utils.nifutils import DDS_DIFFUSE_RE, remove_padding_from_texture_using_nif_uv
from utils.capabilities import CAPABILITIES
from src.utils.chainner_utils import run_chainner_directory
from src.utils.appconfig import cfg as _cfg


class UVPaddingWorker(QThread):
    progress = Signal(int, int)
    info = Signal(str)
    finished = Signal(int, int, int)
    error = Signal(str)

    def __init__(self, textures_dir: str, data_root: str, output_dir: Optional[str], do_mip_flooding: bool, do_dilation: bool, do_ai_upscale: bool):
        super().__init__()
        self.textures_dir = textures_dir
        self.data_root = data_root
        self.output_dir = output_dir
        self.do_mip_flooding = do_mip_flooding
        self.do_dilation = do_dilation
        self._stop = False
        self.do_ai_upscale = do_ai_upscale

    def abort(self):
        self._stop = True

    def run(self):
        try:
            tex_dir_path = Path(self.textures_dir)
            data_root_path = Path(self.data_root)
            files = [t for t in sorted(tex_dir_path.glob('*.dds')) if DDS_DIFFUSE_RE.search(t.name)]
            total = len(files)
            if total == 0:
                self.finished.emit(0, 0, 0)
                return
            out_dir_path: Optional[Path] = Path(self.output_dir) if self.output_dir else None
            if out_dir_path:
                try:
                    out_dir_path.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
            saved = 0
            skipped = 0
            failed = 0
            # Collect created files per target directory when AI upscale is enabled
            created_by_dir: dict[Path, list[str]] = {}
            for i, tex in enumerate(files, start=1):
                if self._stop:
                    break
                try:
                    result = remove_padding_from_texture_using_nif_uv(tex, data_root_path)
                    if result is None:
                        skipped += 1
                    else:
                        target_dir = out_dir_path or tex.parent
                        try:
                            target_dir.mkdir(parents=True, exist_ok=True)
                        except Exception:
                            pass
                        out_path = target_dir / f"{tex.stem}_nopad.png"
                        # Save base result
                        result.save(out_path, format='PNG')
                        if self.do_ai_upscale and CAPABILITIES.get("ChaiNNer", False):
                            # Defer post-processing; collect file name for batch upscaling
                            created_by_dir.setdefault(target_dir, []).append(out_path.name)
                        else:
                            # Optional post-processing immediately when not doing AI upscale
                            if self.do_mip_flooding:
                                try:
                                    _ = _apply_mip_flooding_to_png(out_path, result)
                                except Exception:
                                    pass
                            elif self.do_dilation:
                                try:
                                    dilation_fill_static(out_path, result)
                                except Exception:
                                    pass
                        saved += 1
                except Exception as e:
                    logger.warning(f"Failed processing {tex}: {e}")
                    failed += 1
                self.progress.emit(i, total)
                if i % 10 == 0 or i == total:
                    self.info.emit(f"Processed {i}/{total} (saved={saved}, skipped={skipped}, failed={failed})")
            # If AI upscale requested, run ChaiNNer on the created files using a name list glob per directory
            if self.do_ai_upscale and CAPABILITIES.get("ChaiNNer", False) and created_by_dir:
                textures_model = _cfg.get(_cfg.upscale_textures_cfg)
                for folder, names in created_by_dir.items():
                    try:
                        if not names:
                            continue
                        # Build a WCMatch braces glob of exact filenames
                        # Example: "{a_nopad.png,b_nopad.png}"
                        brace = "{" + ",".join(names) + "}"
                        ok = run_chainner_directory(str(folder), textures_model, str(folder), brace)
                        # After upscaling, run optional post-processing on each upscaled output
                        for base in names:
                            stem = Path(base).stem
                            upscaled = folder / f"{stem}_upscaled.png"
                            if upscaled.exists():
                                try:
                                    pil = Image.open(upscaled).convert('RGBA')
                                except Exception:
                                    pil = None
                                if self.do_mip_flooding and pil is not None:
                                    try:
                                        _ = _apply_mip_flooding_to_png(upscaled, pil)
                                    except Exception:
                                        pass
                                elif self.do_dilation and pil is not None:
                                    try:
                                        dilation_fill_static(upscaled, pil)
                                    except Exception:
                                        pass
                            else:
                                failed += 1
                    except Exception as e:
                        logger.warning(f"AI Upscale batch failed in {folder}: {e}")
                        # conservatively count as failures
                        failed += len(names)
            self.finished.emit(saved, skipped, failed)
        except Exception as e:
            self.error.emit(str(e))


class UVPaddingRemoverWidget(BaseWidget):
    """UI to remove padding from textures using NIF UVs.

    - User selects a Textures folder under Data\Textures.
    - Optionally selects an output folder (defaults to same as texture file).
    - Shows dual preview (original vs. masked) for the selected file.
    - Can batch process all *_d.dds in the folder.
    """

    def __init__(self, parent: Optional[QWidget] = None, text: str = "UV Padding Remover"):
        super().__init__(text=text, parent=parent, vertical=True)

        # Cards
        self.data_root_card = PushSettingCard(
            self.tr("Data Root (contains Textures/Meshes/Materials)"),
            FIF.FOLDER.icon(),
            self.tr("Select Data Folder"),
            content=cfg.get(cfg.data_root_cfg),
        )
        self.data_root_card.clicked.connect(self._on_pick_data_root)

        self.textures_dir_card = PushSettingCard(
            self.tr("Textures Folder"),
            CustomIcons.IMAGE.icon(stroke=True),
            self.tr("Select a folder under Data/Textures"),
            content=cfg.get(cfg.textures_dir_cfg),
        )
        self.textures_dir_card.clicked.connect(self._on_pick_textures_dir)

        self.output_dir_card = PushSettingCard(
            self.tr("Output Folder (optional)"),
            FIF.SAVE.icon(),
            self.tr("Select output folder for PNGs (default: same as texture)"),
            cfg.get(cfg.output_dir_cfg),
        )
        self.output_dir_card.clicked.connect(self._on_pick_output_dir)

        self.addToFrame(self.data_root_card)
        self.addToFrame(self.textures_dir_card)
        self.addToFrame(self.output_dir_card)

        if CAPABILITIES["mip_flooding"]:
            self.chk_mip_flooding_card = SwitchSettingCard(icon=CustomIcons.FLOOD.icon(),
                                                     title=self.tr("Mip Flooding"),
                                                   content=self.tr("Run mip flooding on the output to reduce edge artifacts."),
                                                   configItem=cfg.mip_flooding)



        self.chk_color_fill = SwitchSettingCard(icon=CustomIcons.INFINITY.icon(),
                                                 title=self.tr("Infinite Dilation"),
                                                 content = "Apply infinite dilation to the new transparent areas to fill holes.",
                                                 configItem=cfg.color_fill)

        # Option toggles
        if CAPABILITIES["ChaiNNer"]:
            self.chk_ai_upscale = SwitchSettingCard(icon=CustomIcons.ENHANCE.icon(),
                                                    title=self.tr("AI Upscale"),
                                                    content=self.tr("Run AI upscaler after cutting, before mip flooding/dilation."),
                                                    configItem=cfg.do_ai_upscale)
            self.chk_ai_upscale.switchButton.setChecked(False)
            self.addToFrame(self.chk_ai_upscale)

        if CAPABILITIES["mip_flooding"]:
            self.addToFrame(self.chk_mip_flooding_card)

        self.addToFrame(self.chk_color_fill)

        # Previews
        self.original_label = QLabel(self.tr("Original"))
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setMinimumSize(450, 450)
        self.original_label.setStyleSheet("border: 1px dashed gray;")

        self.masked_label = QLabel(self.tr("No-Padding Preview"))
        self.masked_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.masked_label.setMinimumSize(450, 450)
        self.masked_label.setStyleSheet("border: 1px dashed gray;")

        grid = QGridLayout()
        grid.addWidget(QLabel(self.tr("Original")), 0, 0)
        grid.addWidget(QLabel(self.tr("No-Padding")), 0, 1)
        grid.addWidget(self.original_label, 1, 0)
        grid.addWidget(self.masked_label, 1, 1)
        container = QWidget()
        container.setLayout(grid)
        self.boxLayout.addStretch(1)
        self.addToFrame(container)

        # Buttons
        self.btn_pick_file = PushButton(icon=FIF.ZOOM_IN, text=self.tr("Preview One Texture"))
        self.btn_pick_file.clicked.connect(self._on_preview_one)
        self.btn_process = PrimaryPushButton(icon=FIF.RIGHT_ARROW, text=self.tr("Process Folder"))
        self.btn_process.clicked.connect(self._on_process_folder)
        self.buttons_layout.addWidget(self.btn_pick_file, stretch=1)
        self.addButtonBarToBottom(self.btn_process)

        # Current state for preview
        self.current_texture_path: Optional[str] = None
        self.current_original: Optional[Image.Image] = None
        self.current_result: Optional[Image.Image] = None
        self.current_result: Optional[Image.Image] = None

        # internal state
        self._mip_warned = False

        self.settings_widget = BasicSettings(self)
        self.settings_drawer.addWidget(self.settings_widget)

        self.help_widget = NifUVHelp(self)
        self.help_drawer.addWidget(self.help_widget)

    # ---------------- UI handlers ----------------
    def _on_pick_data_root(self):
        base = cfg.get(cfg.data_root_cfg) or os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(self, self.tr("Select Data Folder"), base)
        if path:
            cfg.set(cfg.data_root_cfg, path)
            self.data_root_card.setContent(path)

    def _on_pick_textures_dir(self):
        base = cfg.get(cfg.textures_dir_cfg) or cfg.get(cfg.data_root_cfg) or os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(self, self.tr("Select Textures Folder"), base)
        if path:
            cfg.set(cfg.textures_dir_cfg,path)
            self.textures_dir_card.setContent(path)
            # If data root not set, try to infer
            if not cfg.get(cfg.data_root_cfg):
                dr = self._infer_data_root_from_textures(path)
                if dr:
                    cfg.set(cfg.data_root_cfg,dr)
                    self.data_root_card.setContent(dr)

    def _on_pick_output_dir(self):
        base = cfg.output_dir_cfg.value or cfg.textures_dir_cfg.value or os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(self, self.tr("Select Output Folder"), base)
        if path:
            cfg.set(cfg.output_dir_cfg, path)
            self.output_dir_card.setContent(path)

    def _on_preview_one(self):
        textures_dir = cfg.textures_dir_cfg.value
        data_root = cfg.data_root_cfg.value
        if not textures_dir or not data_root:
            QMessageBox.warning(self, self.tr("Warning"), self.tr("Please set Data root and a Textures folder."))
            return
        # pick a texture file
        file_path, _ = QFileDialog.getOpenFileName(self, self.tr("Pick a texture image"), textures_dir, self.tr("Image files (*.png *.jpg *.jpeg *.bmp *.tga *.dds)"))
        if not file_path:
            return
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'show_progress'):
            try:
                p.show_progress()
            except Exception:
                pass
        try:
            orig = load_image(file_path, 'RGBA')
            self.current_original = orig
            self.current_texture_path = file_path
            self._display_on_label(orig, self.original_label)

            result = remove_padding_from_texture_using_nif_uv(Path(file_path), Path(data_root))
            if result is None:
                InfoBar.info(title=self.tr("No match"), content=self.tr("Could not build UV mask for this texture."), duration=3000, parent=self)
                self.masked_label.setText(self.tr("No mask produced"))
                self.current_result = None
                return
            self.current_result = result
            self._display_on_label(result, self.masked_label)
            InfoBar.success(title=self.tr("Preview ready"), content=self.tr("UV mask applied"), duration=2000, parent=self)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Preview failed: {e}")
            QMessageBox.critical(self, self.tr("Error"), self.tr(f"Preview failed: {e}"))
        finally:
            p = getattr(self, 'parent', None)
            if p and hasattr(p, 'complete_loader'):
                try:
                    p.complete_loader()
                except Exception:
                    pass

    def _on_process_folder(self):
        textures_dir = cfg.textures_dir_cfg.value
        data_root = cfg.data_root_cfg.value
        out_dir = cfg.output_dir_cfg.value or None
        if not textures_dir or not data_root:
            QMessageBox.warning(self, self.tr("Warning"), self.tr("Please set Data root and a Textures folder."))
            return
        # Disable UI and show progress mask
        try:
            self.btn_process.setEnabled(False)
            self.btn_pick_file.setEnabled(False)
        except Exception:
            pass
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'show_progress'):
            try:
                p.show_progress()
            except Exception:
                pass
        # Start background worker
        do_mip = bool(cfg.get(cfg.mip_flooding)) and CAPABILITIES["mip_flooding"]
        do_dilate = bool(cfg.get(cfg.color_fill))
        do_ai = bool(getattr(self, 'chk_ai_upscale', None) and self.chk_ai_upscale.switchButton.isChecked())
        self.worker = UVPaddingWorker(textures_dir, data_root, out_dir, do_mip, do_dilate, do_ai)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.info.connect(self._on_worker_info)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

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
            # Use InfoBar for periodic status updates
            InfoBar.info(self.tr("Working"), self.tr(text), duration=1500, parent=self)
        except Exception:
            pass

    def _on_worker_finished(self, saved: int, skipped: int, failed: int):
        try:
            InfoBar.success(title=self.tr("Done"), content=self.tr(f"Finished. saved={saved}, skipped={skipped}, failed={failed}"), duration=5000, parent=self)
        finally:
            try:
                self.btn_process.setEnabled(True)
                self.btn_pick_file.setEnabled(True)
            except Exception:
                pass
            p = getattr(self, 'parent', None)
            if p and hasattr(p, 'complete_loader'):
                try:
                    p.complete_loader()
                except Exception:
                    pass

    def _on_worker_error(self, message: str):
        try:
            InfoBar.error(title=self.tr("Error"), content=self.tr(message), duration=5000, parent=self)
        finally:
            try:
                self.btn_process.setEnabled(True)
                self.btn_pick_file.setEnabled(True)
            except Exception:
                pass
            p = getattr(self, 'parent', None)
            if p and hasattr(p, 'complete_loader'):
                try:
                    p.complete_loader()
                except Exception:
                    pass

    # ---------------- helpers ----------------
    def _display_on_label(self, image: Image.Image, label: QLabel):
        # convert PIL to QImage/QPixmap
        rgba = image.convert('RGBA')
        w, h = rgba.size
        data = rgba.tobytes('raw', 'RGBA')
        qimg = QImage(data, w, h, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    @staticmethod
    def _infer_data_root_from_textures(textures_dir: str) -> Optional[str]:
        cur = Path(textures_dir)
        while cur and cur.name.lower() != 'data' and cur.parent != cur:
            cur = cur.parent
        if cur and cur.name.lower() == 'data':
            return str(cur)
        return None
