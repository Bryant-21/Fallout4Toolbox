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

# Reuse the worker and image/palette helpers from existing widget and palette engine
from src.widgets.add_colors_to_palette import AddColorsWorker  # shared heavy-lifting worker
from src.palette.palette_engine import next_power_of_2, extract_existing_palette_rows
from src.settings.palette_settings import PaletteSettings
from src.utils.appconfig import cfg
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger


class AddColorsToSinglePaletteWidget(BaseWidget):
    """
    UI: "Add Colors To Single Pallete"

    Differences from `AddColorsToPaletteWidget`:
    - User selects ONLY the palette texture.
    - We automatically discover a matching greyscale image by filename.
    - We recursively scan all subdirectories (from the palette's folder) for matching color textures.
    - Then we process and append rows to the palette using the shared `AddColorsWorker`.
    """

    IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".dds")

    def __init__(self, parent: Optional[QWidget], text: str):
        super().__init__(text, parent, True)
        self.setObjectName('AddColorsToSinglePaletteWidget')

        # inputs
        self.palette_path: Optional[str] = cfg.get(cfg.base_palette_cfg)
        self.greyscale_path: Optional[str] = None
        self.texture_paths: List[str] = []
        self.output_dir: Optional[str] = cfg.get(cfg.convert_output_dir_cfg)

        # data
        self.existing_palette_rows: Optional[List[np.ndarray]] = None
        self.palette_size: int = 256
        self.generated_palette_image: Optional[Image.Image] = None
        self._save_after_build: bool = False
        self._worker_started: bool = False

        # pickers / info cards
        self.palette_card = PushSettingCard(
            self.tr("Select Palette Texture"),
            CustomIcons.PALETTE.icon(),
            self.tr("Select Palette Texture"),
            self.palette_path or "",
        )
        self.palette_card.clicked.connect(self._on_pick_palette)

        self.greyscale_info_card = PushSettingCard(
            self.tr("Auto‑detected Greyscale"),
            CustomIcons.GREYSCALE.icon(),
            self.tr("Greyscale image matched by filename. Change palette to re‑detect."),
            self.greyscale_path or self.tr("<none>"),
        )
        # info only; not clickable
        self.greyscale_info_card.setEnabled(False)

        self.textures_info_card = PushSettingCard(
            self.tr("Detected Color Texture(s)"),
            CustomIcons.IMAGEADD.icon(stroke=True),
            self.tr("All matching textures found by recursive scan from the palette's folder."),
            ", ".join(self.texture_paths) if self.texture_paths else self.tr("<none>"),
        )
        self.textures_info_card.setEnabled(False)

        self.output_dir_card = PushSettingCard(
            self.tr("Output Folder (optional)"),
            FIF.SAVE.icon(),
            self.tr("Select output folder for generated palette (default: same folder as palette)"),
            self.output_dir or "",
        )
        self.output_dir_card.clicked.connect(self._on_pick_output)

        self.replace_existing = SwitchSettingCard(icon=CustomIcons.REPLACE.icon(),
                                                  title=self.tr("Replace Existing Palette"),
                                                  content=self.tr("Otherwise write to output folder."),
                                                  configItem=cfg.ci_replace_existing)

        self.addToFrame(self.palette_card)
        self.addToFrame(self.greyscale_info_card)
        self.addToFrame(self.textures_info_card)
        self.addToFrame(self.output_dir_card)
        self.addToFrame(self.replace_existing)

        # previews
        self.preview_bw = QLabel(self.tr("Greyscale Preview"))
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
        self.btn_rescan = PushButton(text=self.tr("Rescan"))
        self.btn_rescan.clicked.connect(self._rescan_based_on_palette)
        self.btn_run = PrimaryPushButton(icon=FIF.RIGHT_ARROW, text=self.tr("Run"))
        self.btn_run.clicked.connect(self._on_build_and_save)
        self.buttons_layout.addWidget(self.btn_rescan)
        self.addButtonBarToBottom(self.btn_run)

        # settings and help
        self.settings_widget = PaletteSettings(self)
        self.settings_drawer.addWidget(self.settings_widget)

        # if palette is preconfigured, attempt auto-detection on load
        try:
            if self.palette_path:
                self._after_palette_selected()
        except Exception:
            pass


    def _on_pick_output(self):
        dir_ = QFileDialog.getExistingDirectory(self, self.tr("Select Output Folder"), self.output_dir or "")
        if dir_:
            self.output_dir = dir_
            cfg.set(cfg.convert_output_dir_cfg, dir_)
            self.output_dir_card.setContent(dir_)

    # region core flow
    def _on_pick_palette(self):
        file, _ = QFileDialog.getOpenFileName(self, self.tr("Select Palette Texture"),
                                              self.palette_path or "",
                                              "Images (*.png *.jpg *.jpeg *.dds)")
        if not file:
            return
        self.palette_path = file
        cfg.set(cfg.base_palette_cfg, file)
        self.palette_card.setContent(file)
        try:
            self.existing_palette_rows, self.palette_size = extract_existing_palette_rows(file, int(cfg.get(cfg.ci_palette_row_height)) or 1)
        except Exception as e:
            logger.error(f"Failed to parse palette: {e}")
            QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to parse palette texture."))
            return
        self._after_palette_selected()

    def _after_palette_selected(self):
        # Discover greyscale, then color textures
        try:
            self._auto_detect_greyscale_and_textures()
        except Exception as e:
            logger.error(f"Auto-detect failed: {e}")
        # Update previews/info
        try:
            if self.greyscale_path and os.path.exists(self.greyscale_path):
                img = Image.open(self.greyscale_path).convert('RGB')
                self._display_on_label(img, self.preview_bw)
        except Exception as e:
            logger.warning(f"Failed to preview greyscale: {e}")

    def _rescan_based_on_palette(self):
        self._auto_detect_greyscale_and_textures(show_feedback=True)

    def _auto_detect_greyscale_and_textures(self, show_feedback: bool = False):
        if not self.palette_path:
            if show_feedback:
                QMessageBox.information(self, self.tr("Missing"), self.tr("Please select a palette texture first."))
            return
        base_dir = os.path.dirname(self.palette_path)
        key = self._normalize_base_name(self.palette_path)

        # 1) Find greyscale: prefer same-directory candidates with greyscale-like suffix
        candidates: List[str] = []
        for root, _dirs, files in os.walk(base_dir):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in self.IMAGE_EXTS:
                    continue
                full = os.path.join(root, fn)
                # Don't consider the palette image itself as a greyscale candidate
                if os.path.abspath(full) == os.path.abspath(self.palette_path or ""):
                    continue
                norm = self._normalize_base_name(full)
                if norm != key:
                    continue
                if self._looks_like_greyscale_name(fn):
                    candidates.append(full)
        chosen_grey = None
        if candidates:
            # prefer same dir; else shortest path; then alpha sort
            same_dir = [p for p in candidates if os.path.dirname(p) == base_dir]
            group = same_dir if same_dir else candidates
            group.sort(key=lambda p: (len(p), p.lower()))
            chosen_grey = group[0]
        self.greyscale_path = chosen_grey
        self.greyscale_info_card.setContent(chosen_grey or self.tr("<none>"))

        # 2) Find color textures: all matches that are NOT greyscale-like, only from subdirectories (exclude base_dir)
        textures: List[str] = []
        for root, _dirs, files in os.walk(base_dir):
            if os.path.abspath(root) == os.path.abspath(base_dir):
                # Skip files in the same directory as the palette; only consider subdirectories
                continue
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in self.IMAGE_EXTS:
                    continue
                full = os.path.join(root, fn)
                # Skip the palette image itself
                if os.path.abspath(full) == os.path.abspath(self.palette_path or ""):
                    continue
                norm = self._normalize_base_name(full)
                if norm != key:
                    continue
                # Skip greyscale-like and palette-like files when collecting color textures
                if self._looks_like_greyscale_name(fn) or self._looks_like_palette_name(fn):
                    continue
                textures.append(full)
        textures = self._sort_texture_paths(textures)
        self.texture_paths = textures
        self.textures_info_card.setContent(", ".join(textures) if textures else self.tr("<none>"))

        if show_feedback:
            if not self.greyscale_path:
                QMessageBox.warning(self, self.tr("Greyscale not found"),
                                    self.tr("No matching greyscale image was found for this palette."))
            elif not self.texture_paths:
                QMessageBox.information(self, self.tr("No textures"),
                                        self.tr("No matching color textures were found in subfolders."))

    def _on_build(self):
        if not self.palette_path or not self.existing_palette_rows:
            QMessageBox.warning(self, self.tr("Missing"), self.tr("Please select a palette texture first."))
            return
        if not self.greyscale_path:
            QMessageBox.warning(self, self.tr("Missing"), self.tr("No matching greyscale image detected. Try Rescan or choose a different palette."))
            return
        if not self.texture_paths:
            QMessageBox.warning(self, self.tr("Missing"), self.tr("No matching color textures detected in subdirectories."))
            return

        # Pairs: one greyscale to many colors
        pairs: List[Tuple[str, str]] = [(self.greyscale_path, c) for c in self.texture_paths]

        # Show parent mask progress
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'show_progress'):
            try:
                p.show_progress()
            except Exception:
                pass
        try:
            self.btn_run.setEnabled(False)
            row_height = int(cfg.get(cfg.ci_palette_row_height))
            self._worker = AddColorsWorker(
                existing_palette_rows=self.existing_palette_rows or [],
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

    def _on_build_and_save(self):
        self._save_after_build = True
        self._worker_started = False
        self._on_build()
        if not self._worker_started:
            self._save_after_build = False
            if self.generated_palette_image is not None:
                self._on_save()

    # region worker callbacks
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
                if self._save_after_build:
                    try:
                        self._on_save()
                    finally:
                        self._save_after_build = False
        finally:
            self.btn_run.setEnabled(True)
            p = getattr(self, 'parent', None)
            if p and hasattr(p, 'complete_loader'):
                try:
                    p.complete_loader()
                except Exception:
                    pass

    def _on_worker_error(self, message: str):
        try:
            self._save_after_build = False
            QMessageBox.critical(self, self.tr("Error"), self.tr(f"Failed to build palette rows: {message}"))
        finally:
            self.btn_run.setEnabled(True)
            p = getattr(self, 'parent', None)
            if p and hasattr(p, 'complete_loader'):
                try:
                    p.complete_loader()
                except Exception:
                    pass
    # endregion

    # region save
    def _on_save(self):
        if self.generated_palette_image is None:
            QMessageBox.information(self, self.tr("Nothing to save"), self.tr("Please build the palette first."))
            return
        replace_existing = bool(cfg.get(cfg.ci_replace_existing))
        palette_path = self.palette_path or ""
        base_name = os.path.splitext(os.path.basename(palette_path or 'palette.png'))[0]
        source_is_dds = bool(palette_path and palette_path.lower().endswith('.dds'))

        if replace_existing and palette_path:
            out_path = palette_path
            base_dir = os.path.dirname(palette_path)
            try:
                os.makedirs(base_dir, exist_ok=True)
                if source_is_dds:
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
                self._reload_palette(out_path)
            except Exception as e:
                logger.error(f"Failed to replace existing palette: {e}")
                QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to save palette image."))
            return

        base_dir = self.output_dir or (os.path.dirname(palette_path) if palette_path else ".")
        output_extension = '.dds' if source_is_dds else '.png'
        out_path = os.path.join(base_dir, f"{base_name}_with_added_rows{output_extension}")
        try:
            os.makedirs(base_dir, exist_ok=True)
            if source_is_dds:
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
    # endregion

    # region helpers (shared pattern copied for this widget)
    def _reload_palette(self, path: str):
        try:
            self.palette_path = path
            cfg.set(cfg.base_palette_cfg, path)
            self.palette_card.setContent(path)
            self.existing_palette_rows, self.palette_size = extract_existing_palette_rows(path, int(cfg.get(cfg.ci_palette_row_height)) or 1)
            logger.info(f"Reloaded base palette from: {path}")
        except Exception as e:
            logger.error(f"Failed to reload palette from {path}: {e}")
            QMessageBox.warning(self, self.tr("Warning"), self.tr("Palette was saved, but failed to reload updated palette for preview/next run."))

    def _normalize_base_name(self, path: str) -> str:
        name = os.path.splitext(os.path.basename(path))[0]
        base = name.lower()
        # Strip common trailing qualifiers (including 'palette') to form the match key
        base = re.sub(r'(?:[\s_\-]?(?:greyscale|grayscale|grey|gray|bw|mask|palette))+$', '', base, flags=re.IGNORECASE)
        # Preserve underscores in keys so filenames with underscores still match
        base = re.sub(r"[^a-z0-9_]+", "", base)
        return base

    def _looks_like_greyscale_name(self, filename: str) -> bool:
        fname = os.path.splitext(os.path.basename(filename))[0].lower()
        return re.search(r'(greyscale|grayscale|grey|gray|bw|mask)$', fname) is not None

    def _looks_like_palette_name(self, filename: str) -> bool:
        """Heuristic: file base name ends with 'palette' (common palette textures)."""
        fname = os.path.splitext(os.path.basename(filename))[0].lower()
        return re.search(r'(?:^|[\s_\-])palette$', fname) is not None

    def _sort_texture_paths(self, paths: List[str]) -> List[str]:
        try:
            def _key(p: str):
                base_name = os.path.splitext(os.path.basename(p))[0].lower()
                parent_dir = os.path.basename(os.path.dirname(p)).lower()
                return (base_name, parent_dir, p.lower())
            return sorted(paths or [], key=_key)
        except Exception as e:
            logger.warning(f"Failed to sort texture paths: {e}")
            return list(paths or [])

    def _extract_existing_palette_rows(self, path: str) -> tuple[List[np.ndarray], int]:
        pil = load_image(path, format='RGB')
        arr = np.array(pil)
        h, w = arr.shape[:2]
        palette_size = w
        try:
            row_height = int(cfg.get(cfg.ci_palette_row_height))
            if row_height <= 0:
                row_height = 1
        except Exception:
            row_height = 1
        num_blocks = max(1, h // row_height)

        rows: List[np.ndarray] = []
        for block in range(num_blocks):
            start_row = block * row_height
            y = min(start_row, h - 1)
            row = arr[y, :w, :]
            eq = (row[:, 0] == row[:, 1]) & (row[:, 1] == row[:, 2])
            frac_grey = float(np.mean(eq))
            if frac_grey >= 0.9:
                continue
            rows.append(row[:palette_size].astype(np.uint8))

        if not rows:
            # Fallback: choose middle row
            y = h // 2
            row = arr[y, :w, :]
            rows = [row[:palette_size].astype(np.uint8)]
        return rows, palette_size

    def _display_on_label(self, pil_image: Image.Image, label: QLabel):
        qimage = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, pil_image.width * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
    # endregion
