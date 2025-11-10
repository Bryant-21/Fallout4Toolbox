import os
import re
import traceback
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

from src.palette.palette_engine import next_power_of_2, build_row_from_pair, compose_palette_image
from src.settings.palette_settings import PaletteSettings
from src.utils.appconfig import cfg
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger


class AddColorsWorker(QThread):
    progress = Signal(int, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, existing_palette_rows: List[np.ndarray], palette_size: int, pairs: List[Tuple[str, str]], row_height: int):
        super().__init__()
        self.existing_palette_rows = existing_palette_rows  # list of rows (palette_size, 3)
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
            # Fallback base row is the first existing row if available; else zeros
            base_row = self.existing_palette_rows[0] if (self.existing_palette_rows and len(self.existing_palette_rows) > 0) else np.zeros((self.palette_size, 3), dtype=np.uint8)

            new_rows: List[np.ndarray] = []

            quant_method = cfg.get(cfg.ci_default_quant_method)

            for i, (greyscale_path, color_path) in enumerate(self.pairs, start=1):
                try:
                    row_palette = build_row_from_pair(
                        greyscale_path=greyscale_path,
                        color_path=color_path,
                        base_row=base_row,
                        palette_size=self.palette_size,
                        quant_method=quant_method,
                        log_top_k=3,
                    )
                    new_rows.append(row_palette)
                finally:
                    self.progress.emit(i, total)

            # compose palette image: keep all existing rows and append all new rows in order
            all_blocks = (self.existing_palette_rows or []) + new_rows
            palette_img = compose_palette_image(
                rows=all_blocks,
                row_height=self.row_height,
                palette_size=self.palette_size,
                pad_mode='none'
            )
            self.finished.emit(palette_img)
            return
        except Exception as e:
            traceback.print_exc()
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
        self.existing_palette_rows: Optional[List[np.ndarray]] = None  # list of palette rows, each shape (palette_size, 3)
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
            self.tr("Add Source Texture(s) ..."),
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
                self.existing_palette_rows, self.palette_size = self._extract_existing_palette_rows(file)
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
            # Append to existing selection with case-insensitive de-duplication, preserving prior order
            existing = list(self.texture_paths or [])
            seen = {p.lower() for p in existing}
            for p in files:
                key = p.lower()
                if key not in seen:
                    existing.append(p)
                    seen.add(key)
            self.texture_paths = existing
            self.textures_card.setContent(", ".join(self.texture_paths))

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
        if not self.palette_path or not self.existing_palette_rows:
            QMessageBox.warning(self, self.tr("Missing"), self.tr("Please select a palette texture first."))
            return
        if not self.greyscale_paths:
            QMessageBox.warning(self, self.tr("Missing"), self.tr("Please select the black & white (index map) texture(s)."))
            return
        if not self.texture_paths:
            QMessageBox.warning(self, self.tr("Missing"), self.tr("Please select the color texture(s) to sample from."))
            return

        # Ensure texture paths are deterministically sorted before processing
        try:
            self.texture_paths = self._sort_texture_paths(self.texture_paths)
            # Refresh UI content to reflect sorted order
            self.textures_card.setContent(", ".join(self.texture_paths))
        except Exception as _sort_ex:
            logger.warning(f"Failed to sort texture paths before processing: {_sort_ex}")

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

        # Inform about any greyscales that didn't find a matching color texture
        try:
            unmatched = getattr(self, '_last_unmatched_greys', [])
            if unmatched:
                QMessageBox.information(self, self.tr("Unmatched greyscale(s)"), self.tr("These greyscale images had no matching color textures and were skipped: ") + ", ".join(unmatched))
        except Exception:
            pass

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
                # Reload the palette after saving in case it changed
                self._reload_palette(out_path)
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
    def _reload_palette(self, path: str):
        """Reload the current palette from the given path and refresh internal state/UI."""
        try:
            self.palette_path = path
            cfg.set(cfg.base_palette_cfg, path)
            self.palette_card.setContent(path)
            self.existing_palette_rows, self.palette_size = self._extract_existing_palette_rows(path)
            logger.info(f"Reloaded base palette from: {path}")
        except Exception as e:
            logger.error(f"Failed to reload palette from {path}: {e}")
            QMessageBox.warning(self, self.tr("Warning"), self.tr("Palette was saved, but failed to reload updated palette for preview/next run."))

    def _normalize_base_name(self, path: str) -> str:
        """Normalize filename to base key used for pairing.
        Removes suffixes like _greyscale/_grayscale/_grey/_gray/_bw/_mask at the end.
        Case-insensitive. Returns lowercase base name without extension.
        Keeps underscores to support names that rely on them for matching.
        """
        name = os.path.splitext(os.path.basename(path))[0]
        base = name.lower()
        # strip common greyscale suffix patterns possibly repeated
        base = re.sub(r'(?:[\s_\-]?(?:greyscale|grayscale|grey|gray|bw|mask))+$', '', base, flags=re.IGNORECASE)
        # remove all non-alphanumeric EXCEPT underscore to preserve keys with underscores
        base = re.sub(r"[^a-z0-9_]+", "", base)
        return base

    def _sort_texture_paths(self, paths: List[str]) -> List[str]:
        """Return a new list of texture paths sorted deterministically using the same
        logic as pairing: base file name, then parent directory, then full path (all case-insensitive).
        """
        try:
            def _key(p: str):
                base_name = os.path.splitext(os.path.basename(p))[0].lower()
                parent_dir = os.path.basename(os.path.dirname(p)).lower()
                return (base_name, parent_dir, p.lower())
            return sorted(paths or [], key=_key)
        except Exception as e:
            logger.warning(f"Failed to sort texture paths: {e}")
            return list(paths or [])

    def _pair_greyscale_and_textures(self, greys: List[str], colors: List[str]) -> List[Tuple[str, str]]:
        """Build a flat list of (greyscale, color) pairs for 1:many processing.
        For each greyscale G, find all color textures whose normalized base equals G's normalized base,
        process them in deterministic order, and append one output row per (G, C) pair.
        - Greyscales are processed in deterministic order (by normalized base, then filename parent dir, then full path).
        - Within each G group, colors are processed in deterministic order (file base → parent dir → full path).
        - If some greyscales have no matching colors, they are skipped; an informational message can be shown by caller.
        - If no pairs are formed at all, a ValueError is raised.
        """
        # Pre-sort greys and colors deterministically
        def _g_key(p: str):
            norm = self._normalize_base_name(p)
            parent_dir = os.path.basename(os.path.dirname(p)).lower()
            return (norm, parent_dir, p.lower())

        def _c_key(p: str):
            base_name = os.path.splitext(os.path.basename(p))[0].lower()
            parent_dir = os.path.basename(os.path.dirname(p)).lower()
            return (base_name, parent_dir, p.lower())

        greys_sorted = sorted(greys or [], key=_g_key)
        colors_sorted = sorted(colors or [], key=_c_key)

        # Build mapping from normalized base -> list of colors (in sorted order)
        colors_by_key = {}
        for c in colors_sorted:
            k = self._normalize_base_name(c)
            colors_by_key.setdefault(k, []).append(c)

        pairs: List[Tuple[str, str]] = []
        unmatched: List[str] = []
        for g in greys_sorted:
            k = self._normalize_base_name(g)
            matches = colors_by_key.get(k, [])
            if not matches:
                unmatched.append(os.path.basename(g))
                continue
            for c in matches:
                pairs.append((g, c))

        # expose unmatched for UI feedback
        try:
            self._last_unmatched_greys = unmatched
        except Exception:
            pass

        if not pairs:
            raise ValueError(self.tr("No matching color textures were found for the selected greyscale(s)."))
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

    def _extract_existing_palette_rows(self, path: str) -> tuple[List[np.ndarray], int]:
        """Parse ALL existing palette rows from an existing palette texture and return them in order.
        We identify logical palette rows by chunking the image height by configured row_height and
        selecting a representative scanline per chunk. Rows that look like greyscale filler are omitted.
        """
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
            # choose the first line of the block as representative
            y = min(start_row, h - 1)
            row = arr[y, :w, :]
            eq = (row[:, 0] == row[:, 1]) & (row[:, 1] == row[:, 2])
            frac_grey = float(np.mean(eq))
            # Treat rows that are predominantly greyscale as filler and skip them
            if frac_grey >= 0.9:
                continue
            rows.append(row[:palette_size].astype(np.uint8))

        # Fallback: if nothing detected (e.g., single-row palette), pick a best-guess row using legacy logic
        if not rows:
            base_row, _ = self._extract_base_palette(path)
            rows = [base_row]

        return rows, palette_size

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
