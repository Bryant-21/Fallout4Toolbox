import json
import os
from fnmatch import fnmatch
from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QWidget, QLabel, QFileDialog, QMessageBox, QHBoxLayout, QVBoxLayout, QLineEdit
from qfluentwidgets import PushSettingCard, PrimaryPushButton, ConfigItem, FluentIcon as FIF

from src.help.palette_help import PaletteHelp
from src.settings.palette_settings import PaletteSettings
from src.utils.dds_utils import load_image, save_image
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger
from src.utils.cards import TextSettingCard, ComboBoxSettingsCard
from src.utils.appconfig import cfg
from src.utils.filesystem_utils import get_app_root


class AddToPaletteWidget(BaseWidget):
    """UI to add a new color row to an existing palette based on a new texture.

    Workflow (mirrors the palette pipeline used by `PaletteLUTGenerator`):

    - User selects:
        1) A palette segment NPZ (created by palette_creator).
        2) The existing palette texture (e.g. *_palette.png or .dds).
        3) The grayscale atlas texture (e.g. *_grayscale.png or .dds).
        4) A new color texture that matches the grayscale atlas layout.

    - The tool infers, for every pixel, which palette index it maps to using the
      same scaling as palette_creator (0..255 grayscale -> 0..palette_width-1).
    - For each palette index, it averages the corresponding colors from the new
      texture, fills gaps by interpolation, and appends the resulting row to the
      palette image.
    - The palette image is overwritten in-place.
    """

    def __init__(self, parent: Optional[QWidget], text: str = "Add To Palette"):
        super().__init__(parent=parent, text=text, vertical=True)

        # Paths / state
        self.npz_path: Optional[str] = None
        self.palette_path: Optional[str] = None
        self.greyscale_path: Optional[str] = None
        self.texture_path: Optional[str] = None
        self.directory_path: Optional[str] = None

        self.greyscale_img: Optional[Image.Image] = None
        self.texture_img: Optional[Image.Image] = None
        self.palette_img: Optional[Image.Image] = None

        self.settings_widget = PaletteSettings(self)
        self.settings_drawer.addWidget(self.settings_widget)
        self.help_widget = PaletteHelp(self)
        self.help_drawer.addWidget(self.help_widget)

        # NPZ (segments) selector
        self.npz_card = PushSettingCard(
            self.tr("Palette Segments (.npz)"),
            CustomIcons.FILE_CODE.icon(stroke=True),
            self.tr("Select NPZ created by the Palette Generator"),
            self.tr("No NPZ selected"),
        )
        self.npz_card.clicked.connect(self.on_select_npz)

        # Palette texture selector
        self.palette_card = PushSettingCard(
            self.tr("Palette Texture"),
            CustomIcons.PALETTE.icon(),
            self.tr("Select the existing palette texture (e.g. *_palette.png/.dds)"),
            self.tr("No palette selected"),
        )
        self.palette_card.clicked.connect(self.on_select_palette)

        # Greyscale atlas selector
        self.greyscale_card = PushSettingCard(
            self.tr("Greyscale Atlas"),
            CustomIcons.GREYSCALE.icon(),
            self.tr("Select the greyscale atlas used with this palette"),
            self.tr("No greyscale selected"),
        )
        self.greyscale_card.clicked.connect(self.on_select_greyscale)

        # New texture selector (single-file mode)
        self.texture_card = PushSettingCard(
            self.tr("Target Texture"),
            CustomIcons.IMAGE.icon(stroke=True),
            self.tr("Select texture that matches the greyscale to pull colors from"),
            self.tr("No texture selected"),
        )
        self.texture_card.clicked.connect(self.on_select_texture)

        # Directory selector (batch mode)
        self.directory_card = PushSettingCard(
            self.tr("Texture Directory"),
            CustomIcons.FOLDERRIGHT.icon(stroke=True),
            self.tr("Optionally select a directory to scan for matching textures"),
            self.tr("No directory selected"),
        )
        self.directory_card.clicked.connect(self.on_select_directory)

        # Filename filter input (used with directory mode)
        self.filename_filter = ConfigItem("palette", "filter", "*")

        self.filter_edit_card = TextSettingCard(
            self.filename_filter,
            FIF.FILTER,
            self.tr("Include Filter"),
            self.tr("Filename filter (optional, supports * and ?)")
        )



        # Action button
        self.apply_button = PrimaryPushButton(self.tr("Add Palette Row"))
        self.apply_button.clicked.connect(self.on_generate_row)

        # Simple preview: new texture vs expected result using the new row
        self.preview_original = QLabel(self.tr("New texture preview"))
        self.preview_original.setAlignment(Qt.AlignCenter)
        self.preview_original.setMinimumSize(256, 256)

        self.preview_result = QLabel(self.tr("Result with new palette row"))
        self.preview_result.setAlignment(Qt.AlignCenter)
        self.preview_result.setMinimumSize(256, 256)

        preview_layout = QHBoxLayout()
        preview_layout.addWidget(self.preview_original, 1)
        preview_layout.addWidget(self.preview_result, 1)

        preview_container = QWidget(self)
        preview_container.setLayout(preview_layout)

        # Layout
        self.addToFrame(self.npz_card)
        self.addToFrame(self.palette_card)
        self.addToFrame(self.greyscale_card)
        self.addToFrame(self.texture_card)
        self.addToFrame(self.directory_card)
        self.addToFrame(self.filter_edit_card)
        self.addToFrame(preview_container)

        self.addButtonBarToBottom(self.apply_button)

    # ----------------- File pickers -----------------
    def on_select_npz(self):
        # Default to the shared npz folder under the application root where
        # palette segment NPZ files are generated by the Palette Creator.
        default_dir = os.path.join(get_app_root(), "npz")
        try:
            os.makedirs(default_dir, exist_ok=True)
        except Exception:
            # If directory creation fails, fall back to the last-used/OS default.
            default_dir = ""

        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Palette Segments NPZ"),
            default_dir,
            self.tr("Palette Segments (*.npz);;All Files (*)"),
        )
        if not path:
            return
        try:
            data = np.load(path, allow_pickle=False)
            raw_meta = data.get("metadata")
            if raw_meta is None:
                raise ValueError("NPZ missing metadata")
            if hasattr(raw_meta, "item"):
                raw_meta = raw_meta.item()
            # Validate JSON structure but we don't need the full content here
            json.loads(str(raw_meta))
        except Exception as e:
            logger.exception("Failed to read NPZ: %s", e)
            QMessageBox.critical(self, "Error", self.tr("Failed to read NPZ file: %s") % str(e))
            return

        self.npz_path = path
        self.npz_card.setContent(os.path.basename(path))

    def on_select_palette(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Palette Texture"),
            "",
            self.tr("Images (*.png *.jpg *.jpeg *.bmp *.tga *.webp *.dds)"),
        )
        if not path:
            return

        img = load_image(path)
        self.palette_path = path
        self.palette_img = img.convert("RGBA")
        w, h = self.palette_img.size
        self.palette_card.setContent(f"{os.path.basename(path)} | {w}x{h}")

    def on_select_greyscale(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Greyscale Atlas"),
            "",
            self.tr("Images (*.png *.jpg *.jpeg *.bmp *.tga *.webp *.dds)"),
        )
        if not path:
            return
        try:
            img = load_image(path, f="L")
        except Exception as e:
            logger.exception("Failed to open greyscale image: %s", e)
            QMessageBox.critical(self, "Error", self.tr("Failed to open greyscale image: %s") % str(e))
            return

        self.greyscale_path = path
        self.greyscale_img = img
        w, h = img.size
        self.greyscale_card.setContent(f"{os.path.basename(path)} | {w}x{h} (L)")

    def on_select_texture(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select New Color Texture"),
            "",
            self.tr("Images (*.png *.jpg *.jpeg *.bmp *.tga *.webp *.dds)"),
        )
        if not path:
            return
        try:
            img = load_image(path)
        except Exception as e:
            logger.exception("Failed to open color texture: %s", e)
            QMessageBox.critical(self, "Error", self.tr("Failed to open color texture: %s") % str(e))
            return

        self.texture_path = path
        self.texture_img = img.convert("RGB")
        w, h = self.texture_img.size
        self.texture_card.setContent(f"{os.path.basename(path)} | {w}x{h}")
        self._update_preview_original()

    def on_select_directory(self):
        path = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Texture Directory"),
            "",
        )
        if not path:
            return

        self.directory_path = path
        self.directory_card.setContent(os.path.abspath(path))

    # ----------------- Core generation -----------------
    def on_generate_row(self):
        if not (self.npz_path and self.palette_img and self.greyscale_img):
            QMessageBox.warning(
                self,
                self.tr("Missing Input"),
                self.tr("Please select NPZ, palette texture, and greyscale atlas."),
            )
            return

        # Build list of textures to process.
        texture_paths: list[str] = []

        # Directory mode takes precedence if a directory is selected.
        if self.directory_path:
            exts = {".dds", ".png", ".jpg", ".jpeg", ".bmp", ".tga", ".webp"}
            pattern = self.filename_filter or "*"

            for root, _dirs, files in os.walk(self.directory_path):
                for fname in files:
                    _, ext = os.path.splitext(fname)
                    if ext.lower() not in exts:
                        continue
                    # Match filename (without path) against filter pattern (case-insensitive)
                    if fnmatch(fname.lower(), pattern.lower()):
                        texture_paths.append(os.path.join(root, fname))

            if not texture_paths:
                QMessageBox.warning(
                    self,
                    self.tr("No Matches"),
                    self.tr("No image files matching the filter were found in the selected directory."),
                )
                return

        elif self.texture_path:
            texture_paths.append(self.texture_path)
        else:
            QMessageBox.warning(
                self,
                self.tr("Missing Texture"),
                self.tr("Please select either an individual texture or a directory to scan."),
            )
            return

        try:
            self._generate_and_apply_rows(texture_paths)
        except Exception as e:
            logger.exception("Failed to generate palette row: %s", e)
            QMessageBox.critical(self, "Error", self.tr("Failed to generate palette row: %s") % str(e))

    def _compute_row_for_texture(self, texture_img: Image.Image, palette_size: int) -> np.ndarray:
        """Compute a palette row from a single color texture image.

        The texture is resized to match the greyscale atlas if needed, then the
        grayscale indices (0-255) are mapped to palette indices 0..(palette_size-1)
        and the corresponding colors are averaged per index.
        """
        assert self.greyscale_img is not None

        # Ensure the texture matches the greyscale atlas size (resize if needed)
        g_w, g_h = self.greyscale_img.size
        t_w, t_h = texture_img.size
        if (g_w, g_h) != (t_w, t_h):
            logger.info(
                "Resizing color texture from %dx%d to match greyscale %dx%d",
                t_w,
                t_h,
                g_w,
                g_h,
            )
            texture_img = texture_img.resize((g_w, g_h), Image.LANCZOS)

        grey_arr = np.array(self.greyscale_img, dtype=np.uint8)
        tex_arr = np.array(texture_img.convert("RGB"), dtype=np.uint8)

        # Compute palette index for each pixel
        idx_arr = np.rint(grey_arr.astype(np.float32) * (palette_size - 1) / 255.0).astype(np.int32)
        idx_arr = np.clip(idx_arr, 0, palette_size - 1)

        # Collect colors per palette index
        buckets = [[] for _ in range(palette_size)]
        flat_idx = idx_arr.reshape(-1)
        flat_colors = tex_arr.reshape(-1, 3)
        for i, idx in enumerate(flat_idx):
            buckets[int(idx)].append(flat_colors[i])

        new_row = np.zeros((palette_size, 3), dtype=np.uint8)
        has_data = np.zeros(palette_size, dtype=bool)

        for j in range(palette_size):
            if buckets[j]:
                cols = np.stack(buckets[j], axis=0).astype(np.float32)
                mean_col = cols.mean(axis=0)
                new_row[j] = np.clip(np.rint(mean_col), 0, 255).astype(np.uint8)
                has_data[j] = True

        if not has_data.any():
            raise ValueError("No usable pixels found to build a new palette row.")

        # Fill gaps by interpolation along the palette axis
        valid_indices = np.nonzero(has_data)[0]
        for j in range(palette_size):
            if has_data[j]:
                continue
            # Find nearest neighbors to interpolate
            left_candidates = valid_indices[valid_indices < j]
            right_candidates = valid_indices[valid_indices > j]

            if left_candidates.size == 0:
                left_idx = None
            else:
                left_idx = int(left_candidates[-1])

            if right_candidates.size == 0:
                right_idx = None
            else:
                right_idx = int(right_candidates[0])

            if left_idx is None and right_idx is None:
                continue
            elif left_idx is None:
                new_row[j] = new_row[right_idx]
            elif right_idx is None:
                new_row[j] = new_row[left_idx]
            else:
                span = float(right_idx - left_idx)
                t = float(j - left_idx) / span if span > 0 else 0.0
                col = (1.0 - t) * new_row[left_idx].astype(np.float32) + t * new_row[right_idx].astype(np.float32)
                new_row[j] = np.clip(np.rint(col), 0, 255).astype(np.uint8)

        return new_row

    def _generate_and_apply_rows(self, texture_paths: list[str]):
        """Generate palette rows for one or more textures and append them.

        Each texture contributes one palette row, and each appended strip is
        stored as 4 pixels high in the palette texture.
        """
        # Always reload palette from disk in case it changed externally
        img = load_image(self.palette_path)
        self.palette_img = img.convert("RGBA")
        w, h = self.palette_img.size
        self.palette_card.setContent(f"{os.path.basename(self.palette_path)} | {w}x{h}")

        p_w, _ = self.palette_img.size
        if p_w <= 1:
            raise ValueError("Palette width must be > 1")

        palette_size = p_w

        rows: list[np.ndarray] = []
        last_texture_img: Optional[Image.Image] = None

        for tex_path in texture_paths:
            try:
                tex_img = load_image(tex_path)
            except Exception as e:
                logger.exception("Failed to open color texture '%s': %s", tex_path, e)
                continue

            row = self._compute_row_for_texture(tex_img, palette_size)
            rows.append(row)
            last_texture_img = tex_img

        if not rows:
            raise ValueError("No palette rows could be generated from the selected textures.")

        pal_arr = np.array(self.palette_img, dtype=np.uint8)
        if pal_arr.ndim != 3 or pal_arr.shape[2] not in (3, 4):
            raise ValueError("Unsupported palette image format")

        h = pal_arr.shape[0]
        w = pal_arr.shape[1]
        if w != palette_size:
            raise ValueError(f"Palette width {w} does not match inferred size {palette_size}")

        rows_arr = np.stack(rows, axis=0)  # (N, W, 3)
        # Repeat each row 4 times vertically: shape becomes (N*4, W, 3)
        new_block = np.repeat(rows_arr, 4, axis=0)

        if pal_arr.shape[2] == 3:
            pal_new = np.concatenate([pal_arr, new_block], axis=0)
            mode = "RGB"
        else:
            row_rgba = np.zeros((new_block.shape[0], palette_size, 4), dtype=np.uint8)
            row_rgba[:, :, :3] = new_block
            row_rgba[:, :, 3] = 255
            pal_new = np.concatenate([pal_arr, row_rgba], axis=0)
            mode = "RGBA"

        new_palette_img = Image.fromarray(pal_new, mode=mode)

        # Overwrite existing palette texture
        save_image(new_palette_img, self.palette_path)

        # Update in-memory image and preview
        self.palette_img = new_palette_img

        QMessageBox.information(
            self,
            self.tr("Palette Updated"),
            self.tr("New palette rows were added and the palette texture was overwritten."),
        )

        # Update previews using the last successfully processed texture and row
        try:
            if last_texture_img is not None:
                # Show resized version used for computation
                g_w, g_h = self.greyscale_img.size  # type: ignore[union-attr]
                tex_preview = last_texture_img
                if tex_preview.size != (g_w, g_h):
                    tex_preview = tex_preview.resize((g_w, g_h), Image.LANCZOS)
                self._set_preview(self.preview_original, tex_preview)

            # Use the last generated row for the result preview
            self._update_result_preview(rows[-1])
        except Exception:
            logger.exception("Failed to update result preview")

    # ----------------- Preview helpers -----------------
    def _update_preview_original(self):
        if self.texture_img is None:
            return
        self._set_preview(self.preview_original, self.texture_img)

    def _update_result_preview(self, palette_row: np.ndarray):
        if self.greyscale_img is None:
            return
        # Re-use the PaletteApplier logic locally (simple LUT over greyscale)
        pw = palette_row.shape[0]
        if pw == 256:
            lut = palette_row
        else:
            x = np.linspace(0, pw - 1, num=pw)
            xi = np.linspace(0, pw - 1, num=256)
            lut = np.stack([
                np.interp(xi, x, palette_row[:, c]).astype(np.uint8) for c in range(3)
            ], axis=1)

        g = np.array(self.greyscale_img, dtype=np.uint8)
        colored = lut[g]
        img = Image.fromarray(colored, mode="RGB")
        self._set_preview(self.preview_result, img)

    def _set_preview(self, label: QLabel, img: Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img)
        h, w, _ = arr.shape
        qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        target_w = max(1, label.width())
        target_h = max(1, label.height())
        pix = pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pix)
