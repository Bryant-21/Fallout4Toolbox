import json
import os
import traceback
from pathlib import Path
from typing import List, Optional, Tuple, Any

from PIL import Image, ImageChops
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QLabel, QFileDialog, QGridLayout
from qfluentwidgets import (
    PushSettingCard,
    PrimaryPushButton,
    InfoBar,
    FluentIcon as FIF,
    SwitchSettingCard, PushButton,
)
from src.utils.cards import ComboBoxSettingsCard
from utils.capabilities import CAPABILITIES
from palette.palette_engine import load_image
from src.utils.appconfig import cfg
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger
from utils.imageutils import dilation_fill_static
from utils.mipflooding import _apply_mip_flooding_to_png
from utils.nifutils import load_nif, rasterize_uv_mask
from src.utils.chainner_utils import run_chainner, get_or_download_model


def _collect_shape_uv_sets(shape: Any) -> List[List[Tuple[float, float]]]:
    """
    Return a list of UV sets for the given shape.
    Tries multiple attribute layouts to be robust against nifly variations.
    """
    # Common case: shape.uvs is a flat list for the primary set
    uvs_attr = getattr(shape, 'uvs', None)
    if uvs_attr is None:
        return []

    # If it's already a list of (u,v) tuples → single set
    if len(uvs_attr) > 0 and isinstance(uvs_attr[0], (tuple, list)) and \
            len(uvs_attr[0]) == 2 and not isinstance(uvs_attr[0][0], (tuple, list)):
        return [list(map(lambda p: (float(p[0]), float(p[1])), uvs_attr))]

    # If it's a list of sets (list[list[(u,v)]])
    if len(uvs_attr) > 0 and isinstance(uvs_attr[0], (list, tuple)) and \
            len(uvs_attr[0]) > 0 and isinstance(uvs_attr[0][0], (list, tuple)):
        sets: List[List[Tuple[float, float]]] = []
        for s in uvs_attr:
            sets.append([ (float(p[0]), float(p[1])) for p in s ])
        return sets

    # Some nifly builds expose shape.uv_sets
    uv_sets = getattr(shape, 'uv_sets', None)
    if uv_sets:
        sets2: List[List[Tuple[float, float]]] = []
        for s in uv_sets:
            sets2.append([ (float(p[0]), float(p[1])) for p in s ])
        return sets2

    return []


def _read_uvw_file(path: Path) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int, int]]]:
    """
    Load UVs and triangle indices from a .uvw file.
    Supported format: JSON with keys {"uvs": [[u,v],...], "tris": [[i0,i1,i2],...]}
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        uvs = [ (float(u), float(v)) for u, v in data['uvs'] ]
        tris = [ (int(a), int(b), int(c)) for a, b, c in data['tris'] ]
        return uvs, tris
    except Exception as e:
        raise RuntimeError(f"Unsupported .uvw format. Expect JSON with 'uvs' and 'tris'. Error: {e}")


class SingleModelUVPadWidget(BaseWidget):
    """UI to remove padding for a single texture using a selected NIF (with selectable UV set)."""

    def __init__(self, parent: Optional[QWidget] = None, text: str = "Single Model UV Cleaner") -> None:
        super().__init__(text, parent, True)
        self.setObjectName('singleModelUvPad')
        self._title = text

        self.texture_path: Optional[Path] = None
        self.model_path: Optional[Path] = None  # .nif
        self.available_uv_set_count: int = 1
        self.selected_uv_set_index: int = 0
        # mapping of combobox index -> (shape_index, uv_set_index, label)
        self._uv_entries: List[Tuple[int, int, str]] = []

        # UI
        self.grid_layout = QGridLayout(self)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.grid_layout.setContentsMargins(20, 20, 20, 20)
        self.grid_layout.setHorizontalSpacing(12)
        self.grid_layout.setVerticalSpacing(12)

        self.card_pick_texture = PushSettingCard(
            title=self.tr("Pick Texture"),
            icon=CustomIcons.IMAGE.icon(stroke=True),
            text=self.tr("Select Texture"),
        )

        self.card_pick_model = PushSettingCard(
            title=self.tr("Pick NIF"),
            icon=CustomIcons.CUBE.icon(stroke=True),
            text=self.tr("Select a .nif"),
        )

        # UV set chooser (populated after model load)
        self.card_uv_set = ComboBoxSettingsCard(
            icon=FIF.GLOBE,
            title=self.tr("UV Set"),
            content=self.tr("If the model has multiple UV sets, choose which to use."),
        )
        self.card_uv_set.combox.addItems(["None"])

        if CAPABILITIES["mip_flooding"]:
            self.card_mip = SwitchSettingCard(
                configItem=cfg.mip_flooding,
                title=self.tr("Mip Flooding"),
                icon= CustomIcons.FLOOD.icon(),
                content=self.tr("Run mip flooding on the output to reduce edge artifacts."),
            )

        self.card_dilate = SwitchSettingCard(
            configItem=cfg.color_fill,
            title=self.tr("Infinite Dilation"),
            icon= CustomIcons.INFINITY.icon(),
            content="Apply infinite dilation to the new transparent areas to fill holes.",
        )

        self.btn_preview = PushButton(icon=FIF.ZOOM_IN, text="Preview")
        self.btn_save = PrimaryPushButton(icon=FIF.SAVE, text="Save")

        # Match nif_texture_edit dual preview layout
        self.original_label = QLabel(self.tr("Original"))
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setMinimumSize(450, 450)
        self.original_label.setStyleSheet("border: 1px dashed gray;")

        self.masked_label = QLabel(self.tr("No-Padding Preview"))
        self.masked_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.masked_label.setMinimumSize(450, 450)
        self.masked_label.setStyleSheet("border: 1px dashed gray;")

        self.addToFrame(self.card_pick_model)
        self.addToFrame(self.card_pick_texture)
        self.addToFrame(self.card_uv_set)
        # Optional AI Upscale toggle (requires ChaiNNer capability)
        if CAPABILITIES["ChaiNNer"]:
            self.card_ai_upscale = SwitchSettingCard(
                configItem=cfg.do_ai_upscale,
                title=self.tr("AI Upscale"),
                icon= CustomIcons.ENHANCE.icon(),
                content=self.tr("Run AI upscaler after cutting, before mip flooding/dilation."),
            )
            self.card_ai_upscale.switchButton.setChecked(False)
            self.addToFrame(self.card_ai_upscale)
        if CAPABILITIES["mip_flooding"]:
            self.addToFrame(self.card_mip)

        self.addToFrame(self.card_dilate)

        self.fix_scaled_uv = SwitchSettingCard(
            configItem=cfg.scale_uvs,
            title=self.tr("Scale UV"),
            icon=CustomIcons.ENHANCE.icon(),
            content=self.tr("Sometimes needed, not sure why."),
        )

        self.addToFrame(self.fix_scaled_uv)
        
        grid = QGridLayout()
        grid.addWidget(QLabel(self.tr("Original")), 0, 0)
        grid.addWidget(QLabel(self.tr("No-Padding")), 0, 1)
        grid.addWidget(self.original_label, 1, 0)
        grid.addWidget(self.masked_label, 1, 1)
        container = QWidget()
        container.setLayout(grid)
        self.boxLayout.addStretch(1)
        self.addToFrame(container)

        self.buttons_layout.addWidget(self.btn_preview, stretch=1)
        self.addButtonBarToBottom(self.btn_save)

        self._connect()


    def _connect(self) -> None:
        self.card_pick_texture.clicked.connect(self._on_pick_texture)
        self.card_pick_model.clicked.connect(self._on_pick_model)
        self.card_uv_set.combox.currentIndexChanged.connect(self._on_uv_index_changed)
        self.btn_preview.clicked.connect(self._on_preview)
        self.btn_save.clicked.connect(self._on_save)

    # ----- Slots -----
    def _on_pick_texture(self) -> None:
        file, _ = QFileDialog.getOpenFileName(self, self.tr("Select Texture"), cfg.get(cfg.data_root_cfg), "Image files (*.png *.jpg *.jpeg *.bmp *.tga *.dds)")
        if not file:
            return
        self.texture_path = Path(file)
        self.card_pick_texture.setContent(str(Path(file)))
        InfoBar.success(self.tr("Texture Selected"), str(self.texture_path), parent=self)

    def _on_pick_model(self) -> None:
        file, _ = QFileDialog.getOpenFileName(self, self.tr("Select NIF"), cfg.get(cfg.data_root_cfg), "NIF (*.nif)")
        if not file:
            return
        self.model_path = Path(file)
        self.card_pick_model.setContent(str(Path(file)))
        self._refresh_uv_set_count()
        InfoBar.success(self.tr("Model Selected"), str(self.model_path), parent=self)

    def _on_uv_index_changed(self, idx: int) -> None:
        self.selected_uv_set_index = idx

    # ----- UV utilities -----
    def _maybe_fix_quarter_uv(self, uvs: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        return [((float(u) - 0.5) * 2.0, (float(v) - 0.5) * 2.0) for (u, v) in uvs]

    # ----- Core logic -----
    def _refresh_uv_set_count(self) -> None:
        # Build user-friendly list of UV sets grouped by diffuse texture name
        self._uv_entries = []
        try:
            if self.model_path and self.model_path.suffix.lower() == '.nif':
                nif = load_nif(self.model_path)
                shapes = list(getattr(nif, 'shapes', []))
                # key: (diffuse_str, uv_index) -> list[int] shape indices
                groups = {}
                for si, shape in enumerate(shapes):
                    sets = _collect_shape_uv_sets(shape)
                    if not sets:
                        continue
                    tex_slots = shape.textures if hasattr(shape, 'textures') else None
                    if not tex_slots:
                        continue
                    if not tex_slots.get('Diffuse'):
                        continue
                    diffuse = str(tex_slots.get('Diffuse'))
                    for ui, _ in enumerate(sets):
                        key = (diffuse, ui)
                        if key not in groups:
                            groups[key] = []
                        groups[key].append(si)
                # Populate entries: (shape_indices, uv_index, label)
                for (diffuse, ui), shape_indices in groups.items():
                    label = f"{diffuse} - UV {ui}"
                    self._uv_entries.append((shape_indices, ui, label))
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"Failed to inspect UV sets: {e}")
            self._uv_entries = []
        # repopulate combobox
        self.card_uv_set.combox.clear()
        if self._uv_entries:
            items = [label for (_, _, label) in self._uv_entries]
            self.available_uv_set_count = len(items)
            self.card_uv_set.combox.addItems(items)
        else:
            # Fallback to previous behavior
            self.available_uv_set_count = 1
            self.card_uv_set.combox.addItems(["UV 0"])
        self.card_uv_set.combox.setCurrentIndex(0)
        self.selected_uv_set_index = 0

    def _build_mask_from_model(self, tex_w: int, tex_h: int) -> Optional[Image.Image]:
        if not self.model_path:
            return None
        ext = self.model_path.suffix.lower()
        try:
            if ext == '.nif':
                nif = load_nif(self.model_path)
                shapes = list(getattr(nif, 'shapes', []))
                # Use mapping built in _refresh_uv_set_count for user-friendly selection
                idx = int(self.selected_uv_set_index)
                if 0 <= idx < len(self._uv_entries):
                    entry = self._uv_entries[idx]
                    # Support both old (shape_index,int) and new (list[int],int) formats
                    shape_indices, uv_index = None, None
                    if isinstance(entry[0], list):
                        shape_indices, uv_index, _label = entry
                    else:
                        shape_index, uv_index, _label = entry
                        shape_indices = [shape_index]
                    # Combine masks for all shapes in this group
                    from PIL import ImageChops as _IC
                    combined = Image.new('L', (tex_w, tex_h), 0)
                    any_mask = False
                    for si in shape_indices:
                        if not (0 <= si < len(shapes)):
                            continue
                        shape = shapes[si]
                        tris = getattr(shape, 'tris', None)
                        if not tris:
                            continue
                        sets = _collect_shape_uv_sets(shape)
                        if not sets or not (0 <= uv_index < len(sets)):
                            continue
                        uvs = sets[uv_index]
                        if(cfg.get(cfg.scale_uvs)):
                            uvs = self._maybe_fix_quarter_uv(uvs)
                        mask = rasterize_uv_mask(tex_w, tex_h, uvs, tris, wrap=True)
                        combined = _IC.lighter(combined, mask)
                        any_mask = True
                    if any_mask:
                        return combined
                # Fallback: previous global-index iteration if mapping missing/out of date
                remaining = max(0, idx)
                for shape in shapes:
                    tris = getattr(shape, 'tris', None)
                    if not tris:
                        continue
                    sets = _collect_shape_uv_sets(shape)
                    if not sets:
                        continue
                    if remaining < len(sets):
                        uvs = sets[remaining]
                        if cfg.get(cfg.scale_uvs):
                            uvs = self._maybe_fix_quarter_uv(uvs)
                        return rasterize_uv_mask(tex_w, tex_h, uvs, tris, wrap=True)
                    else:
                        remaining -= len(sets)
                return None
        except Exception as e:
            logger.warning(f"Failed to build mask: {e}")
            return None
        return None

    def _process(self) -> Optional[Image.Image]:
        if not self.texture_path:
            InfoBar.error(self.tr("Missing Texture"), self.tr("Please select a texture file."), parent=self)
            return None
        if not self.model_path:
            InfoBar.error(self.tr("Missing Model"), self.tr("Please select a NIF file."), parent=self)
            return None

        try:
            img = load_image(str(self.texture_path), 'RGBA')
        except Exception as e:
            InfoBar.error(self.tr("Failed to load texture"), str(e), parent=self)
            return None
        w, h = img.size

        mask = self._build_mask_from_model(w, h)
        if mask is None:
            InfoBar.warning(self.tr("No UVs"), self.tr("No UV data was found to build a mask."), parent=self)
            return None

        r, g, b, a = img.split()
        bin_mask = mask.point(lambda v: 255 if v > 0 else 0)
        new_alpha = ImageChops.multiply(a, bin_mask)
        out = Image.merge('RGBA', (r, g, b, new_alpha))
        return out

    # Preview and Save
    def _on_preview(self) -> None:
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'show_progress'):
            try:
                p.show_progress()
            except Exception:
                pass
        try:
            # Show original on left if available
            if not self.texture_path:
                InfoBar.error(self.tr("Missing Texture"), self.tr("Please select a texture file."), parent=self)
                return
            try:
                orig = load_image(str(self.texture_path), 'RGBA')
                self._display_on_label(orig, self.original_label)
            except Exception as e:
                InfoBar.error(self.tr("Failed to load texture"), str(e), parent=self)
                return

            out = self._process()
            if out is None:
                self.masked_label.setText(self.tr("No mask produced"))
                return
            self._display_on_label(out, self.masked_label)
        finally:
            p = getattr(self, 'parent', None)
            if p and hasattr(p, 'complete_loader'):
                try:
                    p.complete_loader()
                except Exception:
                    pass

    def _on_save(self) -> None:
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'show_progress'):
            try:
                p.show_progress()
            except Exception:
                pass
        try:
            out = self._process()
            if out is None:
                return
            default_name = (self.texture_path.stem + "_uvpad.png") if self.texture_path else "output.png"
            file, _ = QFileDialog.getSaveFileName(self, self.tr("Save Output"), default_name, "PNG (*.png)")
            if not file:
                return
            out_path = Path(file)
            try:
                out.save(out_path, format='PNG')
                # Optional AI Upscale step: after cutting, before mip flooding/dilation
                did_upscale = False
                final_path = out_path
                final_img = out
                if CAPABILITIES["ChaiNNer"] and hasattr(self, 'card_ai_upscale') and self.card_ai_upscale.switchButton.isChecked():
                    # Choose model based on texture name (_n treated as normals)
                    tex_name = (self.texture_path.name if self.texture_path else "").lower()
                    model_name = cfg.get(cfg.upscale_normals_cfg) if tex_name.endswith('_n.dds') or tex_name.endswith('_n.png') else cfg.get(cfg.upscale_textures_cfg)
                    try:
                        model_path = get_or_download_model(model_name)
                        expected = out_path.with_name(out_path.stem + "_upscaled.png")
                        ok = run_chainner(str(out_path), model_path, str(out_path.parent), str(expected))
                        if ok and expected.exists():
                            did_upscale = True
                            final_path = expected
                            try:
                                from PIL import Image as _PILImage
                                final_img = _PILImage.open(expected).convert('RGBA')
                            except Exception:
                                final_img = out
                        else:
                            did_upscale = False
                    except Exception:
                        did_upscale = False
                # Post-processing on the (possibly) upscaled output
                did_mip = False
                if cfg.get(cfg.mip_flooding) and CAPABILITIES["mip_flooding"]:
                    did_mip = _apply_mip_flooding_to_png(final_path, final_img)
                did_dilate = False
                if cfg.get(cfg.color_fill):
                    did_dilate = dilation_fill_static(final_path, final_img)
                InfoBar.success(
                    self.tr("Saved"),
                    self.tr(f"Saved to {Path(final_path).name} (ai: {'yes' if did_upscale else 'no'}, mip: {'yes' if did_mip else 'no'}, dilate: {'yes' if did_dilate else 'no'})"),
                    parent=self,
                )
            except Exception as e:
                InfoBar.error(self.tr("Failed to save"), str(e), parent=self)
        finally:
            p = getattr(self, 'parent', None)
            if p and hasattr(p, 'complete_loader'):
                try:
                    p.complete_loader()
                except Exception:
                    pass


    def _display_on_label(self, image: Image.Image, label: QLabel) -> None:
        """Display a PIL Image on a QLabel, scaled to fit while keeping aspect ratio."""
        try:
            from PySide6.QtGui import QImage, QPixmap
            rgba = image.convert('RGBA')
            w, h = rgba.size
            data = rgba.tobytes('raw', 'RGBA')
            qimg = QImage(data, w, h, QImage.Format.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimg)
            label.setPixmap(pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        except Exception:
            # As a fallback, set text
            label.setText(self.tr("Preview unavailable"))
