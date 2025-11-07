import os
import re
import traceback
from pathlib import Path
from typing import Optional, Tuple, List
import tempfile

from PIL import Image, ImageDraw, ImageChops
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QLabel, QFileDialog, QMessageBox, QGridLayout, QCheckBox
from io_scene_nifly.pynifly import NifFile
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
from src.utils.logging_utils import logger


from mipflooding.wrapper import image_processing as _mip_image_processing


# Ensure the Nifly DLL is loaded before using NifFile (works in dev and PyInstaller)
try:
    import sys
    # Only load once per process
    nifly_loaded = getattr(NifFile, 'nifly', None) is not None
    if not nifly_loaded:
        candidates = []
        try:
            import io_scene_nifly  # package to locate DLL when not frozen
            nifly_dir = os.path.dirname(io_scene_nifly.__file__)
            candidates.append(os.path.join(nifly_dir, 'NiflyDLL.dll'))
        except Exception:
            pass
        # When frozen, PyInstaller extracts to sys._MEIPASS
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            # Common locations: root, package subdir
            candidates.append(os.path.join(meipass, 'NiflyDLL.dll'))
            candidates.append(os.path.join(meipass, 'io_scene_nifly', 'NiflyDLL.dll'))
        # Also try alongside the executable
        exe_dir = os.path.dirname(getattr(sys, 'executable', '') or '')
        if exe_dir:
            candidates.append(os.path.join(exe_dir, 'NiflyDLL.dll'))
            candidates.append(os.path.join(exe_dir, 'io_scene_nifly', 'NiflyDLL.dll'))
        loaded = False
        for cand in candidates:
            if cand and os.path.exists(cand):
                try:
                    NifFile.Load(cand)
                    logger.info(f"Loaded Nifly DLL from: {cand}")
                    loaded = True
                    break
                except Exception as _e:
                    logger.warning(f"Failed to load Nifly DLL at {cand}: {_e}")
        if not loaded:
            logger.warning("NiflyDLL.dll not found in expected locations. Ensure it is bundled with the app.")
except Exception as _ex:
    logger.warning(f"Error while attempting to load Nifly DLL: {_ex}")


DDS_DIFFUSE_RE = re.compile(r"_d\.dds$", re.IGNORECASE)

def uv_to_px(uv, w, h, wrap=True):
    u, v = uv

    if wrap:
        u = u % 1.0
        v = v % 1.0
    else:
        u = min(max(u, 0.0), 1.0)
        v = min(max(v, 0.0), 1.0)

    x = u * (w - 1)
    y = v * (h - 1)
    return x, y

def rasterize_uv_mask(w: int, h: int, uvs: List[Tuple[float, float]], tris: List[Tuple[int, int, int]], wrap=True) -> Image.Image:
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for i0, i1, i2 in tris:
        p0 = uv_to_px(uvs[i0], w, h, wrap)
        p1 = uv_to_px(uvs[i1], w, h, wrap)
        p2 = uv_to_px(uvs[i2], w, h, wrap)
        draw.polygon([p0, p1, p2], fill=255)
    return mask


# -----------------------------
# NIF/texture matching and processing
# -----------------------------

def diffuse_matches_texture(nif_tex_path: str, tex_path: Path) -> bool:
    nt = (nif_tex_path or '').replace('/', '\\').lower()
    tt = str(tex_path).replace('/', '\\').lower()
    if 'textures\\' in nt:
        nt_tail = nt[nt.index('textures\\'):]
    else:
        nt_tail = nt
    return tt.endswith(nt_tail)


def try_find_nif_for_texture(tex_path: Path, data_root: Path) -> Optional[Path]:
    try:
        tex_rel = tex_path.relative_to(data_root / 'Textures')
    except Exception:
        # If not under Textures, just bail
        return None
    base_stem = tex_rel.stem  # e.g., FlamerNozzle_d
    base_no_suffix = re.sub(r"_d$", "", base_stem, flags=re.IGNORECASE)
    meshes_dir = data_root / 'Meshes' / tex_rel.parent
    candidates = [
        meshes_dir / f"{base_no_suffix}_1.nif",
        meshes_dir / f"{base_no_suffix}.nif",
    ]
    for c in candidates:
        if c.exists():
            return c
    if meshes_dir.exists():
        for c in sorted(meshes_dir.glob(f"{base_no_suffix}*.nif")):
            return c
    return None

def resolve_diffuse_texture(path):
    """
    Returns the actual diffuse texture path.
    If path is .dds → return it.
    If path is .bgsm → parse BGSM to get the embedded diffuse texture.
    """
    path = path.replace("\\", "/")
    ext = os.path.splitext(path)[1].lower()

    if ext == ".dds":
        return path  # already a texture

    if ext == ".bgsm":
        try:
            from material_tools.bgsm_bin import read_bgsm  # If you have a bgsm reader installed
            mat = read_bgsm(path)
            return mat.DiffuseTexture  # normalized by BGSM parser
        except Exception:
            # Fallback: strip extension and assume naming consistent
            # (not perfect, but works in FO4 for many weapons)
            return path.replace(".bgsm", "_d.dds")

    return None

def remove_padding_from_texture_using_nif_uv(tex_path: Path, data_root: Path, wrap_uv=True) -> Optional[Image.Image]:
    if NifFile is None:
        raise RuntimeError('io_scene_nifly is not available in this environment')

    # Load texture (RGBA)
    img = load_image(str(tex_path), cfg.get(cfg.texconv_file), 'RGBA')
    w, h = img.size

    nif_path = try_find_nif_for_texture(tex_path, data_root)
    if not nif_path or not nif_path.exists():
        logger.info(f"No NIF found for {tex_path}")
        return None

    nif = NifFile(str(nif_path))
    any_match = False
    combined_mask = Image.new('L', (w, h), 0)

    for shape in nif.shapes:
        try:
            tex_slots = shape.textures if hasattr(shape, 'textures') else None
            if not tex_slots:
                continue

            if not tex_slots.get('Diffuse'):
                continue

            resolved_diffuse = resolve_diffuse_texture(str(tex_slots.get('Diffuse')))

            if not resolved_diffuse and not diffuse_matches_texture(resolved_diffuse, tex_path):
                continue

            uvs = shape.uvs if hasattr(shape, 'uvs') else []
            tris = shape.tris if hasattr(shape, 'tris') else []

            if not uvs or not tris:
                continue

            mask = rasterize_uv_mask(w, h, uvs, tris, wrap=wrap_uv)
            combined_mask = ImageChops.lighter(combined_mask, mask)
            any_match = True

        except Exception as e:
            logger.warning(f"Failed to process shape in {nif_path.name}: {e}")
            continue

    if not any_match:
        # Fallback union of all shapes
        for shape in nif.shapes:
            try:
                uvs = list(shape.uvs()) if hasattr(shape, 'uvs') else []
                tris = list(shape.tris()) if hasattr(shape, 'tris') else []
                if not uvs or not tris:
                    continue
                mask = rasterize_uv_mask(w, h, uvs, tris, wrap=wrap_uv)
                combined_mask = ImageChops.lighter(combined_mask, mask)
                any_match = True
            except Exception:
                pass

    if not any_match:
        logger.info(f"No UV data available for {tex_path}")
        return None

    r, g, b, a = img.split()
    bin_mask = combined_mask.point(lambda v: 255 if v > 0 else 0)
    new_alpha = ImageChops.multiply(a, bin_mask)
    out = Image.merge('RGBA', (r, g, b, new_alpha))
    return out


# -----------------------------
# UI Widget
# -----------------------------

def _apply_mip_flooding_to_png(out_path: Path, rgba_img: Image.Image) -> bool:
    """Apply mip flooding to a just-produced transparent PNG.

    - Creates temporary color (RGB) and alpha (mask) PNG files
    - Calls mipflooding.wrapper.image_processing.run_mip_flooding
    - Writes the corrected output to out_path (overwrites)
    Returns True if processed, False otherwise.
    """
    if _mip_image_processing is None:
        return False
    try:
        with tempfile.TemporaryDirectory(prefix="mipflood_") as tmpdir:
            tmpdir_p = Path(tmpdir)
            color_path = tmpdir_p / "color_C.png"
            mask_path = tmpdir_p / "mask_A.png"
            # Split channels
            if rgba_img.mode != 'RGBA':
                rgba_img = rgba_img.convert('RGBA')
            r, g, b, a = rgba_img.split()
            color_img = Image.merge('RGB', (r, g, b))
            # Save intermediates
            color_img.save(color_path, format='PNG')
            a.save(mask_path, format='PNG')
            # Run mip flooding
            _mip_image_processing.run_mip_flooding(str(color_path), str(mask_path), str(out_path))
            return True
    except Exception as e:
        logger.warning(f"Mip flooding failed for {out_path}: {e}")
        return False


def dilation_fill_static(out_path: Path, rgba_img: Image.Image, max_iters: int = 64) -> bool:
    try:
        if rgba_img.mode != 'RGBA':
            rgba_img = rgba_img.convert('RGBA')
        r, g, b, a = rgba_img.split()
        base_rgb = Image.merge('RGB', (r, g, b))
        known = a.point(lambda v: 255 if v > 0 else 0)
        unknown = a.point(lambda v: 0 if v > 0 else 255)
        unknown_initial = unknown.copy()
        if unknown.getbbox() is None:
            return False
        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        filled_any = False
        for _ in range(max_iters):
            iter_filled = False
            for dx, dy in neighbors:
                shifted_known = ImageChops.offset(known, dx, dy)
                fill_mask = ImageChops.multiply(shifted_known, unknown)
                if fill_mask.getbbox() is None:
                    continue
                shifted_rgb = ImageChops.offset(base_rgb, dx, dy)
                base_rgb.paste(shifted_rgb, mask=fill_mask)
                unknown = ImageChops.subtract(unknown, fill_mask)
                known = ImageChops.lighter(known, fill_mask)
                iter_filled = True
                filled_any = True
            if not iter_filled:
                break
        filled_mask = ImageChops.subtract(unknown_initial, unknown)
        new_alpha = ImageChops.lighter(a, filled_mask)
        nr, ng, nb = base_rgb.split()
        out_img = Image.merge('RGBA', (nr, ng, nb, new_alpha))
        out_img.save(out_path, format='PNG')
        if filled_any:
            logger.info(f"Color fill applied (alpha made opaque in filled regions): {out_path}")
        return filled_any
    except Exception as e:
        logger.warning(f"Color fill failed for {out_path}: {e}")
        return False


class UVPaddingWorker(QThread):
    progress = Signal(int, int)
    info = Signal(str)
    finished = Signal(int, int, int)
    error = Signal(str)

    def __init__(self, textures_dir: str, data_root: str, output_dir: Optional[str], do_mip_flooding: bool, do_dilation: bool):
        super().__init__()
        self.textures_dir = textures_dir
        self.data_root = data_root
        self.output_dir = output_dir
        self.do_mip_flooding = do_mip_flooding
        self.do_dilation = do_dilation
        self._stop = False

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
                        # Optional post-processing
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
            cfg.get(cfg.data_root_cfg),
        )
        self.data_root_card.clicked.connect(self._on_pick_data_root)

        self.textures_dir_card = PushSettingCard(
            self.tr("Textures Folder"),
            CustomIcons.IMAGE.icon(stroke=True),
            self.tr("Select a folder under Data/Textures"),
            cfg.get(cfg.textures_dir_cfg),
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



        self.chk_mip_flooding_card = SwitchSettingCard(icon=CustomIcons.FLOOD.icon(),
                                                 title=self.tr("Mip Flooding"),
                                                 content = "Fill the new transparent areas using mip flooding.",
                                                 configItem=cfg.mip_flooding)



        self.chk_color_fill = SwitchSettingCard(icon=CustomIcons.INFINITY.icon(),
                                                 title=self.tr("Dilation"),
                                                 content = "Apply standard infinite dilation to the new transparent areas to fill holes.",
                                                 configItem=cfg.color_fill)

        # Option toggles
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
        file_path, _ = QFileDialog.getOpenFileName(self, self.tr("Pick a diffuse DDS (_d.dds)"), textures_dir, self.tr("DDS files (*.dds)"))
        if not file_path:
            return
        if not DDS_DIFFUSE_RE.search(os.path.basename(file_path)):
            QMessageBox.information(self, self.tr("Info"), self.tr("Please select a *_d.dds texture."))
            return
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'show_progress'):
            try:
                p.show_progress()
            except Exception:
                pass
        try:
            orig = load_image(file_path, cfg.get(cfg.texconv_file), 'RGBA')
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
        do_mip = bool(cfg.get(cfg.mip_flooding))
        do_dilate = bool(cfg.get(cfg.color_fill))
        self.worker = UVPaddingWorker(textures_dir, data_root, out_dir, do_mip, do_dilate)
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
