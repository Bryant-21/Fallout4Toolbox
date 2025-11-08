import os
import re
from typing import Optional, List, Tuple

from PIL import ImageChops, Image, ImageDraw
from io_scene_nifly.pynifly import NifFile
from pathlib import Path
from palette.palette_engine import load_image
from src.utils.logging_utils import logger

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

    # Build search directories. If the texture path is under Actors/Actor, also look in a deeper "CharacterAssets" folder.
    parts_lower = [p.lower() for p in tex_rel.parts]
    is_actor_path = ('actor' in parts_lower) or ('actors' in parts_lower)
    search_dirs: List[Path] = []
    if is_actor_path:
        search_dirs.append(meshes_dir / 'CharacterAssets')  # prefer CharacterAssets first when dealing with actors
    search_dirs.append(meshes_dir)

    # Special handling for Armor/PowerArmor where NIF names differ from texture names
    is_armor_path = ('armor' in parts_lower) or ('powerarmor' in parts_lower)
    name_lower = base_no_suffix.lower()
    armor_keywords = ["_leg", "_arm", "_head", "_helmet", "_body", "_torso"]

    if is_armor_path and any(k in name_lower for k in armor_keywords):
        # We are already searching the correct folder; filenames vary by body part.
        # Try wildcard matches using the detected keywords across all candidate directories.
        for d in search_dirs:
            if d.exists():
                for k in armor_keywords:
                    if k in name_lower:
                        # Prefer more specific matches first by sorting
                        for c in sorted(d.glob(f"*{k}*.nif")):
                            if c.exists():
                                return c
        # If nothing matched with keywords, fall back to the default logic below.

    # Default behavior: direct name matches and prefix matches across all candidate directories
    for d in search_dirs:
        candidates = [
            d / f"{base_no_suffix}_1.nif",
            d / f"{base_no_suffix}.nif",
        ]
        for c in candidates:
            if c.exists():
                return c
        if d.exists():
            for c in sorted(d.glob(f"{base_no_suffix}*.nif")):
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


def load_nif(nif_path) -> NifFile:
    if NifFile is None:
        raise RuntimeError('io_scene_nifly is not available in this environment')

    return NifFile(str(nif_path))


def remove_padding_from_texture_using_nif_uv(tex_path: Path, data_root: Path, wrap_uv=True) -> Optional[Image.Image]:


    # Load texture (RGBA)
    img = load_image(str(tex_path), 'RGBA')
    w, h = img.size

    nif_path = try_find_nif_for_texture(tex_path, data_root)
    if not nif_path or not nif_path.exists():
        logger.info(f"No NIF found for {tex_path}")
        return None

    nif = load_nif(nif_path)
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