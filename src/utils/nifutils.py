import os
import re
from typing import Optional, List, Tuple, Any

from PIL import ImageChops, Image, ImageDraw
from io_scene_nifly.pynifly import NifFile
from pathlib import Path
from src.utils.logging_utils import logger
from src.utils.dds_utils import load_image

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


def collect_shape_uv_sets(shape: Any) -> List[List[Tuple[float, float]]]:
    """Return a list of UV sets for the given shape.

    This mirrors the robust UV collection used in the NIF editor so it can be
    shared with other widgets (e.g., palette creator). It attempts multiple
    nifly layouts to support different builds.
    """
    uvs_attr = getattr(shape, 'uvs', None)
    if uvs_attr is None:
        return []

    # If it's already a list of (u, v) tuples → single set
    if len(uvs_attr) > 0 and isinstance(uvs_attr[0], (tuple, list)) and \
            len(uvs_attr[0]) == 2 and not isinstance(uvs_attr[0][0], (tuple, list)):
        return [list(map(lambda p: (float(p[0]), float(p[1])), uvs_attr))]

    # If it's a list of sets (list[list[(u, v)]])
    if len(uvs_attr) > 0 and isinstance(uvs_attr[0], (list, tuple)) and \
            len(uvs_attr[0]) > 0 and isinstance(uvs_attr[0][0], (list, tuple)):
        sets: List[List[Tuple[float, float]]] = []
        for s in uvs_attr:
            sets.append([(float(p[0]), float(p[1])) for p in s])
        return sets

    # Some nifly builds expose shape.uv_sets
    uv_sets = getattr(shape, 'uv_sets', None)
    if uv_sets:
        sets2: List[List[Tuple[float, float]]] = []
        for s in uv_sets:
            sets2.append([(float(p[0]), float(p[1])) for p in s])
        return sets2

    return []


def build_uv_entries_for_nif(nif_path: Path) -> List[Tuple[List[int], int, str]]:
    """Collect grouped UV entries for a NIF, grouped by diffuse texture name.

    Returns a list of tuples: (shape_indices, uv_index, label)
    """
    entries: List[Tuple[List[int], int, str]] = []
    try:
        if not nif_path or nif_path.suffix.lower() != '.nif':
            return entries
        nif = load_nif(nif_path)
        shapes = list(getattr(nif, 'shapes', []))
        groups = {}
        for si, shape in enumerate(shapes):
            sets = collect_shape_uv_sets(shape)
            if not sets:
                continue
            tex_slots = shape.textures if hasattr(shape, 'textures') else None
            if not tex_slots:
                continue
            diffuse = tex_slots.get('Diffuse') if hasattr(tex_slots, 'get') else None
            if not diffuse:
                continue
            diffuse_str = str(diffuse)
            for ui, _ in enumerate(sets):
                key = (diffuse_str, ui)
                groups.setdefault(key, []).append(si)
        for (diffuse, ui), shape_indices in groups.items():
            label = f"{diffuse} - UV {ui}"
            entries.append((shape_indices, ui, label))
    except Exception as e:
        logger.warning(f"Failed to inspect UV sets: {e}")
    return entries


def maybe_fix_quarter_uv(uvs: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Optional UV scaling helper (ported from nif_edit)."""
    return [((float(u) - 0.5) * 2.0, (float(v) - 0.5) * 2.0) for (u, v) in uvs]


def build_mask_from_nif(
    nif_path: Path,
    uv_entries: List[Tuple[List[int], int, str]],
    selected_uv_index: int,
    tex_w: int,
    tex_h: int,
    scale_uvs: bool = False,
    wrap: bool = True,
) -> Optional[Image.Image]:
    """Build a combined UV mask for the given NIF and UV selection."""
    if not nif_path:
        return None
    try:
        nif = load_nif(nif_path)
        shapes = list(getattr(nif, 'shapes', []))
        idx = int(selected_uv_index)
        # Preferred path: use provided UV entries
        if 0 <= idx < len(uv_entries):
            entry = uv_entries[idx]
            shape_indices, uv_index, _label = entry
            combined = Image.new('L', (tex_w, tex_h), 0)
            any_mask = False
            for si in shape_indices:
                if not (0 <= si < len(shapes)):
                    continue
                shape = shapes[si]
                tris = getattr(shape, 'tris', None)
                if not tris:
                    continue
                sets = collect_shape_uv_sets(shape)
                if not sets or not (0 <= uv_index < len(sets)):
                    continue
                uvs = sets[uv_index]
                if scale_uvs:
                    uvs = maybe_fix_quarter_uv(uvs)
                mask = rasterize_uv_mask(tex_w, tex_h, uvs, tris, wrap=wrap)
                combined = ImageChops.lighter(combined, mask)
                any_mask = True
            if any_mask:
                return combined

        # Fallback: iterate UV sets globally if entries missing/out of date
        remaining = max(0, idx)
        for shape in shapes:
            tris = getattr(shape, 'tris', None)
            if not tris:
                continue
            sets = collect_shape_uv_sets(shape)
            if not sets:
                continue
            if remaining < len(sets):
                uvs = sets[remaining]
                if scale_uvs:
                    uvs = maybe_fix_quarter_uv(uvs)
                return rasterize_uv_mask(tex_w, tex_h, uvs, tris, wrap=wrap)
            else:
                remaining -= len(sets)
    except Exception as e:
        logger.warning(f"Failed to build UV mask: {e}")
        return None
    return None

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
    img = load_image(str(tex_path))
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