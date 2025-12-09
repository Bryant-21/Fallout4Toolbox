import json
import math
import os
import logging

import cv2
import imagequant
import numpy as np
from PIL import Image
from PIL.Image import Quantize, Palette
from scipy import interpolate, ndimage

from src.utils.appconfig import QuantAlgorithm
from src.utils.appconfig import cfg
from src.utils.filesystem_utils import get_app_root
from src.utils.logging_utils import logger

SEMI_TRANSPARENT_ALPHA_THRESHOLD = 250


# --- Quantize -------------------------------------------------------------
def quantize_image(i, method: QuantAlgorithm = QuantAlgorithm.libimagequant, final_colors: int = 0):
    """Quantize image using the specified method

    Implements optional two-stage strategy (1): over-quantize per-image to preserve rare colors,
    leaving global reduction to later steps. Controlled by cfg.ci_advanced_quant.
    Also biases libimagequant toward quality over speed when available.
    """

    if isinstance(method, QuantAlgorithm):
        method = str(method.value).lower()

    method = method.lower()
    info = {'method': method}
    logger.debug(f"Quantizing with method: {method}")

    # Determine palette sizes (cap at 128 per new pipeline requirements)
    if final_colors is None or final_colors <= 0:
        final_colors = int(cfg.get(cfg.ci_default_quant_size))

    try:

        if method == "median_cut":
            # Must be RGB
            image = i.convert('RGB')
            quantized = image.quantize(colors=final_colors, method=Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
            info['description'] = "Median Cut - Good color relationships, can be blocky"

        elif method == "max_coverage":
            # Must be RGB
            image = i.convert('RGB')
            quantized = image.quantize(colors=final_colors, method=Quantize.MAXCOVERAGE, dither=Image.Dither.FLOYDSTEINBERG)
            info['description'] = "Max Coverage - Maximizes color variety"

        elif method == "fast_octree":
            # Must be RGB
            image = i
            quantized = image.quantize(colors=final_colors, method=Quantize.FASTOCTREE, dither=Image.Dither.FLOYDSTEINBERG)
            info['description'] = "Fast Octree - Fast, good for photos"

        elif method == "libimagequant":
            try:
                image = i
                # Favor quality over speed by allowing more colors and optional dithering.
                quantized = imagequant.quantize_pil_image(
                    image,
                    dithering_level=0.5,
                    max_colors=final_colors,
                    min_quality=90,
                    max_quality=100,
                )

                info['description'] = "LibImageQuant - High quality (favoring quality over speed)"
            except Exception as e:
                logger.warning(f"LibImageQuant failed with method {method}: {str(e)}")
                quantized = image.quantize(colors=final_colors, method=Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
                info['description'] = "LibImageQuant (fallback to Median Cut)"

        elif method == "kmeans_adaptive":
            image = i
            # Use a larger k when advanced enabled so k-means can place centroids on rare colors
            quantized = image.quantize(colors=final_colors, method=Quantize.FASTOCTREE, kmeans=final_colors, dither=Image.Dither.FLOYDSTEINBERG)
            info['description'] = "K-means Adaptive - Adaptive color distribution"

        elif method == "uniform":
            image = i.convert('RGB')
            # For uniform method, ensure we get close to target colors
            uniform_img = image.convert("P", palette=Palette.ADAPTIVE, colors=final_colors)
            quantized = uniform_img.convert("RGB").quantize(colors=final_colors, dither=Image.Dither.FLOYDSTEINBERG)
            info['description'] = "Uniform - Helps with color banding"

        else:
            image = i.convert('RGB')
            quantized = image.quantize(colors=final_colors, method=Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
            info['description'] = "Median Cut (default)"

        return quantized
    except Exception as e:
        logger.error(f"Quantization failed with method {method}: {str(e)}")
        raise


def _apply_semi_transparent_mode(rgba: np.ndarray, mode: str, threshold: int = SEMI_TRANSPARENT_ALPHA_THRESHOLD) -> np.ndarray:
    """Normalize semi-transparent pixels according to the configured mode.

    Modes:
      - "mask": set alpha<threshold to 0 (remove semi-transparent)
      - "nearest_fill": copy nearest opaque RGB into semi-transparent pixels, then set alpha to 0
      - "premultiply_snap": premultiply RGB by alpha, then snap alpha to 0/255 and clear RGB where alpha==0
    """
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
        return rgba

    mode = (mode or "mask").strip().lower()
    out = rgba.copy()
    alpha = out[:, :, 3].astype(np.uint8)
    solid = alpha >= threshold

    if mode == "mask":
        out[:, :, 3] = np.where(solid, 255, 0).astype(np.uint8)
        out[~solid, :3] = 0
        return out

    if mode == "nearest_fill":
        out[:, :, 3] = np.where(solid, 255, 0).astype(np.uint8)
        if solid.any():
            transparent = ~solid
            if transparent.any():
                _, nearest_indices = ndimage.distance_transform_edt(transparent, return_indices=True)
                ny, nx = nearest_indices
                rgb = out[:, :, :3]
                rgb[transparent] = rgb[ny[transparent], nx[transparent]]
        else:
            out[:, :, :3] = 0
        return out

    if mode == "premultiply_snap":
        alpha_f = alpha.astype(np.float32) / 255.0
        premult = (out[:, :, :3].astype(np.float32) * alpha_f[:, :, None]).clip(0, 255).astype(np.uint8)
        out[:, :, :3] = premult
        out[:, :, 3] = np.where(solid, 255, 0).astype(np.uint8)
        out[~solid, :3] = 0
        return out

    return out


# -- Palette retrieval helper
def get_palette(q_img: Image.Image):
    """Return the palette rows actually referenced by the P-mode image (order-preserving).

    This preserves the original row order from the P image's palette to keep
    index-to-color mappings intact. Only rows up to the maximum used index are returned.

    Returns:
        numpy.ndarray of shape (N, 3) where N = max_used_index + 1 (or 0 for empty images).
    """
    # Raw palette rows in palette order (no deduplication to preserve indices)
    palette_raw = np.array(q_img.getpalette(), dtype=np.uint8).reshape(-1, 3)

    # Determine the highest palette index actually referenced by pixels.
    idx_img = np.array(q_img, dtype=np.uint8)
    if idx_img.size == 0:
        # No pixels; return empty palette to avoid accidental bad indexing
        return palette_raw[:0]

    max_idx = int(idx_img.max())
    needed = max_idx + 1
    # Guard against malformed palettes (ensure we don't slice beyond bounds)
    needed = min(needed, palette_raw.shape[0])
    return palette_raw[:needed]

def get_palette_row(palette_img, y=0) -> np.ndarray:
    w, h = palette_img.size
    y = max(0, min(h - 1, y))
    row_pixels = np.array(palette_img)[y, :, :3]
    if row_pixels.ndim == 1:
        row_pixels = np.expand_dims(row_pixels, axis=0)
    return row_pixels.astype(np.uint8)


# --- Island state helpers -------------------------------------------------
def load_island_npz(npz_path: str):
    """Load palette island metadata and masks from a saved NPZ.

    Returns (metadata_dict, mask_stack_bool, islands_list)
    islands_list is a list of tuples: (name, gray_start, gray_end).
    """
    if not npz_path or not os.path.isfile(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    raw_meta = data.get("metadata")
    if raw_meta is None:
        raise ValueError("Missing metadata in NPZ")
    if hasattr(raw_meta, "item"):
        raw_meta = raw_meta.item()
    metadata = json.loads(str(raw_meta))

    mask_stack = data.get("masks")
    if mask_stack is None:
        raise ValueError("Missing masks in NPZ")
    mask_stack = mask_stack.astype(bool)

    islands = []
    for entry in metadata.get("islands", []):
        name = entry.get("name", "")
        gs = int(entry.get("gray_start", 0))
        ge = int(entry.get("gray_end", 0))
        islands.append((name, gs, ge))

    return metadata, mask_stack, islands


def _map_grey_to_palette_indices(grey: np.ndarray, palette_size: int) -> np.ndarray:
    if palette_size <= 1:
        return np.zeros_like(grey, dtype=np.int32)
    scale = 255.0 / float(palette_size - 1)
    mapped = np.rint(grey.astype(np.float32) / scale)
    return np.clip(mapped, 0, palette_size - 1).astype(np.int32)


def build_palette_row_from_recolor(grey_img: Image.Image,
                                   recolor_img: Image.Image,
                                   islands: list,
                                   mask_stack: np.ndarray,
                                   palette_size: int) -> np.ndarray:
    """Reconstruct a palette row from a recolored image using saved island mappings.

    Args:
        grey_img: Grayscale atlas produced by palette_creator (values 0-255).
        recolor_img: Recolored version of the original source texture (aligned size).
        islands: List of (name, gray_start, gray_end).
        mask_stack: Boolean mask stack aligned to image (N, H, W) for islands.
        palette_size: Target palette width.

    Returns:
        palette_row: np.ndarray shape (palette_size, 3) uint8.
    """
    if grey_img is None or recolor_img is None:
        raise ValueError("Grey image and recolor image are required")

    grey_arr = np.array(grey_img.convert('L'), dtype=np.uint8)
    # Match palette_creator behavior: quantize recolor before extracting colors
    quant_method = cfg.get(cfg.ci_default_quant_method) if hasattr(cfg, "ci_default_quant_method") else None
    quantized = quantize_image(recolor_img.convert('RGB'), quant_method) if quant_method is not None else recolor_img.convert('RGB')
    # Preserve original alpha for mask filtering, but use quantized RGB for color sampling
    recolor_alpha = np.array(recolor_img.convert('RGBA'), dtype=np.uint8)[:, :, 3]
    recolor_rgb = np.array(quantized.convert('RGB'), dtype=np.uint8)
    recolor_rgba = np.dstack([recolor_rgb, recolor_alpha])
    h, w = grey_arr.shape
    if recolor_rgba.shape[0] != h or recolor_rgba.shape[1] != w:
        raise ValueError("Recolored image size does not match greyscale")

    if mask_stack is not None and mask_stack.size > 0:
        if mask_stack.shape[1] != h or mask_stack.shape[2] != w:
            raise ValueError("Mask stack size does not match greyscale")
    else:
        mask_stack = np.zeros((0, h, w), dtype=bool)

    palette_indices = _map_grey_to_palette_indices(grey_arr, palette_size)
    palette_row = np.zeros((palette_size, 3), dtype=np.uint8)

    # Build per-gray value color lists limited to island masks
    all_masks_union = np.zeros((h, w), dtype=bool)
    island_colors = {}
    for idx, (name, gs, ge) in enumerate(islands):
        mask = mask_stack[idx] if idx < mask_stack.shape[0] else np.zeros((h, w), dtype=bool)
        all_masks_union |= mask

        color_map = {g: [] for g in range(gs, ge + 1)}
        island_pixels = mask
        if not island_pixels.any():
            island_colors[name] = color_map
            continue

        # Respect alpha in recolor but keep NPZ as the authoritative mask for transparency
        alpha = recolor_rgba[:, :, 3]
        valid = island_pixels & (alpha > 0)
        if not valid.any():
            island_colors[name] = color_map
            continue

        island_pal_indices = palette_indices[valid]
        island_rgb = recolor_rgba[:, :, :3][valid]

        for rgb, gray in zip(island_rgb, island_pal_indices):
            if gray < gs or gray > ge:
                continue
            color_map[gray].append(rgb)

        island_colors[name] = color_map

    # Fill palette row with averaged/interpolated colors per island
    for idx, (name, gs, ge) in enumerate(islands):
        colors = island_colors.get(name, {})
        if not colors:
            continue
        for g in range(gs, min(ge + 1, palette_size)):
            entries = colors.get(g, [])
            if entries:
                palette_row[g] = np.mean(np.stack(entries, axis=0), axis=0).astype(np.uint8)
            else:
                # interpolate from nearest known neighbors within this island
                prev_val, next_val = None, None
                for gg in range(g - 1, gs - 1, -1):
                    if colors.get(gg):
                        prev_val = gg
                        break
                for gg in range(g + 1, ge + 1):
                    if colors.get(gg):
                        next_val = gg
                        break
                if prev_val is not None and next_val is not None:
                    t = (g - prev_val) / float(next_val - prev_val)
                    color = (1 - t) * np.mean(colors[prev_val], axis=0) + t * np.mean(colors[next_val], axis=0)
                    palette_row[g] = np.clip(color, 0, 255).astype(np.uint8)
                elif prev_val is not None:
                    palette_row[g] = np.mean(colors[prev_val], axis=0).astype(np.uint8)
                elif next_val is not None:
                    palette_row[g] = np.mean(colors[next_val], axis=0).astype(np.uint8)

    return palette_row


def apply_palette_to_greyscale(palette_img: Image.Image, grey_img: Image.Image, palette_row=None, filter_type=None) -> Image.Image:
    """Apply palette row to a greyscale image, preserving alpha if present.

    Accepts grey_img in modes:
      - 'L' (grayscale)
      - 'LA' (grayscale with alpha)
      - 'RGB'/'RGBA' (uses the first channel as greyscale index; preserves alpha if present)
    
    Args:
        palette_img: The palette image to sample colors from
        grey_img: The greyscale image to colorize
        palette_row: Optional pre-extracted palette row
        filter_type: "linear" for smooth interpolation, "nearest" for exact color preservation.
                     If None, uses the config setting ci_palette_filter_type.
    
    Returns RGB if no alpha, RGBA if alpha present.
    """
    if palette_row is None or palette_row.size == 0:
        palette_row = get_palette_row(palette_img)

    # Get filter type from config if not specified
    if filter_type is None:
        filter_type = cfg.get(cfg.ci_palette_filter_type)

    pw = palette_row.shape[0]

    if pw == 256:
        lut = palette_row
    else:
        if filter_type == "nearest":
            # Nearest neighbor: map each of 256 greyscale values to closest palette index
            # This preserves exact colors without blending
            indices = np.round(np.linspace(0, pw - 1, num=256)).astype(int)
            lut = palette_row[indices]

        elif filter_type == "linear":
            # Linear interpolation: smooth blending between colors (default)
            x = np.linspace(0, pw - 1, num=pw)
            xi = np.linspace(0, pw - 1, num=256)
            lut = np.stack([
                np.interp(xi, x, palette_row[:, c]).astype(np.uint8) for c in range(3)
            ], axis=1)

        elif filter_type == "cubic":
            # Cubic interpolation: smoother blending using 4 neighboring points
            # Creates a more continuous gradient with less "knot" feeling
            x = np.linspace(0, pw - 1, num=pw)
            xi = np.linspace(0, pw - 1, num=256)

            lut = np.zeros((256, 3), dtype=np.uint8)
            for c in range(3):
                # Use cubic spline interpolation
                f = interpolate.interp1d(x, palette_row[:, c], kind='cubic',
                                         fill_value='extrapolate')
                interpolated = f(xi)
                # Clip to valid range and convert to uint8
                lut[:, c] = np.clip(interpolated, 0, 255).astype(np.uint8)

        elif filter_type == "gaussian":
            # Gaussian filtering: applies smoothing before interpolation
            # Helps reduce noise/banding in the palette
            x = np.linspace(0, pw - 1, num=pw)
            xi = np.linspace(0, pw - 1, num=256)

            # Calculate sigma based on palette width (adjustable)
            # Smaller sigma = less smoothing, larger sigma = more smoothing
            sigma = max(1.0, pw / 64)  # Adjust divisor for desired smoothing

            lut = np.zeros((256, 3), dtype=np.uint8)
            for c in range(3):
                # Reshape to 2D for GaussianBlur (height=1, width=pw)
                channel_2d = palette_row[:, c].reshape(1, -1).astype(np.float32)

                # Apply Gaussian filter (kernel size automatically calculated from sigma)
                smoothed = cv2.GaussianBlur(channel_2d, (0, 0), sigmaX=sigma)

                # Linear interpolation on smoothed data
                lut[:, c] = np.interp(xi, x, smoothed.flatten()).astype(np.uint8)

        elif filter_type == "cubic_gaussian":
            # Combined approach: Gaussian smoothing followed by cubic interpolation
            x = np.linspace(0, pw - 1, num=pw)
            xi = np.linspace(0, pw - 1, num=256)

            sigma = max(0.5, pw / 128)  # Lighter smoothing for cubic combo

            lut = np.zeros((256, 3), dtype=np.uint8)
            for c in range(3):
                # Apply light Gaussian smoothing
                channel_2d = palette_row[:, c].reshape(1, -1).astype(np.float32)
                smoothed = cv2.GaussianBlur(channel_2d, (0, 0), sigmaX=sigma)

                # Cubic interpolation on smoothed data
                f = interpolate.interp1d(x, smoothed.flatten(), kind='cubic',
                                         fill_value='extrapolate')
                interpolated = f(xi)
                lut[:, c] = np.clip(interpolated, 0, 255).astype(np.uint8)

        else:
            # Default to linear if unknown filter type
            x = np.linspace(0, pw - 1, num=pw)
            xi = np.linspace(0, pw - 1, num=256)
            lut = np.stack([
                np.interp(xi, x, palette_row[:, c]).astype(np.uint8) for c in range(3)
            ], axis=1)

    # Extract greyscale channel and optional alpha
    alpha = None
    mode = grey_img.mode
    if mode == 'L':
        g = np.array(grey_img, dtype=np.uint8)
    elif mode == 'LA':
        arr = np.array(grey_img, dtype=np.uint8)
        g = arr[:, :, 0]
        alpha = arr[:, :, 1]
    elif mode in ('RGBA', 'RGBa'):
        arr = np.array(grey_img, dtype=np.uint8)
        g = arr[:, :, 0]
        alpha = arr[:, :, 3]
    elif mode == 'RGB':
        arr = np.array(grey_img, dtype=np.uint8)
        g = arr[:, :, 0]
    else:
        # Fallback: convert to L and proceed
        g = np.array(grey_img.convert('L'), dtype=np.uint8)

    colored = lut[g]
    rgb_img = Image.fromarray(colored, mode='RGB')
    if alpha is not None:
        a_img = Image.fromarray(alpha, mode='L')
        return Image.merge('RGBA', (rgb_img.split()[0], rgb_img.split()[1], rgb_img.split()[2], a_img))
    return rgb_img


# Save grayscale atlas
def fill_transparent_with_nearest(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill transparent pixels by copying the nearest non-transparent value (fast EDT-based)."""
    if mask is None or img.shape != mask.shape:
        return img
    if not mask.any():
        return img

    transparent = ~mask
    if not transparent.any():
        return img

    # Distance transform: for each pixel, get indices of the nearest non-transparent pixel
    # ndimage.distance_transform_edt returns (distance, indices); indices has shape (ndim, h, w)
    _, nearest_indices = ndimage.distance_transform_edt(transparent, return_indices=True)
    nearest_y, nearest_x = nearest_indices

    filled = img.copy()
    filled[transparent] = img[nearest_y[transparent], nearest_x[transparent]]
    return filled


# --- Shared palette/island generation helpers --------------------------------


def _lab_image(rgb_image: np.ndarray) -> np.ndarray:
    """Convert an RGB image array to Lab (float32)."""
    return np.array(Image.fromarray(rgb_image, mode='RGB').convert('LAB'), dtype=np.float32)


def _lab_histogram(lab_pixels: np.ndarray) -> np.ndarray:
    """Compute 8x8x8 (512-bin) Lab histogram normalized to 1."""
    l_bins = np.clip((lab_pixels[:, 0] / 100.0 * 8).astype(np.int32), 0, 7)
    a_bins = np.clip(((lab_pixels[:, 1] + 128.0) / 255.0 * 8).astype(np.int32), 0, 7)
    b_bins = np.clip(((lab_pixels[:, 2] + 128.0) / 255.0 * 8).astype(np.int32), 0, 7)
    idx = l_bins * 64 + a_bins * 8 + b_bins
    hist = np.bincount(idx, minlength=512).astype(np.float32)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist /= hist_sum
    return hist


def _histogram_intersection_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Histogram intersection distance (0 same, 1 disjoint)."""
    return 1.0 - float(np.minimum(h1, h2).sum())


def _mean_lab_distance(m1: np.ndarray, m2: np.ndarray) -> float:
    """Perceptual Lab distance normalized to ~0..1."""
    return float(np.linalg.norm(m1 - m2) / 100.0)


def _lab_bin_center(bin_index: int) -> np.ndarray:
    l_bin = bin_index // 64
    rem = bin_index % 64
    a_bin = rem // 8
    b_bin = rem % 8
    l_center = (l_bin + 0.5) * (100.0 / 8.0)
    a_center = (a_bin + 0.5) * (255.0 / 8.0) - 128.0
    b_center = (b_bin + 0.5) * (255.0 / 8.0) - 128.0
    return np.array([l_center, a_center, b_center], dtype=np.float32)


def _dominant_bin_guard(comp_hist: np.ndarray, grp_hist: np.ndarray, share_gap_max: float = 0.25, center_tol: float = 15.0) -> bool:
    top_bin_comp = int(comp_hist.argmax())
    top_share_comp = float(comp_hist[top_bin_comp])
    top_bin_grp = int(grp_hist.argmax())
    top_share_grp = float(grp_hist[top_bin_grp])

    if top_bin_comp == top_bin_grp:
        share_gap = abs(top_share_comp - top_share_grp)
        if share_gap > share_gap_max:
            return False

        comp_center = _lab_bin_center(top_bin_comp)
        grp_center = _lab_bin_center(top_bin_grp)
        return float(np.linalg.norm(comp_center - grp_center)) <= center_tol

    if top_share_comp < 0.60 and top_share_grp < 0.60:
        comp_center = _lab_bin_center(top_bin_comp)
        grp_center = _lab_bin_center(top_bin_grp)
        if float(np.linalg.norm(comp_center - grp_center)) <= center_tol * 1.25:
            return True

    overlap = float(np.minimum(comp_hist, grp_hist).sum())
    return overlap >= 0.55


def auto_create_islands_from_rgba(rgba: np.ndarray,
                                  palette_size: int,
                                  desired_islands: int = 4,
                                  min_pixels: int = 8) -> tuple[list[tuple[str, int, int]], np.ndarray, bool]:
    """Headless version of palette_creator.auto_create_islands.

    Returns islands list, mask stack, and overflow flag (True when unique colors exceed slot capacity).
    Raises ValueError when input is invalid.
    """
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
        raise ValueError("RGBA image required for island generation")

    semi_mode = cfg.get(cfg.ci_semi_transparent_mode) if hasattr(cfg, "ci_semi_transparent_mode") else "mask"
    rgba = _apply_semi_transparent_mode(rgba, semi_mode, SEMI_TRANSPARENT_ALPHA_THRESHOLD)

    if palette_size <= 0:
        raise ValueError("Palette size must be greater than zero to auto create islands.")

    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3]
    lab_image = _lab_image(rgb)
    non_transparent = alpha > 0

    if not non_transparent.any():
        raise ValueError("Image has no opaque pixels.")

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    labels, num = ndimage.label(non_transparent, structure=structure)

    if num == 0:
        raise ValueError("No regions detected.")

    slices = ndimage.find_objects(labels)

    components = []

    for lbl in range(1, num + 1):
        sl = slices[lbl - 1]
        if sl is None:
            continue
        lbl_region = labels[sl]
        region_mask = lbl_region == lbl
        pixel_count = int(region_mask.sum())
        if pixel_count < min_pixels:
            continue

        region_rgb = rgb[sl][region_mask]
        region_lab = lab_image[sl][region_mask]
        if region_rgb.size == 0 or region_lab.size == 0:
            continue

        unique_colors = np.unique(region_rgb.reshape(-1, 3), axis=0)

        hist = _lab_histogram(region_lab)
        mean_lab = region_lab.mean(axis=0)

        components.append({
            "slice": sl,
            "mask": region_mask,
            "hist": hist,
            "mean_lab": mean_lab,
            "pixels": pixel_count,
            "unique_colors": set(map(tuple, unique_colors.tolist())),
            "regions": [(sl, region_mask)],
        })

    if not components:
        raise ValueError("No sufficiently large regions found.")

    base_size = palette_size // desired_islands
    remainder = palette_size % desired_islands

    island_specs = []
    current_start = 0
    for i in range(desired_islands):
        size = base_size + (1 if i < remainder else 0)
        if size <= 0:
            continue
        gray_start = current_start
        gray_end = current_start + size - 1
        island_specs.append({
            "gray_start": gray_start,
            "gray_end": gray_end,
            "capacity": size
        })
        current_start += size

    if not island_specs:
        raise ValueError("Unable to divide the palette into islands with the current palette size.")

    hist_weight = 0.75

    def _comp_group_score(comp: dict, ref_hist: np.ndarray, ref_mean: np.ndarray) -> float:
        d_hist = _histogram_intersection_distance(comp["hist"], ref_hist)
        d_mean = _mean_lab_distance(comp["mean_lab"], ref_mean)
        base = hist_weight * d_hist + (1.0 - hist_weight) * d_mean
        if not _dominant_bin_guard(comp["hist"], ref_hist):
            base += 0.25
        return base

    components_sorted = sorted(components, key=lambda c: c["pixels"], reverse=True)
    seeds: list[dict] = []
    if components_sorted:
        seeds.append(components_sorted[0])
        remaining = components_sorted[1:]
        while len(seeds) < min(desired_islands, len(components_sorted)) and remaining:
            far_idx = None
            far_score = -1.0
            for idx, cand in enumerate(remaining):
                min_dist = min(_comp_group_score(cand, s["hist"], s["mean_lab"]) for s in seeds)
                if min_dist > far_score:
                    far_score = min_dist
                    far_idx = idx
            seeds.append(remaining.pop(far_idx))

    groups: list[dict] = []
    for seed in seeds:
        groups.append({
            "hist_centroid": seed["hist"],
            "mean_lab_centroid": seed["mean_lab"],
            "pixel_total": seed["pixels"],
            "unique_colors": set(seed["unique_colors"]),
            "regions": [(sl, m) for sl, m in seed.get("regions", [(seed["slice"], seed["mask"])])]
        })

    for comp in components_sorted:
        if comp in seeds:
            continue

        best_idx = None
        best_score = math.inf

        for idx, grp in enumerate(groups):
            score = _comp_group_score(comp, grp["hist_centroid"], grp["mean_lab_centroid"])
            if score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            continue

        grp = groups[best_idx]
        total_pixels = grp["pixel_total"] + comp["pixels"]
        grp["hist_centroid"] = (grp["hist_centroid"] * grp["pixel_total"] + comp["hist"] * comp["pixels"]) / total_pixels
        grp["mean_lab_centroid"] = (grp["mean_lab_centroid"] * grp["pixel_total"] + comp["mean_lab"] * comp["pixels"]) / total_pixels
        grp["pixel_total"] = total_pixels
        grp["unique_colors"].update(comp["unique_colors"])
        grp["regions"].extend(comp.get("regions", [(comp["slice"], comp["mask"])]))

    while len(groups) < desired_islands:
        groups.append({
            "hist_centroid": np.zeros(512, dtype=np.float32),
            "mean_lab_centroid": np.zeros(3, dtype=np.float32),
            "pixel_total": 0,
            "unique_colors": set(),
            "regions": []
        })

    island_data: list[dict | None] = [None] * len(island_specs)
    groups_sorted_for_capacity = sorted(enumerate(groups), key=lambda t: len(t[1]["unique_colors"]), reverse=True)
    specs_sorted_by_capacity = sorted(enumerate(island_specs), key=lambda t: t[1]["capacity"], reverse=True)

    overflow_flag = False

    for (grp_idx, grp), (spec_idx, spec) in zip(groups_sorted_for_capacity, specs_sorted_by_capacity):
        mask = np.zeros(non_transparent.shape, dtype=bool)
        for sl, m in grp.get("regions", []):
            mask[sl][m] = True

        island_data[spec_idx] = {
            "gray_start": spec["gray_start"],
            "gray_end": spec["gray_end"],
            "capacity": spec["capacity"],
            "unique_colors": set(grp["unique_colors"]),
            "mask": mask,
            "pixel_total": grp.get("pixel_total", 0),
        }

        if len(grp["unique_colors"]) > spec["capacity"]:
            overflow_flag = True

    for idx, spec in enumerate(island_specs):
        if island_data[idx] is None:
            island_data[idx] = {
                "gray_start": spec["gray_start"],
                "gray_end": spec["gray_end"],
                "capacity": spec["capacity"],
                "unique_colors": set(),
                "mask": np.zeros(non_transparent.shape, dtype=bool),
                "pixel_total": 0,
            }

    combined_mask = np.zeros(non_transparent.shape, dtype=bool)
    for isl in island_data:
        combined_mask |= isl["mask"]

    leftovers = non_transparent & ~combined_mask
    if leftovers.any():
        def remaining_capacity(idx: int) -> tuple[int, int]:
            isl = island_data[idx]
            rem = isl["capacity"] - len(isl["unique_colors"])
            return rem, idx

        target_idx = max(range(len(island_data)), key=remaining_capacity)
        target = island_data[target_idx]
        target["mask"] |= leftovers

        leftover_colors = set(map(tuple, rgb[leftovers].reshape(-1, 3)))
        target["unique_colors"].update(leftover_colors)
        if len(target["unique_colors"]) > target["capacity"]:
            overflow_flag = True

    islands: list[tuple[str, int, int]] = []
    mask_stack = []
    for idx, isl in enumerate(island_data, start=1):
        island_name = f"AutoIsland_{idx}"
        gray_start, gray_end = isl["gray_start"], isl["gray_end"]
        islands.append((island_name, gray_start, gray_end))
        mask_stack.append(isl["mask"].astype(bool, copy=False))

    mask_stack_arr = np.stack(mask_stack, axis=0) if mask_stack else np.zeros((0,) + non_transparent.shape, dtype=bool)
    return islands, mask_stack_arr, overflow_flag


def _map_luminosity_default(luminosity: np.ndarray, gray_start: int, gray_end: int, 
                            palette_to_game_scale: float, guard_band_width: int = 0) -> np.ndarray:
    """Default luminosity-based linear mapping (current behavior)."""
    lum_min = luminosity.min()
    lum_max = luminosity.max()
    
    if lum_max - lum_min < 1:
        lum_max = lum_min + 1
    
    normalized = (luminosity - lum_min) / (lum_max - lum_min)
    remapped_palette_space = gray_start + normalized * (gray_end - gray_start)
    return (remapped_palette_space * palette_to_game_scale).astype(np.uint8)


def _map_guard_bands_quantile(luminosity: np.ndarray, gray_start: int, gray_end: int,
                               palette_to_game_scale: float, guard_band_width: int = 1) -> np.ndarray:
    """Hybrid: Guard bands + quantile distribution (recommended)."""
    effective_start = gray_start + guard_band_width
    effective_end = gray_end - guard_band_width
    effective_range = max(1, effective_end - effective_start + 1)
    
    lum_min = luminosity.min()
    lum_max = luminosity.max()
    
    if lum_max - lum_min < 1:
        lum_max = lum_min + 1
    
    # Sort luminosity values and assign ranks
    sorted_indices = np.argsort(luminosity)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(luminosity))
    
    # Map ranks to effective range using quantile distribution
    # This preserves all unique colors by spreading them evenly
    rank_normalized = ranks.astype(np.float32) / max(1, len(luminosity) - 1)
    remapped_palette_space = effective_start + rank_normalized * (effective_end - effective_start)
    
    return (remapped_palette_space * palette_to_game_scale).astype(np.uint8)


def _map_quantile(luminosity: np.ndarray, gray_start: int, gray_end: int,
                  palette_to_game_scale: float, guard_band_width: int = 0) -> np.ndarray:
    """Quantile-based distribution without guard bands."""
    effective_range = max(1, gray_end - gray_start + 1)
    
    lum_min = luminosity.min()
    lum_max = luminosity.max()
    
    if lum_max - lum_min < 1:
        lum_max = lum_min + 1
    
    # Sort luminosity values and assign ranks
    sorted_indices = np.argsort(luminosity)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(luminosity))
    
    # Map ranks to palette range using quantile distribution
    # This preserves all unique colors by spreading them evenly
    rank_normalized = ranks.astype(np.float32) / max(1, len(luminosity) - 1)
    remapped_palette_space = gray_start + rank_normalized * (gray_end - gray_start)
    
    return (remapped_palette_space * palette_to_game_scale).astype(np.uint8)


def _map_guard_bands(luminosity: np.ndarray, gray_start: int, gray_end: int,
                     palette_to_game_scale: float, guard_band_width: int = 1) -> np.ndarray:
    """Simple guard bands with luminosity mapping."""
    effective_start = gray_start + guard_band_width
    effective_end = gray_end - guard_band_width
    effective_range = max(1, effective_end - effective_start)
    
    lum_min = luminosity.min()
    lum_max = luminosity.max()
    
    if lum_max - lum_min < 1:
        lum_max = lum_min + 1
    
    normalized = (luminosity - lum_min) / (lum_max - lum_min)
    remapped_palette_space = effective_start + normalized * effective_range
    return (remapped_palette_space * palette_to_game_scale).astype(np.uint8)


def _map_color_clustering(rgb_array: np.ndarray, mask: np.ndarray, gray_start: int, gray_end: int,
                          palette_to_game_scale: float, guard_band_width: int = 0) -> np.ndarray:
    """Hue-based color clustering (preserves color identity over brightness)."""
    island_rgb = rgb_array[mask]
    unique_colors, inverse = np.unique(island_rgb.reshape(-1, 3), axis=0, return_inverse=True)
    
    # Convert to HSV and sort by hue
    if unique_colors.shape[0] > 0:
        # Reshape for cv2
        unique_rgb_img = unique_colors.reshape(1, -1, 3).astype(np.uint8)
        hsv_colors = cv2.cvtColor(unique_rgb_img, cv2.COLOR_RGB2HSV).reshape(-1, 3)
        sorted_indices = np.argsort(hsv_colors[:, 0])  # Sort by hue
        
        # Assign sorted colors to palette range
        num_colors = len(unique_colors)
        palette_indices = np.linspace(gray_start, gray_end, num_colors).astype(int)
        
        # Create mapping from color to index
        color_to_index = np.zeros(num_colors, dtype=np.uint8)
        for i, sorted_idx in enumerate(sorted_indices):
            color_to_index[sorted_idx] = int(palette_indices[i] * palette_to_game_scale)
        
        # Map all pixels using inverse indices
        result = np.zeros(mask.shape, dtype=np.uint8)
        result[mask] = color_to_index[inverse]
        return result
    else:
        return np.zeros(mask.shape, dtype=np.uint8)


def _map_perceptual(rgb_array: np.ndarray, mask: np.ndarray, gray_start: int, gray_end: int,
                    palette_to_game_scale: float, guard_band_width: int = 0) -> np.ndarray:
    """Perceptual brightness using CIE Lab L* channel."""
    from skimage import color as skcolor
    
    # Convert masked region to Lab
    island_rgb = rgb_array[mask]
    if island_rgb.size == 0:
        return np.zeros(mask.shape, dtype=np.uint8)
    
    # Normalize to 0-1 for skimage
    island_rgb_float = island_rgb.astype(np.float32) / 255.0
    lab_pixels = skcolor.rgb2lab(island_rgb_float.reshape(-1, 3))
    perceptual_luminosity = lab_pixels[:, 0]  # L* channel (0-100)
    
    lum_min = perceptual_luminosity.min()
    lum_max = perceptual_luminosity.max()
    
    if lum_max - lum_min < 1:
        lum_max = lum_min + 1
    
    normalized = (perceptual_luminosity - lum_min) / (lum_max - lum_min)
    remapped_palette_space = gray_start + normalized * (gray_end - gray_start)
    
    result = np.zeros(mask.shape, dtype=np.uint8)
    result[mask] = (remapped_palette_space * palette_to_game_scale).astype(np.uint8)
    return result


def _map_reverse_luminosity(luminosity: np.ndarray, gray_start: int, gray_end: int,
                            palette_to_game_scale: float, guard_band_width: int = 0) -> np.ndarray:
    """Reverse luminosity mapping (dark -> high indices, bright -> low indices)."""
    lum_min = luminosity.min()
    lum_max = luminosity.max()
    
    if lum_max - lum_min < 1:
        lum_max = lum_min + 1
    
    normalized = (luminosity - lum_min) / (lum_max - lum_min)
    # Reverse the mapping
    remapped_palette_space = gray_end - normalized * (gray_end - gray_start)
    return (remapped_palette_space * palette_to_game_scale).astype(np.uint8)


def _map_alternating_luminosity(luminosity: np.ndarray, gray_start: int, gray_end: int,
                                 palette_to_game_scale: float, guard_band_width: int = 0,
                                 island_index: int = 0) -> np.ndarray:
    """Alternating luminosity mapping where direction reverses per island.
    
    - Island 0 (even): bright -> high indices (normal)
    - Island 1 (odd): bright -> low indices (reversed)
    - Island 2 (even): bright -> high indices (normal)
    - And so on...
    """
    lum_min = luminosity.min()
    lum_max = luminosity.max()
    
    if lum_max - lum_min < 1:
        lum_max = lum_min + 1
    
    normalized = (luminosity - lum_min) / (lum_max - lum_min)
    
    # Alternate direction based on island index
    if island_index % 2 == 0:
        # Even islands: normal mapping (low to high)
        remapped_palette_space = gray_start + normalized * (gray_end - gray_start)
    else:
        # Odd islands: reversed mapping (high to low)
        remapped_palette_space = gray_end - normalized * (gray_end - gray_start)
    
    return (remapped_palette_space * palette_to_game_scale).astype(np.uint8)


def _map_nearest_neighbor_reserve(luminosity: np.ndarray, gray_start: int, gray_end: int,
                                   palette_to_game_scale: float, guard_band_width: int = 0) -> np.ndarray:
    """Reserve first and last pixels as guard bands, map to effective range only.
    
    This strategy:
    - Reserves gray_start and gray_end as boundary guard pixels
    - Maps all pixels to the effective range (gray_start + 1 to gray_end - 1)
    - Total usable colors = island_size - 2
    - Guard bands will be filled later with nearest neighbor from effective range
    """
    # Effective range excludes first and last index
    effective_start = gray_start + 1
    effective_end = gray_end - 1
    
    # If island is too small, fall back to using full range
    if effective_end < effective_start:
        effective_start = gray_start
        effective_end = gray_end
    
    effective_range = max(1, effective_end - effective_start)
    
    lum_min = luminosity.min()
    lum_max = luminosity.max()
    
    if lum_max - lum_min < 1:
        lum_max = lum_min + 1
    
    normalized = (luminosity - lum_min) / (lum_max - lum_min)
    remapped_palette_space = effective_start + normalized * effective_range
    return (remapped_palette_space * palette_to_game_scale).astype(np.uint8)


def _smooth_palette_gradient(palette_row: np.ndarray, method: str = "gaussian", 
                             strength: float = 1.0) -> np.ndarray:
    """Smooth harsh transitions in palette to reduce interpolation artifacts.
    
    Args:
        palette_row: (N, 3) palette colors
        method: "gaussian", "median", or "bilateral"
        strength: 0.0 (no smoothing) to 1.0 (maximum smoothing)
    
    Returns:
        Smoothed palette row
    """
    if strength <= 0.0 or palette_row.shape[0] < 3:
        return palette_row
    
    smoothed = palette_row.copy().astype(np.float32)
    
    if method == "gaussian":
        # Gaussian blur: smooth based on spatial proximity
        sigma = max(0.5, strength * palette_row.shape[0] / 32)
        for c in range(3):
            channel = smoothed[:, c].reshape(1, -1)
            blurred = cv2.GaussianBlur(channel, (0, 0), sigmaX=sigma)
            smoothed[:, c] = blurred.flatten()
    
    elif method == "median":
        # Median filter: preserves edges better while smoothing
        kernel_size = max(3, int(strength * 9))
        if kernel_size % 2 == 0:
            kernel_size += 1
        for c in range(3):
            smoothed[:, c] = ndimage.median_filter(smoothed[:, c], size=kernel_size)
    
    elif method == "bilateral":
        # Bilateral: edge-preserving smoothing (best quality, slower)
        # Smooth similar colors more, preserve color boundaries
        sigma_color = 25.0 * (1.0 - strength * 0.5)  # Color similarity
        sigma_space = max(1.0, strength * palette_row.shape[0] / 16)  # Spatial smoothing
        
        # Reshape for cv2.bilateralFilter (needs 2D image)
        palette_2d = palette_row.reshape(1, -1, 3).astype(np.uint8)
        smoothed_2d = cv2.bilateralFilter(palette_2d, d=-1, 
                                          sigmaColor=sigma_color, 
                                          sigmaSpace=sigma_space)
        smoothed = smoothed_2d.reshape(-1, 3).astype(np.float32)
    
    return np.clip(smoothed, 0, 255).astype(np.uint8)


def _smooth_palette_image(palette_img: np.ndarray, method: str = "gaussian", 
                          strength: float = 1.0) -> np.ndarray:
    """Smooth the entire palette image to reduce artifacts.
    
    This operates on a full 2D palette image (height x width x 3) rather than a single row.
    Smoothing is applied horizontally only to preserve the row structure.
    
    Args:
        palette_img: (H, W, 3) palette image
        method: "gaussian", "median", or "bilateral"
        strength: 0.0 (no smoothing) to 1.0 (maximum smoothing)
    
    Returns:
        Smoothed palette image
    """
    if strength <= 0.0 or palette_img.shape[1] < 3:
        return palette_img
    
    palette_width = palette_img.shape[1]
    smoothed = palette_img.copy().astype(np.uint8)
    
    if method == "gaussian":
        # Gaussian blur: smooth horizontally only (along width axis)
        sigma_x = max(0.5, strength * palette_width / 32)
        # Use sigmaY=0 to disable vertical smoothing (keep rows identical)
        smoothed = cv2.GaussianBlur(smoothed, (0, 0), sigmaX=sigma_x, sigmaY=0)
    
    elif method == "median":
        # Median filter: horizontal only
        kernel_width = max(3, int(strength * 9))
        if kernel_width % 2 == 0:
            kernel_width += 1
        # Apply to each row independently
        for row in range(smoothed.shape[0]):
            for c in range(3):
                smoothed[row, :, c] = ndimage.median_filter(smoothed[row, :, c], size=kernel_width)
    
    elif method == "bilateral":
        # Bilateral: edge-preserving smoothing
        sigma_color = 25.0 * (1.0 - strength * 0.5)
        sigma_space = max(1.0, strength * palette_width / 16)
        smoothed = cv2.bilateralFilter(smoothed, d=-1, 
                                       sigmaColor=sigma_color, 
                                       sigmaSpace=sigma_space)
    
    return smoothed.astype(np.uint8)


def _upscale_palette_to_256(palette_img: np.ndarray, original_width: int) -> np.ndarray:
    """Upscale palette from original_width to 256 width using interpolation.
    
    The palette should already be duplicated to the desired height before calling this.
    This function scales horizontally only, preserving all rows identically.
    
    Args:
        palette_img: (H, W, 3) palette image where W = original_width
        original_width: Original palette width (e.g., 128)
    
    Returns:
        Upscaled palette image (H, 256, 3)
    """
    if original_width >= 256 or palette_img.shape[1] != original_width:
        return palette_img
    
    palette_height = palette_img.shape[0]
    
    # Use PIL for high-quality resizing with LANCZOS filter
    palette_pil = Image.fromarray(palette_img, mode='RGB')
    upscaled_pil = palette_pil.resize((256, palette_height), Image.Resampling.LANCZOS)
    
    return np.array(upscaled_pil, dtype=np.uint8)


def _fill_guard_bands(palette_row: np.ndarray, islands: list[tuple[str, int, int]], 
                      guard_band_width: int) -> None:
    """Fill guard band indices with interpolated colors between islands."""
    if guard_band_width <= 0:
        return
    
    for i in range(len(islands) - 1):
        curr_name, curr_start, curr_end = islands[i]
        next_name, next_start, next_end = islands[i + 1]
        
        # Check if islands are adjacent
        if curr_end + 1 == next_start:
            # Get colors from both islands (avoid guard bands themselves)
            curr_safe = curr_end - guard_band_width
            next_safe = next_start + guard_band_width
            
            if curr_safe >= curr_start and next_safe <= next_end:
                curr_color = palette_row[curr_safe].astype(np.float32)
                next_color = palette_row[next_safe].astype(np.float32)
                
                # Fill boundary with weighted interpolation
                # 67/33 split favoring the "owning" island
                if curr_end < len(palette_row):
                    palette_row[curr_end] = (0.67 * curr_color + 0.33 * next_color).astype(np.uint8)
                if next_start < len(palette_row):
                    palette_row[next_start] = (0.33 * curr_color + 0.67 * next_color).astype(np.uint8)


def _fill_nearest_neighbor_guard_bands(palette_row: np.ndarray, islands: list[tuple[str, int, int]]) -> None:
    """Fill first and last indices of each island with nearest neighbor colors.
    
    For nearest_neighbor_reserve strategy:
    - First index (gray_start) copies from gray_start + 1
    - Last index (gray_end) copies from gray_end - 1
    """
    for island_name, gray_start, gray_end in islands:
        island_size = gray_end - gray_start + 1
        
        # Only fill if island is large enough to have effective range
        if island_size > 2:
            # First index copies from next index
            if gray_start < len(palette_row) and gray_start + 1 < len(palette_row):
                palette_row[gray_start] = palette_row[gray_start + 1]
            
            # Last index copies from previous index
            if gray_end < len(palette_row) and gray_end - 1 >= 0:
                palette_row[gray_end] = palette_row[gray_end - 1]


def build_grayscale_and_palette_from_islands(rgba: np.ndarray,
                                             islands: list[tuple[str, int, int]],
                                             mask_stack: np.ndarray,
                                             palette_size: int,
                                             palette_height: int = 16) -> tuple[np.ndarray, Image.Image, np.ndarray]:
    """Headless version of palette_creator.generate_both core pipeline."""
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
        raise ValueError("RGBA image required for palette generation")

    semi_mode = cfg.get(cfg.ci_semi_transparent_mode) if hasattr(cfg, "ci_semi_transparent_mode") else "mask"
    rgba = _apply_semi_transparent_mode(rgba, semi_mode, SEMI_TRANSPARENT_ALPHA_THRESHOLD)

    height, width = rgba.shape[:2]
    alpha_channel = rgba[:, :, 3]
    non_transparent = alpha_channel > 0
    rgb_array = rgba[:, :, :3]

    luminosity = (0.299 * rgb_array[:, :, 0] + 0.587 * rgb_array[:, :, 1] + 0.114 * rgb_array[:, :, 2])
    grayscale_output = np.zeros((height, width), dtype=np.uint8)

    palette_to_game_scale = 1.0 if palette_size <= 1 else 255.0 / float(palette_size - 1)
    
    # Get greyscale mapping strategy from config
    mapping_strategy = cfg.get(cfg.ci_greyscale_mapping_strategy) if hasattr(cfg, "ci_greyscale_mapping_strategy") else "luminosity"
    guard_band_width = int(cfg.get(cfg.ci_guard_band_width)) if hasattr(cfg, "ci_guard_band_width") else 0

    masks = []
    if mask_stack is not None and mask_stack.size > 0:
        masks = [mask_stack[idx].astype(bool, copy=False) for idx in range(min(mask_stack.shape[0], len(islands)))]
    # Ensure one mask per island
    while len(masks) < len(islands):
        masks.append(np.zeros((height, width), dtype=bool))

    island_colors = {}

    for island_index, ((island_name, gray_start, gray_end), mask) in enumerate(zip(islands, masks)):
        if mask is None:
            continue

        # Ensure transparent pixels are excluded even if the mask was drawn over them.
        mask = mask & non_transparent

        if not mask.any():
            continue

        island_luminosity = luminosity[mask]
        if island_luminosity.size == 0:
            continue

        # Apply selected greyscale mapping strategy
        if mapping_strategy == "guard_bands_quantile":
            remapped = _map_guard_bands_quantile(island_luminosity, gray_start, gray_end, 
                                                  palette_to_game_scale, guard_band_width)
            grayscale_output[mask] = remapped
        elif mapping_strategy == "quantile":
            remapped = _map_quantile(island_luminosity, gray_start, gray_end, 
                                    palette_to_game_scale, guard_band_width)
            grayscale_output[mask] = remapped
        elif mapping_strategy == "guard_bands":
            remapped = _map_guard_bands(island_luminosity, gray_start, gray_end, 
                                       palette_to_game_scale, guard_band_width)
            grayscale_output[mask] = remapped
        elif mapping_strategy == "nearest_neighbor_reserve":
            remapped = _map_nearest_neighbor_reserve(island_luminosity, gray_start, gray_end, 
                                                     palette_to_game_scale, guard_band_width)
            grayscale_output[mask] = remapped
        elif mapping_strategy == "alternating_luminosity":
            remapped = _map_alternating_luminosity(island_luminosity, gray_start, gray_end, 
                                                   palette_to_game_scale, guard_band_width, island_index)
            grayscale_output[mask] = remapped
        elif mapping_strategy == "color_clustering":
            remapped_full = _map_color_clustering(rgb_array, mask, gray_start, gray_end, 
                                                   palette_to_game_scale, guard_band_width)
            grayscale_output[mask] = remapped_full[mask]
        elif mapping_strategy == "perceptual":
            remapped_full = _map_perceptual(rgb_array, mask, gray_start, gray_end, 
                                           palette_to_game_scale, guard_band_width)
            grayscale_output[mask] = remapped_full[mask]
        elif mapping_strategy == "reverse_luminosity":
            remapped = _map_reverse_luminosity(island_luminosity, gray_start, gray_end, 
                                               palette_to_game_scale, guard_band_width)
            grayscale_output[mask] = remapped
        else:  # Default: "luminosity"
            remapped = _map_luminosity_default(island_luminosity, gray_start, gray_end, 
                                               palette_to_game_scale, guard_band_width)
            grayscale_output[mask] = remapped

        # Get palette space indices for color mapping
        if mapping_strategy in ["color_clustering", "perceptual"]:
            # For these strategies, remapped values are already in game scale
            island_gray = (grayscale_output[mask] / palette_to_game_scale).astype(np.uint8)
        else:
            # For luminosity-based strategies, convert back to palette space
            island_gray = (grayscale_output[mask] / palette_to_game_scale).astype(np.uint8)

        island_rgb = rgb_array[mask]

        if logger.isEnabledFor(logging.DEBUG):
            unique_colors, unique_counts = np.unique(island_rgb.reshape(-1, 3), axis=0, return_counts=True)
            total_px = int(mask.sum())
            color_stats = [
                {
                    "rgb": unique_colors[idx].tolist(),
                    "count": int(unique_counts[idx]),
                    "percent": round((float(unique_counts[idx]) / float(total_px)) * 100.0, 4)
                    if total_px > 0 else 0.0,
                }
                for idx in np.argsort(unique_counts)[::-1]
            ]
            logger.debug(
                "Island '%s' colors: pixels=%d, unique_colors=%d, full_distribution=%s",
                island_name,
                total_px,
                int(unique_colors.shape[0]),
                color_stats,
            )

            try:
                ys, xs = np.nonzero(mask)
                y0, y1 = ys.min(), ys.max()
                x0, x1 = xs.min(), xs.max()
                sub_mask = mask[y0:y1 + 1, x0:x1 + 1]
                sub_rgba = rgba[y0:y1 + 1, x0:x1 + 1]
                export_rgba = np.zeros_like(sub_rgba)
                export_rgba[sub_mask] = sub_rgba[sub_mask]
                export_img = Image.fromarray(export_rgba, mode='RGBA')
                debug_dir = os.path.join(get_app_root(), "logs", "palette_debug")
                os.makedirs(debug_dir, exist_ok=True)
                safe_name = island_name.replace(os.sep, "_")
                export_path = os.path.join(debug_dir, f"{safe_name}.png")
                export_img.save(export_path)
                logger.debug("Saved debug island PNG: %s", export_path)
            except Exception:
                logger.exception("Failed to export debug PNG for island %s", island_name)

        # Determine actual range of indices (may include guard bands outside island range)
        if island_gray.size > 0:
            actual_min = int(island_gray.min())
            actual_max = int(island_gray.max())
            color_map = {gray_val: [] for gray_val in range(actual_min, actual_max + 1)}
        else:
            color_map = {}
        
        for rgb_val, gray_val in zip(island_rgb, island_gray):
            if gray_val not in color_map:
                color_map[gray_val] = []
            color_map[gray_val].append(rgb_val)

        averaged_colors = {}
        for gray_val, colors in color_map.items():
            if colors:
                averaged_colors[gray_val] = np.mean(colors, axis=0).astype(np.uint8)
            else:
                averaged_colors[gray_val] = None

        island_colors[island_name] = averaged_colors

    all_selected = np.zeros((height, width), dtype=bool)
    for m in masks:
        if m is not None:
            all_selected |= m

    unselected_pixels = non_transparent & ~all_selected

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Unselected non-transparent pixels: %d", int(unselected_pixels.sum())
        )

    if unselected_pixels.any() and islands:
        # Map remaining non-transparent pixels to grayscale for completeness
        # but do NOT add them into any island palette. Palette colors must
        # come strictly from the explicitly selected island masks.
        unselected_luminosity = luminosity[unselected_pixels]
        lum_min = unselected_luminosity.min()
        lum_max = unselected_luminosity.max()

        if lum_max - lum_min < 1:
            lum_max = lum_min + 1

        normalized = (luminosity[unselected_pixels] - lum_min) / (lum_max - lum_min)
        remapped_palette_space = normalized * (palette_size - 1)
        remapped = remapped_palette_space * palette_to_game_scale
        grayscale_output[unselected_pixels] = remapped.astype(np.uint8)

    grayscale_filled = fill_transparent_with_nearest(grayscale_output, non_transparent)

    palette_width = palette_size
    palette = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)

    palette_row = palette[0, :]

    for island_name, gray_start, gray_end in islands:
        colors = island_colors.get(island_name, {})
        for gray_val in range(gray_start, min(gray_end + 1, palette_width)):
            if colors.get(gray_val) is not None:
                palette_row[gray_val] = colors[gray_val]
            else:
                prev_val, next_val = None, None
                for g in range(gray_val - 1, gray_start - 1, -1):
                    if colors.get(g) is not None:
                        prev_val = g
                        break
                for g in range(gray_val + 1, gray_end + 1):
                    if colors.get(g) is not None:
                        next_val = g
                        break

                if prev_val is not None and next_val is not None:
                    t = (gray_val - prev_val) / (next_val - prev_val)
                    color = (1 - t) * colors[prev_val] + t * colors[next_val]
                    palette_row[gray_val] = color.astype(np.uint8)
                elif prev_val is not None:
                    palette_row[gray_val] = colors[prev_val]
                elif next_val is not None:
                    palette_row[gray_val] = colors[next_val]

    # Fill guard bands with appropriate colors based on strategy
    if mapping_strategy in ["guard_bands_quantile", "guard_bands"] and guard_band_width > 0:
        _fill_guard_bands(palette_row, islands, guard_band_width)
    elif mapping_strategy == "nearest_neighbor_reserve":
        _fill_nearest_neighbor_guard_bands(palette_row, islands)

    # Duplicate the palette row to all rows BEFORE smoothing
    # This ensures we have a proper 2D image for smoothing filters
    for theme_row in range(1, palette_height):
        palette[theme_row, :] = palette_row

    # Apply palette smoothing if enabled to reduce harsh color transitions
    # Now smooth the entire palette image (all rows together)
    palette_smooth_method = cfg.get(cfg.ci_palette_smooth_method) if hasattr(cfg, "ci_palette_smooth_method") else "none"
    palette_smooth_strength = float(cfg.get(cfg.ci_palette_smooth_strength)) if hasattr(cfg, "ci_palette_smooth_strength") else 0.0

    if palette_smooth_method != "none" and palette_smooth_strength > 0.0:
        palette = _smooth_palette_image(palette, palette_smooth_method, palette_smooth_strength)

    # Implement palette upscaling if enabled
    upscale_enabled = cfg.get(cfg.ci_palette_upscale_to_256) if hasattr(cfg, "ci_palette_upscale_to_256") else False
    if upscale_enabled and palette_width < 256:
        palette = _upscale_palette_to_256(palette, palette_width)

    palette_img = Image.fromarray(palette, mode='RGB')

    mask_stack_out = np.stack(masks, axis=0) if masks else np.zeros((0, height, width), dtype=bool)
    return grayscale_filled, palette_img, mask_stack_out


def save_islands_npz(image_path: str,
                     islands: list[tuple[str, int, int]],
                     mask_stack: np.ndarray,
                     width: int,
                     height: int) -> str | None:
    """Save islands/masks to the shared npz folder (matches palette_creator auto-save)."""
    if not image_path:
        return None

    try:
        os.makedirs(os.path.join(get_app_root(), "npz"), exist_ok=True)
    except Exception:
        logger.warning("Failed to create npz directory", exc_info=True)
        return None

    try:
        masks = mask_stack.astype(bool, copy=False) if mask_stack is not None else np.zeros((0, height, width), dtype=bool)
        metadata = {
            "version": 1,
            "image_path": image_path,
            "islands": [
                {
                    "name": name,
                    "gray_start": int(gs),
                    "gray_end": int(ge),
                }
                for name, gs, ge in islands
            ],
            "width": int(width),
            "height": int(height),
            "current_island": None,
            "model_path": None,
            "selected_uv_index": 0,
            "quant_method": str(cfg.get(cfg.ci_default_quant_method)) if hasattr(cfg, "ci_default_quant_method") else None,
        }

        base_name, _ = os.path.splitext(os.path.basename(image_path))
        file_path = os.path.join(get_app_root(), "npz", f"{base_name}_palette_state.npz")
        np.savez_compressed(file_path, metadata=json.dumps(metadata), masks=masks)
        return file_path
    except Exception:
        logger.warning("Failed to save islands NPZ", exc_info=True)
        return None
