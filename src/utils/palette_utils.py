import json
import logging
import math
import os

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

SEMI_TRANSPARENT_ALPHA_THRESHOLD = 254


# --- Island Auto-Balancing --------------------------------------------------
def _autobalance_island_ranges(islands: list[tuple[str, int, int]],
                               masks: list[np.ndarray],
                               rgb_array: np.ndarray,
                               palette_size: int) -> list[tuple[str, int, int]]:
    """Shift palette index boundaries so that under-utilized islands receive slots from
    neighboring islands with spare capacity. Runs before quantization.

    Rules:
      - Each island keeps at least 1 slot.
      - Only adjust at shared boundaries with immediate neighbors to preserve order.
      - Do not exceed [0, palette_size-1].
    """
    if not islands or palette_size <= 1:
        return islands

    # Compute unique color counts per island and initial sizes
    island_stats = []
    for (name, g0, g1), mask in zip(islands, masks):
        try:
            size = max(1, int(g1) - int(g0) + 1)
        except Exception:
            size = 1

        unique_colors = 0
        try:
            if mask is not None and mask.any():
                arr = rgb_array[mask]
                if arr.size > 0:
                    unique_colors = int(np.unique(arr.reshape(-1, 3), axis=0).shape[0])
        except Exception:
            unique_colors = 0

        deficit = max(0, unique_colors - size)
        extra = max(0, size - unique_colors)
        island_stats.append({
            'name': name,
            'g0': int(g0),
            'g1': int(g1),
            'size': size,
            'unique': unique_colors,
            'deficit': deficit,
            'extra': extra,
        })

    total_deficit = sum(s['deficit'] for s in island_stats)
    total_extra = sum(s['extra'] for s in island_stats)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Island auto-balance: total_deficit=%d, total_extra=%d", total_deficit, total_extra)

    if total_deficit == 0 or total_extra == 0:
        # Nothing to do
        return [(s['name'], s['g0'], s['g1']) for s in island_stats]

    n = len(island_stats)

    # Helper to ensure constraints after any boundary change
    def clamp_and_fix(idx: int):
        s = island_stats[idx]
        # Clamp boundaries to palette limits
        s['g0'] = max(0, min(s['g0'], palette_size - 1))
        s['g1'] = max(0, min(s['g1'], palette_size - 1))
        if s['g1'] < s['g0']:
            s['g1'] = s['g0']
        s['size'] = s['g1'] - s['g0'] + 1

    # Iteratively satisfy deficits by borrowing from neighbors
    progress = True
    while progress:
        progress = False
        for i in range(n):
            rec = island_stats[i]
            rec_need = rec['deficit']
            if rec_need <= 0:
                continue

            # Try left neighbor
            take_left = 0
            if i - 1 >= 0:
                left = island_stats[i - 1]
                # available to donate from left is its current extra, but also cannot reduce below 1 slot
                left_available = max(0, left['extra'])
                # Additionally, boundary constraint: we can only donate if left has at least 2 slots
                left_available = min(left_available, max(0, (left['g1'] - left['g0']) - 0))
                if left_available > 0 and rec['g0'] > 0 and left['g1'] + 1 == rec['g0']:
                    take_left = min(rec_need, left_available)

            # Try right neighbor
            take_right = 0
            if i + 1 < n:
                right = island_stats[i + 1]
                right_available = max(0, right['extra'])
                right_available = min(right_available, max(0, (right['g1'] - right['g0']) - 0))
                if right_available > 0 and rec['g1'] < palette_size - 1 and rec['g1'] + 1 == right['g0']:
                    take_right = min(rec_need - take_left, right_available)

            if take_left == 0 and take_right == 0:
                # If neighbors have no "extra" by our metric, allow borrowing down to at least 1 slot if they are under-utilized (size>1)
                if i - 1 >= 0 and rec_need > 0:
                    left = island_stats[i - 1]
                    if left['g1'] + 1 == rec['g0'] and (left['g1'] - left['g0'] + 1) > 1:
                        take_left = min(rec_need, (left['g1'] - left['g0']))
                if i + 1 < n and (rec_need - take_left) > 0:
                    right = island_stats[i + 1]
                    if rec['g1'] + 1 == right['g0'] and (right['g1'] - right['g0'] + 1) > 1:
                        take_right = min(rec_need - take_left, (right['g1'] - right['g0']))

            if take_left > 0:
                # Move boundary leftward by take_left
                left = island_stats[i - 1]
                left['g1'] -= take_left
                rec['g0'] -= take_left
                left['extra'] = max(0, left['g1'] - left['g0'] + 1 - left['unique'])
                clamp_and_fix(i - 1)
                clamp_and_fix(i)
                rec_need -= take_left
                progress = True

            if take_right > 0:
                # Move boundary rightward by take_right
                right = island_stats[i + 1]
                right['g0'] += take_right
                rec['g1'] += take_right
                right['extra'] = max(0, right['g1'] - right['g0'] + 1 - right['unique'])
                clamp_and_fix(i + 1)
                clamp_and_fix(i)
                rec_need -= take_right
                progress = True

            # Update receiver deficit after borrowing
            rec['size'] = rec['g1'] - rec['g0'] + 1
            rec['deficit'] = max(0, rec['unique'] - rec['size'])

    # Final log per island
    if logger.isEnabledFor(logging.DEBUG):
        for s in island_stats:
            logger.debug("Island balance: %s -> range [%d, %d], unique=%d, size=%d",
                         s['name'], s['g0'], s['g1'], s['unique'], s['size'])

    return [(s['name'], s['g0'], s['g1']) for s in island_stats]


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


def postprocess_palette_row(palette_row: np.ndarray,
                            islands: list,
                            guard_band_width: int | None = None,
                            smoothing: str | None = None,
                            smoothing_strength: float | None = None) -> np.ndarray:
    """Apply optional guard-band fill and gradient smoothing to a palette row.

    This mirrors creator-time post-processing so rows added later match the
    look of palettes produced by the creator pipeline.

    Args:
        palette_row: Array of shape (W, 3), dtype uint8.
        islands: List of (name, gray_start, gray_end) tuples.
        guard_band_width: Width in indices to blend at island boundaries. If None, uses cfg.ci_guard_band_width.
        smoothing: One of {"none","gaussian","median","bilateral"}. If None, uses cfg.ci_palette_smooth_method.
        smoothing_strength: 0..1 float strength (if None, derived from cfg.ci_palette_smooth_strength as 0..100 -> 0..1).

    Returns:
        A new palette row (W, 3) uint8 after post-processing.
    """
    if palette_row is None or palette_row.size == 0:
        return palette_row

    row = np.array(palette_row, copy=True)

    # Resolve defaults from config if not provided
    if guard_band_width is None:
        try:
            guard_band_width = int(cfg.get(cfg.ci_guard_band_width))
        except Exception:
            guard_band_width = 0

    if smoothing is None:
        try:
            smoothing = cfg.get(cfg.ci_palette_smooth_method)
        except Exception:
            smoothing = "none"

    if smoothing_strength is None:
        try:
            # Config stores 0..100; convert to 0..1
            smoothing_strength = float(cfg.get(cfg.ci_palette_smooth_strength)) / 100.0
        except Exception:
            smoothing_strength = 0.0

    # 1) Guard-band fill at island boundaries
    if guard_band_width and guard_band_width > 0 and islands:
        try:
            _fill_guard_bands(row, islands, guard_band_width)
        except Exception as e:
            logger.warning("Guard-band fill failed: %s", e)

    # 2) Optional palette gradient smoothing
    if smoothing and smoothing.lower() != "none" and smoothing_strength > 0:
        try:
            row = _smooth_palette_gradient(row, method=smoothing.lower(), strength=float(smoothing_strength))
        except Exception as e:
            logger.warning("Palette smoothing failed: %s", e)

    return row


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
        filter_type: "linear" for smooth interpolation, "nearest" for exact color preservation,
                     "anchored_linear" for smooth interpolation anchored to game greys so palette
                     node colors remain exact at their corresponding grey steps.
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

        elif filter_type == "anchored_linear":
            # Build LUT anchored at the exact greys that correspond to palette indices.
            # The game-addressable greys for node k are round(k * 255 / (pw-1)).
            gk = np.rint(np.linspace(0, 255, num=pw)).astype(int)
            lut = np.zeros((256, 3), dtype=np.uint8)
            # Left of first anchor and right of last anchor will be clamped later
            # Set anchor colors exactly
            lut[gk] = palette_row
            # Fill between anchors with linear interpolation in greyscale domain
            for k in range(pw - 1):
                start_g = int(gk[k])
                end_g = int(gk[k + 1])
                if end_g <= start_g:
                    continue
                span = end_g - start_g
                for c in range(3):
                    start_v = int(palette_row[k, c])
                    end_v = int(palette_row[k + 1, c])
                    # Fill [start_g, end_g) so that lut[end_g] stays as anchor for k+1
                    lut[start_g:end_g, c] = np.linspace(start_v, end_v, span, endpoint=False).astype(np.uint8)
            # Clamp ends
            first_g = int(gk[0])
            last_g = int(gk[-1])
            if first_g > 0:
                lut[:first_g, :] = palette_row[0]
            if last_g < 255:
                lut[last_g+1:, :] = palette_row[-1]

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

    if semi_mode != "none":
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

    # We will allocate grayscale capacity AFTER grouping, proportionally to
    # each group's unique color diversity, using standard step sizes (multiples of 8).

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

    # Determine proportional island capacities (multiples of 8) based on unique color counts
    step = 8
    if palette_size < step:
        raise ValueError("Palette size must be at least 8.")

    # Total units in steps of 8
    total_units = palette_size // step
    if total_units * step != palette_size:
        logger.warning("Palette size %d is not a multiple of 8; truncating to %d.", palette_size, total_units * step)

    # Limit to desired number of islands; pad with empty groups if needed (already done below)
    # Compute weights from unique color counts
    uniq_counts = [max(0, len(g["unique_colors"])) for g in groups]
    sum_w = sum(uniq_counts)
    # If there is no color diversity at all, fallback to pixel totals
    if sum_w == 0:
        uniq_counts = [max(0, int(g.get("pixel_total", 0))) for g in groups]
        sum_w = sum(uniq_counts)
    # If still zero (all empty), distribute evenly
    if sum_w == 0:
        uniq_counts = [1 for _ in groups]
        sum_w = len(uniq_counts)

    # Initial floor allocation in units, with minimum 1 unit (i.e., 8) per island that has any weight
    prelim_units = []
    fracs = []
    for w in uniq_counts:
        prop = (w / sum_w) * total_units if sum_w > 0 else 0.0
        units_floor = int(math.floor(prop))
        prelim_units.append(units_floor)
        fracs.append(prop - units_floor)

    used_units = sum(prelim_units)
    # Ensure at least 1 unit for islands that have non-zero weight
    for i, w in enumerate(uniq_counts):
        if w > 0 and prelim_units[i] == 0:
            prelim_units[i] = 1
            used_units += 1

    # Adjust to exactly total_units by adding/removing units based on fractional parts
    def add_units(k: int):
        nonlocal used_units
        # Give units to the islands with largest fractional part first
        order = sorted(range(len(prelim_units)), key=lambda i: fracs[i], reverse=True)
        idx = 0
        while k > 0 and used_units < total_units and idx < len(order):
            prelim_units[order[idx]] += 1
            used_units += 1
            k -= 1
            idx += 1

    def remove_units(k: int):
        nonlocal used_units
        # Remove units from islands with smallest fractional part first, but keep at least 1 if they had weight
        order = sorted(range(len(prelim_units)), key=lambda i: fracs[i])
        idx = 0
        while k > 0 and used_units > total_units and idx < len(order):
            i = order[idx]
            min_allowed = 1 if uniq_counts[i] > 0 else 0
            if prelim_units[i] > min_allowed:
                prelim_units[i] -= 1
                used_units -= 1
                k -= 1
            idx += 1

    if used_units < total_units:
        add_units(total_units - used_units)
    elif used_units > total_units:
        remove_units(used_units - total_units)

    # Build island specs in image order (contiguous grayscale ranges)
    island_specs = []
    current_start = 0
    for units in prelim_units:
        size = max(0, units * step)
        if size == 0:
            # Create an empty slice when no capacity; still maintain contiguous ranges
            gray_start = current_start
            gray_end = current_start - 1  # empty
        else:
            gray_start = current_start
            gray_end = current_start + size - 1
        island_specs.append({
            "gray_start": gray_start,
            "gray_end": gray_end,
            "capacity": size
        })
        current_start += size

    if not island_specs or current_start != total_units * step:
        # Safety: ensure specs cover exactly the truncated palette size
        raise ValueError("Failed to allocate proportional island capacities.")

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
    return np.rint(remapped_palette_space * palette_to_game_scale).astype(np.uint8)


def _map_guard_bands_quantile(luminosity: np.ndarray, gray_start: int, gray_end: int,
                               palette_to_game_scale: float, guard_band_width: int = 1) -> np.ndarray:
    """Hybrid: Guard bands + quantile distribution (recommended).
    
    This function preserves all unique luminosity values by mapping them evenly
    across the effective palette range, avoiding color loss.
    """
    effective_start = gray_start + guard_band_width
    effective_end = gray_end - guard_band_width
    effective_range = max(1, effective_end - effective_start + 1)
    
    # Get unique luminosity values and their inverse mapping
    unique_lum, inverse_indices = np.unique(luminosity, return_inverse=True)
    num_unique = len(unique_lum)
    
    if num_unique == 0:
        return np.zeros_like(luminosity, dtype=np.uint8)
    
    # Map unique luminosity values evenly across the effective range
    if num_unique == 1:
        # Single color: map to middle of effective range
        unique_palette_indices = np.array([effective_start + effective_range // 2])
    else:
        # Multiple colors: spread evenly across effective range
        unique_palette_indices = np.linspace(effective_start, effective_end, num_unique)
    
    # Map all pixels using inverse indices
    remapped_palette_space = unique_palette_indices[inverse_indices]
    
    return np.rint(remapped_palette_space * palette_to_game_scale).astype(np.uint8)


def _map_quantile(luminosity: np.ndarray, gray_start: int, gray_end: int,
                  palette_to_game_scale: float, guard_band_width: int = 0) -> np.ndarray:
    """Quantile-based distribution without guard bands.
    
    This function preserves all unique luminosity values by mapping them evenly
    across the palette range, avoiding color loss.
    """
    # Get unique luminosity values and their inverse mapping
    unique_lum, inverse_indices = np.unique(luminosity, return_inverse=True)
    num_unique = len(unique_lum)
    
    if num_unique == 0:
        return np.zeros_like(luminosity, dtype=np.uint8)
    
    # Map unique luminosity values evenly across the palette range
    if num_unique == 1:
        # Single color: map to middle of range
        unique_palette_indices = np.array([gray_start + (gray_end - gray_start) // 2])
    else:
        # Multiple colors: spread evenly across full range
        unique_palette_indices = np.linspace(gray_start, gray_end, num_unique)
    
    # Map all pixels using inverse indices
    remapped_palette_space = unique_palette_indices[inverse_indices]
    
    return np.rint(remapped_palette_space * palette_to_game_scale).astype(np.uint8)


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
    return np.rint(remapped_palette_space * palette_to_game_scale).astype(np.uint8)


def _map_smoothed_quantile(luminosity: np.ndarray, gray_start: int, gray_end: int,
                           palette_to_game_scale: float,
                           guard_band_width: int = 1,
                           bins: int = 256,
                           sigma: float = 1.5,
                           alpha: float = 0.3) -> np.ndarray:
    """Smoothed-quantile mapping via blurred histogram ECDF, optionally blended toward linear.

    Args:
        luminosity: 1D array of luminance values for island pixels (uint8 or float)
        gray_start, gray_end: island palette-space bounds (inclusive)
        palette_to_game_scale: scale factor from palette-space index to 0..255 space
        guard_band_width: number of palette indices reserved at each edge
        bins: histogram bins for ECDF
        sigma: Gaussian sigma (in bins) to smooth histogram
        alpha: blend toward linear mapping in [0,1]
    """
    eff_start = gray_start + max(0, int(guard_band_width))
    eff_end = gray_end - max(0, int(guard_band_width))
    rng = max(1, eff_end - eff_start)

    L = luminosity.astype(np.float32)
    Lmin = float(L.min())
    Lmax = float(L.max())
    if not np.isfinite(Lmin) or not np.isfinite(Lmax):
        return np.zeros_like(luminosity, dtype=np.uint8)
    if Lmax - Lmin < 1.0:
        Lmax = Lmin + 1.0
    z = (L - Lmin) / (Lmax - Lmin)
    z = np.clip(z, 0.0, 1.0)

    bins = int(max(16, bins))
    try:
        hist, edges = np.histogram(z, bins=bins, range=(0.0, 1.0), density=False)
    except Exception:
        return np.rint((np.full_like(luminosity, eff_start + rng // 2, dtype=np.float32) * palette_to_game_scale)).astype(np.uint8)

    # Smooth histogram and build CDF
    sigma = float(max(0.0, sigma))
    if sigma > 0.0:
        hist = ndimage.gaussian_filter1d(hist.astype(np.float32), sigma=sigma, mode="nearest")
    else:
        hist = hist.astype(np.float32)
    total = float(hist.sum())
    if total <= 0.0:
        g = np.full_like(z, eff_start + rng * 0.5, dtype=np.float32)
        return (g * palette_to_game_scale).astype(np.uint8)
    cdf = np.cumsum(hist)
    cdf /= (cdf[-1] + 1e-8)
    centers = 0.5 * (edges[:-1] + edges[1:])
    Fz = np.interp(z, centers, cdf, left=0.0, right=1.0).astype(np.float32)

    g_quant = eff_start + Fz * rng

    # Linear mapping component for tempering
    g_lin = eff_start + z * rng

    a = float(np.clip(alpha, 0.0, 1.0))
    g = (1.0 - a) * g_quant + a * g_lin
    return np.rint(g * palette_to_game_scale).astype(np.uint8)


def _map_tempered_quantile(luminosity: np.ndarray, gray_start: int, gray_end: int,
                           palette_to_game_scale: float,
                           guard_band_width: int = 0,
                           alpha: float = 0.3) -> np.ndarray:
    """Blend quantile (optionally with guard bands) with linear luminosity mapping.

    alpha=0 → pure quantile; alpha=1 → pure linear.
    """
    # Quantile part
    if guard_band_width and guard_band_width > 0:
        gq = _map_guard_bands_quantile(luminosity, gray_start, gray_end, palette_to_game_scale, guard_band_width)
        gl = _map_guard_bands(luminosity, gray_start, gray_end, palette_to_game_scale, guard_band_width)
    else:
        gq = _map_quantile(luminosity, gray_start, gray_end, palette_to_game_scale, 0)
        gl = _map_luminosity_default(luminosity, gray_start, gray_end, palette_to_game_scale, 0)

    a = float(np.clip(alpha, 0.0, 1.0))
    # Work in scaled (0..255) space directly
    g = (1.0 - a) * gq.astype(np.float32) + a * gl.astype(np.float32)
    return np.rint(g).astype(np.uint8)


def _map_spline_quantile(luminosity: np.ndarray, gray_start: int, gray_end: int,
                         palette_to_game_scale: float,
                         guard_band_width: int = 1,
                         profile: str = "even",
                         gamma: float = 1.0) -> np.ndarray:
    """Monotone spline mapping using data quantile anchors to avoid sharp steps.

    We compute anchors at fixed quantiles qs of the normalized luminance distribution.
    The x-anchors are the data quantiles xp = Q_z(qs); y-anchors are evenly spaced
    (optionally shape-adjusted) indices within the effective island range. We then
    use a monotone PCHIP interpolator to map z → y smoothly.
    """
    eff_start = gray_start + max(0, int(guard_band_width))
    eff_end = gray_end - max(0, int(guard_band_width))
    rng = max(1, eff_end - eff_start)

    L = luminosity.astype(np.float32)
    Lmin = float(np.min(L))
    Lmax = float(np.max(L))
    if Lmax - Lmin < 1.0:
        Lmax = Lmin + 1.0
    z = (L - Lmin) / (Lmax - Lmin)
    z = np.clip(z, 0.0, 1.0)

    # Anchor quantiles
    qs = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], dtype=np.float32)
    try:
        xp = np.quantile(z, qs)
    except Exception:
        xp = qs.copy()

    # Ensure strictly increasing x for PCHIP
    # If duplicates occur, slightly jitter them forward by tiny eps
    eps = 1e-4
    for i in range(1, len(xp)):
        if xp[i] <= xp[i-1]:
            xp[i] = min(1.0, xp[i-1] + eps)

    # Shape y anchors according to profile
    q_out = qs.copy()
    gamma = float(max(1e-3, gamma))
    if profile == "compressed_ends":
        # push toward center: use power >1
        q_out = np.power(q_out, gamma)
    elif profile == "expanded_ends":
        # expand ends: use root (power <1)
        q_out = np.power(q_out, 1.0 / gamma)

    y = eff_start + q_out * rng

    try:
        pchip = interpolate.PchipInterpolator(xp, y, extrapolate=True)
        g = pchip(z.astype(np.float32)).astype(np.float32)
    except Exception:
        # Fallback to linear if PCHIP fails
        g = eff_start + z * rng

    return np.rint(g * palette_to_game_scale).astype(np.uint8)


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
                     guard_band_width: int,
                     anchor_mask: np.ndarray | None = None) -> None:
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
                if curr_end < len(palette_row) and not (anchor_mask is not None and anchor_mask[curr_end]):
                    palette_row[curr_end] = (0.67 * curr_color + 0.33 * next_color).astype(np.uint8)
                if next_start < len(palette_row) and not (anchor_mask is not None and anchor_mask[next_start]):
                    palette_row[next_start] = (0.33 * curr_color + 0.67 * next_color).astype(np.uint8)


def _fill_nearest_neighbor_guard_bands(palette_row: np.ndarray, islands: list[tuple[str, int, int]],
                                       anchor_mask: np.ndarray | None = None) -> None:
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
                if not (anchor_mask is not None and anchor_mask[gray_start]):
                    palette_row[gray_start] = palette_row[gray_start + 1]
            
            # Last index copies from previous index
            if gray_end < len(palette_row) and gray_end - 1 >= 0:
                if not (anchor_mask is not None and anchor_mask[gray_end]):
                    palette_row[gray_end] = palette_row[gray_end - 1]


def _rgb_to_lab_cv(rgb_list: np.ndarray) -> np.ndarray:
    """Convert an array of RGB uint8 colors (N,3) to Lab uint8 using OpenCV space.
    Returns array (N,3) uint8.
    """
    if rgb_list.size == 0:
        return rgb_list
    rgb_reshaped = rgb_list.reshape(-1, 1, 3)
    lab = cv2.cvtColor(rgb_reshaped, cv2.COLOR_RGB2LAB)
    return lab.reshape(-1, 3)


def _lab_to_rgb_cv(lab_color: np.ndarray) -> np.ndarray:
    """Convert a single Lab uint8 color (3,) to RGB uint8 using OpenCV space."""
    lab = lab_color.reshape(1, 1, 3)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb.reshape(3)


def _delta_e_simple(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    """Simple Euclidean distance in OpenCV Lab space."""
    diff = lab_a.astype(np.int16) - lab_b.astype(np.int16)
    return float(np.sqrt(np.sum(diff * diff)))


def _apply_tone_curve_np(x01: np.ndarray, s: float, gamma: float, b: float, w: float) -> np.ndarray:
    """Apply a simple monotone tone curve to normalized grayscale x in [0,1].
    Parameters are small perturbations to keep appearance natural.
    b and w are black/white point shifts in normalized units.
    """
    # Protect denominator
    denom = max(1e-6, 1.0 - b - w)
    x1 = np.clip((x01 - b) / denom, 0.0, 1.0)
    # Gamma
    x2 = np.power(np.clip(x1, 1e-6, 1.0), gamma)
    # Contrast around mid-gray
    x3 = 0.5 + s * (x2 - 0.5)
    return np.clip(x3, 0.0, 1.0)


def _map_by_strategy_for_scoring(strategy_name: str,
                                 island_lum: np.ndarray,
                                 gray_start: int,
                                 gray_end: int,
                                 palette_to_game_scale: float,
                                 guard_band_width: int,
                                 rgb_array: np.ndarray | None,
                                 mask: np.ndarray | None,
                                 island_index: int) -> np.ndarray:
    """Helper to reuse existing mapping functions for scoring."""
    if strategy_name == "guard_bands_quantile":
        return _map_guard_bands_quantile(island_lum, gray_start, gray_end, palette_to_game_scale, guard_band_width)
    elif strategy_name == "quantile":
        return _map_quantile(island_lum, gray_start, gray_end, palette_to_game_scale, guard_band_width)
    elif strategy_name == "guard_bands":
        return _map_guard_bands(island_lum, gray_start, gray_end, palette_to_game_scale, guard_band_width)
    elif strategy_name == "nearest_neighbor_reserve":
        return _map_nearest_neighbor_reserve(island_lum, gray_start, gray_end, palette_to_game_scale, guard_band_width)
    elif strategy_name == "alternating_luminosity":
        return _map_alternating_luminosity(island_lum, gray_start, gray_end, palette_to_game_scale, guard_band_width, island_index)
    elif strategy_name == "color_clustering":
        # For scoring, fall back to luminosity default to avoid heavy/full-image ops
        return _map_luminosity_default(island_lum, gray_start, gray_end, palette_to_game_scale, guard_band_width)
    elif strategy_name == "perceptual":
        # For scoring, fall back to luminosity default to avoid heavy/full-image ops
        return _map_luminosity_default(island_lum, gray_start, gray_end, palette_to_game_scale, guard_band_width)
    elif strategy_name == "reverse_luminosity":
        return _map_reverse_luminosity(island_lum, gray_start, gray_end, palette_to_game_scale, guard_band_width)
    elif strategy_name == "smoothed_quantile":
        bins = int(cfg.get(cfg.ci_smoothed_quantile_bins)) if hasattr(cfg, "ci_smoothed_quantile_bins") else 256
        sigma = float(cfg.get(cfg.ci_smoothed_quantile_sigma)) if hasattr(cfg, "ci_smoothed_quantile_sigma") else 1.5
        alpha_pct = int(cfg.get(cfg.ci_smoothed_quantile_alpha)) if hasattr(cfg, "ci_smoothed_quantile_alpha") else 30
        alpha_blend = max(0.0, min(1.0, alpha_pct / 100.0))
        return _map_smoothed_quantile(island_lum, gray_start, gray_end, palette_to_game_scale, guard_band_width, bins, sigma, alpha_blend)
    elif strategy_name == "tempered_quantile":
        alpha_pct = int(cfg.get(cfg.ci_tempered_quantile_alpha)) if hasattr(cfg, "ci_tempered_quantile_alpha") else 30
        alpha_blend = max(0.0, min(1.0, alpha_pct / 100.0))
        return _map_tempered_quantile(island_lum, gray_start, gray_end, palette_to_game_scale, guard_band_width, alpha_blend)
    elif strategy_name == "spline_quantile":
        profile = cfg.get(cfg.ci_spline_profile) if hasattr(cfg, "ci_spline_profile") else "even"
        gamma = float(cfg.get(cfg.ci_spline_gamma)) if hasattr(cfg, "ci_spline_gamma") else 1.0
        return _map_spline_quantile(island_lum, gray_start, gray_end, palette_to_game_scale, guard_band_width, profile, gamma)
    else:
        return _map_luminosity_default(island_lum, gray_start, gray_end, palette_to_game_scale, guard_band_width)


def _score_mapping_histograms(histograms: list[np.ndarray]) -> float:
    """Compute a collision score: penalize overfull bins and unused bins.
    Lower is better.
    """
    score = 0.0
    for hist in histograms:
        n_pix = int(hist.sum())
        n_bins = len(hist)
        if n_bins <= 0:
            continue
        if n_pix == 0:
            # no pixels in this island region
            continue
        target = n_pix / max(1, n_bins)
        over = np.maximum(0.0, hist - target)
        score += float((over ** 1.7).sum())
        score += 0.1 * float((hist == 0).sum())
    return score


def resolve_grayscale_collisions(luminosity_u8: np.ndarray,
                                 islands: list[tuple[str, int, int]],
                                 masks: list[np.ndarray],
                                 rgb_array: np.ndarray,
                                 mapping_strategy: str,
                                 guard_band_width: int,
                                 palette_to_game_scale: float) -> np.ndarray:
    """Randomized search over tone-curve parameters to reduce grayscale collisions.

    Returns an adjusted luminosity (uint8) to be used by existing mapping code.
    Applies only global adjustments and optional tiny per-island tweaks (monotone),
    without bespoke per-pixel arbitrary edits.
    """
    tries = int(cfg.get(cfg.ci_collision_resolver_tries)) if hasattr(cfg, "ci_collision_resolver_tries") else 15
    per_island = bool(cfg.get(cfg.ci_collision_resolver_per_island)) if hasattr(cfg, "ci_collision_resolver_per_island") else True
    resolver_mode = cfg.get(cfg.ci_collision_resolver_strategy) if hasattr(cfg, "ci_collision_resolver_strategy") else "gray_curve"
    nat_w = float(cfg.get(cfg.ci_collision_resolver_naturalness_w)) if hasattr(cfg, "ci_collision_resolver_naturalness_w") else 0.10
    coll_w = float(cfg.get(cfg.ci_collision_resolver_collision_w)) if hasattr(cfg, "ci_collision_resolver_collision_w") else 1.0
    if not np.isfinite(nat_w):
        nat_w = 0.10
    nat_w = float(max(0.0, min(1.0, nat_w)))
    if not np.isfinite(coll_w):
        coll_w = 1.0
    coll_w = float(max(0.0, min(10.0, coll_w)))

    # Baseline indices for naturalness
    base_histograms: list[np.ndarray] = []
    base_indices_concat = []
    # Build baseline mapping using current luminosity and masks
    # Also prepare downsample mask for speed
    subsample = 2
    h, w = luminosity_u8.shape[:2]
    ys = slice(0, h, subsample)
    xs = slice(0, w, subsample)
    lum_ds = luminosity_u8[ys, xs].astype(np.float32)
    masks_ds = [m[ys, xs] if m is not None else None for m in masks]

    def map_and_hist(lum_img_u8: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        indices_maps = []
        histograms = []
        for i, (isl, m) in enumerate(zip(islands, masks_ds)):
            island_name, gray_start, gray_end = isl
            if m is None or not np.any(m):
                histograms.append(np.zeros(max(0, gray_end - gray_start + 1), dtype=np.int64))
                continue
            island_lum = lum_img_u8[m].astype(np.float32)
            # Map using selected strategy
            remapped = _map_by_strategy_for_scoring(mapping_strategy,
                                                    island_lum,
                                                    gray_start, gray_end,
                                                    palette_to_game_scale,
                                                    guard_band_width,
                                                    rgb_array,  # can be None for non-RGB strategies
                                                    m,
                                                    i)
            # Convert to palette space indices (integer bins within island range)
            island_gray = np.rint(remapped / palette_to_game_scale).astype(np.int32)
            # Clamp to island range
            island_gray = np.clip(island_gray, gray_start, gray_end)
            # Histogram within island
            bins = max(0, gray_end - gray_start + 1)
            if bins == 0:
                hist = np.zeros(0, dtype=np.int64)
            else:
                hist = np.bincount(island_gray - gray_start, minlength=bins)
            histograms.append(hist.astype(np.float64))
            indices_maps.append(island_gray)
        # Concatenate for naturalness measurement
        if indices_maps:
            concat = np.concatenate([arr.ravel() for arr in indices_maps if arr.size > 0])
        else:
            concat = np.zeros((0,), dtype=np.int32)
        return histograms, concat

    base_histograms, base_indices_concat = map_and_hist(lum_ds)
    base_hist_score_raw = _score_mapping_histograms(base_histograms)
    base_total_pixels = sum(int(h.sum()) for h in base_histograms)
    base_hist_score = base_hist_score_raw / max(1, base_total_pixels)

    # Helper: count collisions across all islands
    def _count_collisions(histos: list[np.ndarray]) -> int:
        total = 0.0
        for h in histos:
            n_pix = float(h.sum())
            n_bins = len(h)
            if n_pix <= 0 or n_bins <= 0:
                continue
            # Ideal uniform occupancy per bin; collisions are the excess above this ideal.
            ideal = n_pix / float(max(1, n_bins))
            over = h.astype(np.float64) - ideal
            over = over[over > 0]
            if over.size:
                total += float(over.sum())
        return int(round(total))

    base_collisions = _count_collisions(base_histograms)
    base_score = base_hist_score + coll_w * float(base_collisions)

    if logger.isEnabledFor(logging.DEBUG):
        try:
            logger.debug(
                "Collision resolver: base score=%.3f, base collisions=%d, strategy=%s, tries=%d, per_island=%s, resolver_mode=%s, naturalness_w=%.3f, collision_w=%.3f",
                base_score,
                base_collisions,
                mapping_strategy,
                int(cfg.get(cfg.ci_collision_resolver_tries)) if hasattr(cfg, "ci_collision_resolver_tries") else 15,
                str(bool(cfg.get(cfg.ci_collision_resolver_per_island)) if hasattr(cfg, "ci_collision_resolver_per_island") else True),
                resolver_mode,
                nat_w,
                coll_w
            )
        except Exception:
            # Avoid breaking pipeline due to logging issues
            pass

    best_score = base_score
    best_params = None
    best_lum = luminosity_u8
    best_collisions = base_collisions
    best_total_from_tie = False

    rng = np.random.default_rng()

    x01_full = luminosity_u8.astype(np.float32) / 255.0
    rgb01 = None
    try:
        if rgb_array is not None and rgb_array.ndim >= 2:
            rgb01 = rgb_array.astype(np.float32) / 255.0
    except Exception:
        rgb01 = None

    for t in range(max(1, tries)):
        # Select which strategy to try this iteration
        mode = resolver_mode
        if resolver_mode == "hybrid":
            mode = rng.choice(["gray_curve", "per_channel_gamma", "rgb_weight_mix"])  # type: ignore[arg-type]

        params_desc = ""
        cand_u8: np.ndarray

        if mode == "per_channel_gamma" and rgb01 is not None:
            # Adjust each RGB channel slightly, recompute luminance.
            gR = float(rng.uniform(0.92, 1.08))
            gG = float(rng.uniform(0.92, 1.08))
            gB = float(rng.uniform(0.92, 1.08))
            kR = float(rng.uniform(0.97, 1.03))
            kG = float(rng.uniform(0.97, 1.03))
            kB = float(rng.uniform(0.97, 1.03))
            oR = float(rng.uniform(-0.01, 0.01))
            oG = float(rng.uniform(-0.01, 0.01))
            oB = float(rng.uniform(-0.01, 0.01))

            R = np.clip(kR * np.power(np.clip(rgb01[:, :, 0], 1e-6, 1.0), gR) + oR, 0.0, 1.0)
            G = np.clip(kG * np.power(np.clip(rgb01[:, :, 1], 1e-6, 1.0), gG) + oG, 0.0, 1.0)
            B = np.clip(kB * np.power(np.clip(rgb01[:, :, 2], 1e-6, 1.0), gB) + oB, 0.0, 1.0)
            # Use fixed Rec.601-ish weights (match existing 0.299/0.587/0.114 mapping)
            x_adj = 0.299 * R + 0.587 * G + 0.114 * B
            cand_u8 = np.clip(np.rint(x_adj * 255.0), 0, 255).astype(np.uint8)
            params_desc = f"mode=per_channel_gamma gR={gR:.4f} gG={gG:.4f} gB={gB:.4f} kR={kR:.4f} kG={kG:.4f} kB={kB:.4f} oR={oR:.4f} oG={oG:.4f} oB={oB:.4f}"

        elif mode == "rgb_weight_mix" and rgb01 is not None:
            # Slightly vary the luminance weights around standard values; keep non-negative and sum=1
            base_w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
            delta = rng.normal(0.0, 0.03, size=3).astype(np.float32)  # small perturbation
            w_raw = np.clip(base_w + delta, 0.0, None)
            if float(w_raw.sum()) <= 1e-6:
                w = base_w
            else:
                w = w_raw / float(w_raw.sum())
            x_adj = w[0] * rgb01[:, :, 0] + w[1] * rgb01[:, :, 1] + w[2] * rgb01[:, :, 2]
            cand_u8 = np.clip(np.rint(x_adj * 255.0), 0, 255).astype(np.uint8)
            params_desc = f"mode=rgb_weight_mix wR={w[0]:.4f} wG={w[1]:.4f} wB={w[2]:.4f}"

        else:
            # Default gray curve mode (existing behavior)
            s = float(rng.uniform(0.95, 1.05))
            gamma = float(rng.uniform(0.90, 1.12))
            b = float(rng.uniform(-0.03, 0.03))
            wshift = float(rng.uniform(-0.03, 0.03))
            if b + wshift > 0.06:
                scale = 0.06 / (b + wshift + 1e-6)
                b *= scale
                wshift *= scale

            x_adj = _apply_tone_curve_np(x01_full, s, gamma, b, wshift)

            if per_island and masks:
                # Tiny per-island tweak within [0,1]
                x_per = x_adj.copy()
                for i, (isl, m) in enumerate(zip(islands, masks)):
                    if m is None or not np.any(m):
                        continue
                    gamma_i = float(rng.uniform(0.97, 1.05))
                    delta_i = float(rng.uniform(-0.015, 0.015))
                    vals = x_per[m]
                    vals = np.power(np.clip(vals, 1e-6, 1.0), gamma_i)
                    vals = np.clip(vals + delta_i, 0.0, 1.0)
                    # Project to island allowed grey range in normalized units
                    lo = (isl[1] / 255.0)
                    hi = (isl[2] / 255.0)
                    vals = np.clip(vals, lo, hi)
                    x_per[m] = vals
                x_adj = x_per

            cand_u8 = np.clip(np.rint(x_adj * 255.0), 0, 255).astype(np.uint8)
            params_desc = f"mode=gray_curve s={s:.4f} gamma={gamma:.4f} b={b:.4f} w={wshift:.4f}"

        # Score on downsampled grid
        cand_ds = cand_u8[ys, xs]
        histos, idx_concat = map_and_hist(cand_ds)
        hist_score_raw = _score_mapping_histograms(histos)
        hist_total_pixels = sum(int(h.sum()) for h in histos)
        hist_score = hist_score_raw / max(1, hist_total_pixels)
        cand_collisions = _count_collisions(histos)
        # Naturalness: penalty for deviating from baseline indices (configurable weight)
        nat_pen = 0.0
        if idx_concat.size == base_indices_concat.size and idx_concat.size > 0:
            nat_pen = nat_w * float(np.mean((idx_concat.astype(np.float32) - base_indices_concat.astype(np.float32)) ** 2))
        total_score = hist_score + nat_pen + coll_w * float(cand_collisions)

        improved = False
        tie_break = False
        eps = 1e-6
        if total_score + eps < best_score:
            improved = True
        elif abs(total_score - best_score) <= eps and cand_collisions < best_collisions:
            improved = True
            tie_break = True

        if improved:
            prev_best_score = best_score
            prev_best_collisions = best_collisions
            best_score = total_score
            best_lum = cand_u8
            best_params = params_desc
            # Track collisions for tie-breaking in later iterations
            best_collisions = cand_collisions
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    if tie_break:
                        logger.debug(
                            "Collision resolver: try %d improved (tie-break by collisions) -> score %.3f, collisions %d (prev best %.3f, %d); %s",
                            t + 1,
                            best_score,
                            cand_collisions,
                            prev_best_score,
                            prev_best_collisions,
                            params_desc
                        )
                    else:
                        logger.debug(
                            "Collision resolver: try %d improved -> score %.3f, collisions %d (prev best %.3f, %d); %s",
                            t + 1,
                            best_score,
                            cand_collisions,
                            prev_best_score,
                            prev_best_collisions,
                            params_desc
                        )
                except Exception:
                    pass

    # Deterministic rescue step: per-island monotone equalization to spread values
    # across available bins, often sharply reducing collisions when stochastic
    # search fails to meaningfully improve them.
    try:
        if masks and len(masks) == len(islands):
            eq_float = best_lum.astype(np.float32) / 255.0
            for (isl, m) in zip(islands, masks):
                if m is None or not np.any(m):
                    continue
                vals = eq_float[m]
                if vals.size < 2:
                    continue
                lo = float(isl[1]) / 255.0
                hi = float(isl[2]) / 255.0
                vals = np.clip(vals, lo, hi)
                order = np.argsort(vals)
                target = np.linspace(lo, hi, vals.size, endpoint=True, dtype=np.float32)
                remapped = np.empty_like(vals)
                remapped[order] = target
                eq_float[m] = remapped
            eq_u8 = np.clip(np.rint(eq_float * 255.0), 0, 255).astype(np.uint8)

            # Score equalized candidate on the downsampled grid
            eq_ds = eq_u8[ys, xs]
            histos_eq, idx_concat_eq = map_and_hist(eq_ds)
            hist_score_raw_eq = _score_mapping_histograms(histos_eq)
            hist_total_eq = sum(int(h.sum()) for h in histos_eq)
            hist_score_eq = hist_score_raw_eq / max(1, hist_total_eq)
            cand_collisions_eq = _count_collisions(histos_eq)
            nat_pen_eq = 0.0
            if idx_concat_eq.size == base_indices_concat.size and idx_concat_eq.size > 0:
                nat_pen_eq = nat_w * float(np.mean((idx_concat_eq.astype(np.float32) - base_indices_concat.astype(np.float32)) ** 2))
            total_score_eq = hist_score_eq + nat_pen_eq + coll_w * float(cand_collisions_eq)

            eps = 1e-6
            if total_score_eq + eps < best_score or (abs(total_score_eq - best_score) <= eps and cand_collisions_eq < best_collisions):
                if logger.isEnabledFor(logging.DEBUG):
                    try:
                        logger.debug(
                            "Collision resolver: deterministic equalization improved -> score %.3f, collisions %d (prev %.3f, %d)",
                            total_score_eq,
                            cand_collisions_eq,
                            best_score,
                            best_collisions,
                        )
                    except Exception:
                        pass
                best_score = total_score_eq
                best_lum = eq_u8
                best_collisions = cand_collisions_eq
                best_params = "deterministic_equalization"
    except Exception:
        # Non-fatal; continue with best stochastic result
        pass

    if logger.isEnabledFor(logging.DEBUG):
        try:
            applied = best_score < base_score
            if best_params is None:
                logger.debug(
                    "Collision resolver: no improvement over base. Final score=%.3f, collisions=%d",
                    base_score, base_collisions
                )
            else:
                logger.debug(
                    "Collision resolver: best score=%.3f, collisions=%d; %s; applied=%s (nat_w=%.3f, coll_w=%.3f)",
                    best_score, best_collisions, str(best_params), str(applied), nat_w, coll_w
                )
        except Exception:
            pass

    # Ensure final grayscale respects island allowed gray ranges if masks provided
    try:
        if masks and len(masks) == len(islands):
            out = best_lum.copy()
            for (isl, m) in zip(islands, masks):
                if m is None or not np.any(m):
                    continue
                gmin = int(isl[1])
                gmax = int(isl[2])
                # clamp only pixels inside this island's mask
                vals = out[m]
                vals = np.clip(vals, gmin, gmax)
                out[m] = vals
            best_lum = out
    except Exception:
        # Non-fatal; mapping stage will still clamp indices
        pass

    return best_lum


def build_grayscale_and_palette_from_islands(rgba: np.ndarray,
                                             islands: list[tuple[str, int, int]],
                                             mask_stack: np.ndarray,
                                             palette_size: int,
                                             palette_height: int = 16) -> tuple[np.ndarray, Image.Image, np.ndarray, list[tuple[str, int, int]]]:
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

    # Optional auto-balance of island ranges BEFORE any quantization/mapping
    active_islands = islands
    try:
        auto_balance_enabled = cfg.get(cfg.ci_island_autobalance_enable) if hasattr(cfg, "ci_island_autobalance_enable") else False
    except Exception:
        auto_balance_enabled = False

    if auto_balance_enabled and len(islands) > 0:
        try:
            balanced = _autobalance_island_ranges(islands, masks, rgb_array, palette_size)
            # Log differences
            if logger.isEnabledFor(logging.DEBUG):
                for (oname, og0, og1), (nname, ng0, ng1) in zip(islands, balanced):
                    if og0 != ng0 or og1 != ng1:
                        logger.debug("Auto-balance: %s range [%d,%d] -> [%d,%d]", oname, og0, og1, ng0, ng1)
            active_islands = balanced
        except Exception:
            logger.exception("Island auto-balance failed; proceeding with original ranges")

    island_colors = {}

    # Optional island pre-quantization flag
    prequant_enabled = cfg.get(cfg.ci_island_prequant_enable) if hasattr(cfg, "ci_island_prequant_enable") else False
    default_quant_method = cfg.get(cfg.ci_default_quant_method) if hasattr(cfg, "ci_default_quant_method") else QuantAlgorithm.libimagequant

    # Optional: attempt to resolve grayscale collisions via global tone curve adjustments
    try:
        resolver_enabled = cfg.get(cfg.ci_enable_collision_resolver) if hasattr(cfg, "ci_enable_collision_resolver") else False
    except Exception:
        resolver_enabled = False

    if resolver_enabled and len(active_islands) > 0:
        try:
            # Provide uint8 luminosity into resolver; it returns adjusted uint8 luminosity
            lum_u8 = np.clip(np.rint(luminosity).astype(np.uint8), 0, 255)
            lum_adj = resolve_grayscale_collisions(
                lum_u8,
                active_islands,
                masks,
                rgb_array,
                mapping_strategy,
                guard_band_width,
                palette_to_game_scale,
            )
            # Use adjusted luminosity for subsequent mapping
            luminosity = lum_adj.astype(np.float32)
        except Exception:
            logger.exception("Collision resolver failed; continuing with baseline luminosity")

    for island_index, ((island_name, gray_start, gray_end), mask) in enumerate(zip(active_islands, masks)):
        if mask is None:
            continue

        # Ensure transparent pixels are excluded even if the mask was drawn over them.
        mask = mask & non_transparent

        if not mask.any():
            continue

        # --- Optional: pre-quantize island colors to its available slots ---
        try:
            island_size = max(1, int(gray_end) - int(gray_start) + 1)
        except Exception:
            island_size = 1

        if prequant_enabled and island_size > 0:
            # Count unique colors within island
            island_rgb_current = rgb_array[mask]
            if island_rgb_current.size > 0:
                try:
                    unique_colors_island = np.unique(island_rgb_current.reshape(-1, 3), axis=0).shape[0]
                except Exception:
                    unique_colors_island = 0
            else:
                unique_colors_island = 0

            if unique_colors_island > island_size:
                # Quantize only the island region; re-apply mask afterwards
                try:
                    ys, xs = np.nonzero(mask)
                    y0, y1 = ys.min(), ys.max()
                    x0, x1 = xs.min(), xs.max()

                    sub_mask = mask[y0:y1 + 1, x0:x1 + 1]
                    sub_rgb = rgb_array[y0:y1 + 1, x0:x1 + 1].copy()

                    # Zero-out non-island pixels in the crop to avoid pulling in outside colors
                    # (they may still influence quantization minimally but will be removed later)
                    sub_rgb_masked = sub_rgb.copy()
                    sub_rgb_masked[~sub_mask] = 0

                    # Quantize the RGB-only image to island_size colors using configured method
                    pil_sub = Image.fromarray(sub_rgb_masked, mode='RGB')
                    try:
                        q_img = quantize_image(pil_sub, method=default_quant_method, final_colors=island_size)
                    except Exception:
                        # Fallback: use PIL median cut if primary fails
                        q_img = pil_sub.convert('RGB').quantize(colors=island_size, method=Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)

                    q_rgb = np.asarray(q_img.convert('RGB'))

                    # Re-apply mask: update only island pixels
                    sub_rgb[sub_mask] = q_rgb[sub_mask]
                    rgb_array[y0:y1 + 1, x0:x1 + 1] = sub_rgb

                    # Update luminosity for the changed pixels
                    sub_lum = (0.299 * sub_rgb[:, :, 0] + 0.587 * sub_rgb[:, :, 1] + 0.114 * sub_rgb[:, :, 2]).astype(luminosity.dtype)
                    # Write back via a view to avoid copy-on-indexing pitfalls
                    lum_view = luminosity[y0:y1 + 1, x0:x1 + 1]
                    lum_view[sub_mask] = sub_lum[sub_mask]
                    luminosity[y0:y1 + 1, x0:x1 + 1] = lum_view

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Pre-quantized island '%s' to %d colors (unique before=%d)", island_name, island_size, unique_colors_island)
                except Exception:
                    logger.exception("Island pre-quantization failed for '%s'", island_name)

        # After optional pre-quantization, compute island luminosity from possibly updated rgb_array
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
        elif mapping_strategy == "smoothed_quantile":
            # Parameters from config
            bins = int(cfg.get(cfg.ci_smoothed_quantile_bins)) if hasattr(cfg, "ci_smoothed_quantile_bins") else 256
            sigma = float(cfg.get(cfg.ci_smoothed_quantile_sigma)) if hasattr(cfg, "ci_smoothed_quantile_sigma") else 1.5
            alpha_pct = int(cfg.get(cfg.ci_smoothed_quantile_alpha)) if hasattr(cfg, "ci_smoothed_quantile_alpha") else 30
            alpha_blend = max(0.0, min(1.0, alpha_pct / 100.0))
            remapped = _map_smoothed_quantile(island_luminosity, gray_start, gray_end,
                                              palette_to_game_scale,
                                              guard_band_width=guard_band_width,
                                              bins=bins, sigma=sigma, alpha=alpha_blend)
            grayscale_output[mask] = remapped
        elif mapping_strategy == "tempered_quantile":
            alpha_pct = int(cfg.get(cfg.ci_tempered_quantile_alpha)) if hasattr(cfg, "ci_tempered_quantile_alpha") else 30
            alpha_blend = max(0.0, min(1.0, alpha_pct / 100.0))
            remapped = _map_tempered_quantile(island_luminosity, gray_start, gray_end,
                                              palette_to_game_scale,
                                              guard_band_width=guard_band_width,
                                              alpha=alpha_blend)
            grayscale_output[mask] = remapped
        elif mapping_strategy == "spline_quantile":
            profile = cfg.get(cfg.ci_spline_profile) if hasattr(cfg, "ci_spline_profile") else "even"
            gamma = float(cfg.get(cfg.ci_spline_gamma)) if hasattr(cfg, "ci_spline_gamma") else 1.0
            remapped = _map_spline_quantile(island_luminosity, gray_start, gray_end,
                                            palette_to_game_scale,
                                            guard_band_width=guard_band_width,
                                            profile=profile, gamma=gamma)
            grayscale_output[mask] = remapped
        else:  # Default: "luminosity"
            remapped = _map_luminosity_default(island_luminosity, gray_start, gray_end, 
                                               palette_to_game_scale, guard_band_width)
            grayscale_output[mask] = remapped

        # Get palette space indices for color mapping
        if mapping_strategy in ["color_clustering", "perceptual"]:
            # For these strategies, remapped values are already in game scale
            island_gray = np.rint(grayscale_output[mask] / palette_to_game_scale).astype(np.uint8)
        else:
            # For luminosity-based strategies, convert back to palette space
            island_gray = np.rint(grayscale_output[mask] / palette_to_game_scale).astype(np.uint8)

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
                # Robust per-bin aggregation in Lab with optional ΔE clamp
                try:
                    robust_enable = cfg.get(cfg.ci_palette_anchor_robust_enable) if hasattr(cfg, "ci_palette_anchor_robust_enable") else True
                    delta_e_max = float(cfg.get(cfg.ci_palette_anchor_deltaE_max)) if hasattr(cfg, "ci_palette_anchor_deltaE_max") else 2.0
                except Exception:
                    robust_enable = True
                    delta_e_max = 2.0

                col_arr = np.asarray(colors, dtype=np.uint8)
                if robust_enable and col_arr.ndim == 2 and col_arr.shape[1] == 3 and col_arr.shape[0] > 0:
                    lab = _rgb_to_lab_cv(col_arr)
                    med = np.median(lab, axis=0).astype(np.uint8)
                    # Clamp to nearest contributing color if ΔE too large
                    dists = np.linalg.norm(lab.astype(np.int16) - med.astype(np.int16), axis=1)
                    min_idx = int(np.argmin(dists))
                    if float(dists[min_idx]) > delta_e_max:
                        med = lab[min_idx]
                    rgb_med = _lab_to_rgb_cv(med)
                    averaged_colors[gray_val] = rgb_med.astype(np.uint8)
                else:
                    averaged_colors[gray_val] = np.mean(col_arr, axis=0).astype(np.uint8)
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

    # Build anchor mask (indices that came from observed colors)
    anchor_mask = np.zeros(palette_width, dtype=bool)

    for island_name, gray_start, gray_end in active_islands:
        colors = island_colors.get(island_name, {})
        for gray_val in range(gray_start, min(gray_end + 1, palette_width)):
            if colors.get(gray_val) is not None:
                palette_row[gray_val] = colors[gray_val]
                anchor_mask[gray_val] = True
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

    # Fill guard bands with appropriate colors based on strategy (skip anchors)
    if mapping_strategy in ["guard_bands_quantile", "guard_bands", "smoothed_quantile", "spline_quantile", "tempered_quantile"] and guard_band_width > 0:
        _fill_guard_bands(palette_row, active_islands, guard_band_width, anchor_mask=anchor_mask)
    elif mapping_strategy == "nearest_neighbor_reserve":
        _fill_nearest_neighbor_guard_bands(palette_row, active_islands, anchor_mask=anchor_mask)

    # Duplicate the palette row to all rows BEFORE smoothing
    # This ensures we have a proper 2D image for smoothing filters
    for theme_row in range(1, palette_height):
        palette[theme_row, :] = palette_row

    # --- Optional: Anchor-snap grayscale to node greys for known original colors ---
    try:
        snap_enable = cfg.get(cfg.ci_linear_anchor_snap_enable) if hasattr(cfg, "ci_linear_anchor_snap_enable") else False
    except Exception:
        snap_enable = False
    if snap_enable:
        try:
            snap_strength_pct = int(cfg.get(cfg.ci_linear_anchor_snap_strength)) if hasattr(cfg, "ci_linear_anchor_snap_strength") else 100
        except Exception:
            snap_strength_pct = 100
        try:
            snap_epsilon = float(cfg.get(cfg.ci_linear_anchor_snap_epsilon)) if hasattr(cfg, "ci_linear_anchor_snap_epsilon") else 2.0
        except Exception:
            snap_epsilon = 2.0

        s_global = max(0.0, min(1.0, snap_strength_pct / 100.0))
        pw = palette_width
        if pw > 1 and s_global > 0.0:
            # Precompute anchor info per island
            for (island_name, gray_start, gray_end), mask in zip(active_islands, masks):
                if mask is None or not mask.any():
                    continue
                # Collect anchors within this island
                island_anchor_idx = [k for k in range(gray_start, min(gray_end + 1, pw)) if anchor_mask[k]]
                if len(island_anchor_idx) == 0:
                    continue
                anchors_rgb = palette_row[island_anchor_idx]
                # Game greys for anchors (rounded)
                gk = np.rint(np.array(island_anchor_idx, dtype=np.float32) * (255.0 / float(pw - 1))).astype(np.uint8)

                # Pixel set for this island
                pix_rgb = rgb_array[mask]
                if pix_rgb.size == 0:
                    continue

                # 1) Hard snap exact matches using dict of bytes → gk
                # Build dictionary from anchor RGB to gk (first occurrence wins)
                anchor_map = {}
                for idx_local, k in enumerate(island_anchor_idx):
                    rgb_t = tuple(int(v) for v in palette_row[k])
                    if rgb_t not in anchor_map:
                        anchor_map[rgb_t] = int(gk[idx_local])

                # Prepare view for unique color processing
                colors_flat = pix_rgb.reshape(-1, 3)
                unique_cols, inv_idx = np.unique(colors_flat, axis=0, return_inverse=True)

                # Targets initialized as NaN (no snap)
                target_g = np.full(unique_cols.shape[0], np.nan, dtype=np.float32)
                # Exact matches
                for ui, col in enumerate(unique_cols):
                    key = (int(col[0]), int(col[1]), int(col[2]))
                    if key in anchor_map:
                        target_g[ui] = float(anchor_map[key])

                # 2) Soft snap near matches within epsilon (if epsilon > 0)
                if snap_epsilon > 0.0:
                    # Compute Lab distances to anchors for the unique colors that are not exact matches
                    need_soft = np.isnan(target_g)
                    if np.any(need_soft):
                        try:
                            uc = unique_cols[need_soft]
                            lab_uc = _rgb_to_lab_cv(uc)
                            lab_anchors = _rgb_to_lab_cv(anchors_rgb)
                            # Broadcast distances (U x A)
                            diff = lab_uc.astype(np.int16)[:, None, :] - lab_anchors.astype(np.int16)[None, :, :]
                            dists = np.sqrt(np.sum(diff * diff, axis=2))
                            min_j = np.argmin(dists, axis=1)
                            min_d = dists[np.arange(dists.shape[0]), min_j]
                            # Apply only where within epsilon
                            within = min_d <= snap_epsilon
                            if np.any(within):
                                # Map back to indices in unique_cols
                                idxs = np.nonzero(need_soft)[0]
                                chosen = idxs[within]
                                # assign target_g for chosen to the corresponding anchor gk
                                target_g[chosen] = gk[min_j[within]].astype(np.float32)
                                # Store per-unique snap weight for soft blending
                                soft_w = np.zeros(unique_cols.shape[0], dtype=np.float32)
                                soft_w[chosen] = s_global * np.clip(1.0 - (min_d[within] / snap_epsilon), 0.0, 1.0)
                            else:
                                soft_w = np.zeros(unique_cols.shape[0], dtype=np.float32)
                        except Exception:
                            # Fallback: no soft snap
                            soft_w = np.zeros(unique_cols.shape[0], dtype=np.float32)
                    else:
                        soft_w = np.zeros(unique_cols.shape[0], dtype=np.float32)
                else:
                    soft_w = np.zeros(unique_cols.shape[0], dtype=np.float32)

                # Build per-pixel targets and weights
                per_pix_target = target_g[inv_idx]
                per_pix_soft_w = soft_w[inv_idx]

                # Current grayscale values (0..255) for this island
                g_lin = grayscale_filled[mask].astype(np.float32)

                # Hard snap where target present and weight is 1 (exact or strong)
                # For exact matches we want hard snap (weight = 1). Ensure those have weight 1.
                # Determine exact matches again quickly: weight == 0 but target_g set (from dict)
                # For those, set weight to 1
                has_target = ~np.isnan(per_pix_target)
                # If weight was 0 but target exists, set to 1 (hard snap for exact)
                per_pix_soft_w = np.where((has_target) & (per_pix_soft_w <= 0.0), 1.0, per_pix_soft_w)

                # Blend
                g_out = g_lin
                if np.any(has_target):
                    tgt = np.nan_to_num(per_pix_target, nan=g_lin)
                    w = per_pix_soft_w.astype(np.float32)
                    g_out = (1.0 - w) * g_lin + w * tgt
                    g_out = np.rint(np.clip(g_out, 0.0, 255.0)).astype(np.uint8)
                    grayscale_filled[mask] = g_out

    # Apply palette smoothing if enabled to reduce harsh color transitions
    # Now smooth the entire palette image (all rows together)
    palette_smooth_method = cfg.get(cfg.ci_palette_smooth_method) if hasattr(cfg, "ci_palette_smooth_method") else "none"
    palette_smooth_strength = float(cfg.get(cfg.ci_palette_smooth_strength) / 100) if hasattr(cfg, "ci_palette_smooth_strength") else 0.0

    # Preserve a copy before smoothing for anchor restore
    palette_before_smooth = palette.copy()
    if palette_smooth_method != "none" and palette_smooth_strength > 0.0:
        palette = _smooth_palette_image(palette, palette_smooth_method, palette_smooth_strength)
        try:
            preserve_anchors = cfg.get(cfg.ci_preserve_observed_palette_indices) if hasattr(cfg, "ci_preserve_observed_palette_indices") else True
        except Exception:
            preserve_anchors = True
        if preserve_anchors:
            # Restore anchor columns exactly
            palette[:, anchor_mask, :] = palette_before_smooth[:, anchor_mask, :]

    # Implement palette upscaling if enabled
    upscale_enabled = cfg.get(cfg.ci_palette_upscale_to_256) if hasattr(cfg, "ci_palette_upscale_to_256") else False
    if upscale_enabled and palette_width < 256:
        palette = _upscale_palette_to_256(palette, palette_width)

    palette_img = Image.fromarray(palette, mode='RGB')

    mask_stack_out = np.stack(masks, axis=0) if masks else np.zeros((0, height, width), dtype=bool)
    # Return possibly adjusted islands so callers can update UI state and persistence
    return grayscale_filled, palette_img, mask_stack_out, active_islands


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
