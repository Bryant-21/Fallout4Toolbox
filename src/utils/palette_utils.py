import cv2
import imagequant
import numpy as np
from PIL import Image, ImageFilter
from PIL.Image import Quantize, Palette

from src.utils.appconfig import QuantAlgorithm
from src.utils.appconfig import cfg
from src.utils.logging_utils import logger


# --- Quantize -------------------------------------------------------------
def quantize_image(i, method: QuantAlgorithm):
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
    final_colors = int(cfg.get(cfg.ci_default_palette_size))

    try:
        dither_on = bool(cfg.get(cfg.ci_quantize_dither_enable)) if hasattr(cfg, 'ci_quantize_dither_enable') else False
    except Exception:
        dither_on = False

    dither_flag = Image.Dither.FLOYDSTEINBERG if dither_on else Image.Dither.NONE

    try:

        if method == "median_cut":
            # Must be RGB
            image = i.convert('RGB')
            quantized = image.quantize(colors=final_colors, method=Quantize.MEDIANCUT, dither=dither_flag)
            info['description'] = "Median Cut - Good color relationships, can be blocky"

        elif method == "max_coverage":
            # Must be RGB
            image = i.convert('RGB')
            quantized = image.quantize(colors=final_colors, method=Quantize.MAXCOVERAGE, dither=dither_flag)
            info['description'] = "Max Coverage - Maximizes color variety"

        elif method == "fast_octree":
            # Must be RGB
            image = i
            quantized = image.quantize(colors=final_colors, method=Quantize.FASTOCTREE, dither=dither_flag)
            info['description'] = "Fast Octree - Fast, good for photos"

        elif method == "libimagequant":
            try:
                image = i
                # Favor quality over speed by allowing more colors and optional dithering.
                quantized = imagequant.quantize_pil_image(
                    image,
                    dithering_level=1.0 if dither_on else 0.0,
                    max_colors=final_colors,
                    min_quality=85,
                    max_quality=100,
                )

                info['description'] = "LibImageQuant - High quality (favoring quality over speed)"
            except Exception as e:
                logger.warning(f"LibImageQuant failed with method {method}: {str(e)}")
                quantized = image.quantize(colors=final_colors, method=Quantize.MEDIANCUT, dither=dither_flag)
                info['description'] = "LibImageQuant (fallback to Median Cut)"

        elif method == "kmeans_adaptive":
            image = i
            # Use a larger k when advanced enabled so k-means can place centroids on rare colors
            quantized = image.quantize(colors=final_colors, method=Quantize.FASTOCTREE, kmeans=final_colors, dither=dither_flag)
            info['description'] = "K-means Adaptive - Adaptive color distribution"

        elif method == "uniform":
            image = i.convert('RGB')
            # For uniform method, ensure we get close to target colors
            uniform_img = image.convert("P", palette=Palette.ADAPTIVE, colors=final_colors)
            quantized = uniform_img.convert("RGB").quantize(colors=final_colors, dither=dither_flag)
            info['description'] = "Uniform - Helps with color banding"

        else:
            image = i.convert('RGB')
            quantized = image.quantize(colors=final_colors, method=Quantize.MEDIANCUT, dither=dither_flag)
            info['description'] = "Median Cut (default)"

        # Optional post-process: use apply_smooth_dither on palette indices (requested)
        try:
            post_enable = bool(cfg.get(cfg.ci_quantize_post_enable)) if hasattr(cfg, 'ci_quantize_post_enable') else True
        except Exception:
            post_enable = True

        if post_enable:
            try:
                # Work directly on palette indices with our smoothing/dither utility
                pal_list = quantized.getpalette()
                idx = np.array(quantized, dtype=np.uint8)
                idx_sm = apply_smooth_dither(idx, final_colors)
                new_p = Image.fromarray(idx_sm.astype('uint8'), mode='P')
                if pal_list is not None:
                    new_p.putpalette(pal_list)
                quantized = new_p
                logger.debug("Applied post-quant apply_smooth_dither on palette indices")
            except Exception as e:
                logger.warning(f"Post-quant apply_smooth_dither failed, using original quantized image: {e}")

        return quantized
    except Exception as e:
        logger.error(f"Quantization failed with method {method}: {str(e)}")
        raise


# --- Grayscale methods -----------------------------------------------------
def grayscale_luma601_no_norm(q_img):
    """Rec.601 luma Y' in gamma space for P-mode images, no per-image normalization.

    Works on palette indices by computing a LUT from the image palette.
    """

    # Expect P-mode (palette) image; operate on indices
    idx_img = np.array(q_img, dtype=np.uint8)

    # Get palette as (N,3) RGB
    palette = np.array(q_img.getpalette(), dtype=np.uint8).reshape(-1, 3)
    if palette.size == 0:
        # Fallback: if no palette, convert to L via PIL
        return q_img.convert('L')

    # Compute integer luma per palette color
    r = palette[:, 0].astype(np.int32)
    g = palette[:, 1].astype(np.int32)
    b = palette[:, 2].astype(np.int32)
    luma_palette = ((299 * r + 587 * g + 114 * b + 500) // 1000).astype(np.uint8)

    # Map indices to luma
    gray = luma_palette[idx_img]
    return Image.fromarray(gray, mode='L')

# --- Grayscale methods -----------------------------------------------------

# -- Palette retrieval helper
# Sometimes libimagequant (or PIL) exposes a 256-entry palette with trailing
# zeros beyond the actually used indices. Trimming blindly to target_size can
# corrupt index->color lookups (indices >= target_size would be treated as
# out-of-range and mapped to black). Instead, trim to the highest palette index
# actually used by the image. This keeps index mapping consistent with the
# quantized image while still dropping any truly-unused trailing rows.
def get_palette(q_img: Image.Image):
    """Return the palette rows actually needed by the quantized image.

    Args:
        q_img: P-mode image
        target_size: kept for backward compatibility; not used for slicing.

    Returns:
        numpy.ndarray of shape (N, 3) where N = max_used_index + 1.
    """
    palette_full = np.array(q_img.getpalette(), dtype=np.uint8).reshape(-1, 3)

    # Determine the highest palette index actually referenced by pixels.
    idx_img = np.array(q_img, dtype=np.uint8)
    if idx_img.size == 0:
        return palette_full

    max_idx = int(idx_img.max())
    needed = max_idx + 1
    # Guard against malformed palettes (ensure we don't slice beyond bounds)
    needed = min(needed, palette_full.shape[0])
    return palette_full[:needed]


def grayscale_lab(q_img, target_size):
    # Method A: perceptual L channel from LAB
    idx_img = np.array(q_img, dtype=np.uint8)
    palette = get_palette(q_img)
    if palette.size == 0:
        return q_img.convert('L')
    bgr_pal = palette[:, ::-1].reshape(-1, 1, 3)
    lab_pal = cv2.cvtColor(bgr_pal, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    L_pal = lab_pal[:, 0].astype(np.uint8)
    gray = L_pal[idx_img]
    return Image.fromarray(gray, mode='L')



def grayscale_high_precision(q_img, target_size):
    # Method B: linear luminance -> normalized to 0..255
    def linearize(c):
        mask = c <= 0.04045
        out = np.empty_like(c)
        out[mask] = c[mask] / 12.92
        out[~mask] = ((c[~mask] + 0.055) / 1.055) ** 2.4
        return out

    idx_img = np.array(q_img, dtype=np.uint8)
    palette = get_palette(q_img)
    if palette.size == 0:
        return q_img.convert('L')
    palf = palette.astype(np.float64) / 255.0
    lin = linearize(palf)
    lum_pal = 0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2]
    # Map indices to luminance then normalize per-image
    lum_img = lum_pal[idx_img]
    lum_img = lum_img - lum_img.min()
    maxv = lum_img.max()
    if maxv > 0:
        lum_img = lum_img / maxv
    lum_255 = (np.clip(lum_img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(lum_255, mode='L')



def unique_color_list(q_img, target_size):
    """Return unique palette colors in image order"""
    idx_img = np.array(q_img, dtype=np.uint8)
    palette = get_palette(q_img)
    h, w = idx_img.shape
    seen = {}
    order = []
    for y in range(h):
        for x in range(w):
            idx = idx_img[y, x]
            if idx < len(palette):
                c = tuple(palette[idx])
            else:
                c = (0, 0, 0)
            if c not in seen:
                seen[c] = True
                order.append(c)
    return order



def grayscale_perceptual_rank(q_img, target_size):
    """Method C: Rank colors by Lab L, spread evenly across 0-255"""
    unique = unique_color_list(q_img)
    if not unique:
        return grayscale_lab(q_img)

    cols = np.array(unique, dtype=np.uint8)[..., ::-1]  # RGB->BGR
    lab = cv2.cvtColor(cols.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    L = lab[:, 0]
    order = np.argsort(L)

    # Assign equally spaced gray values
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    gray_values = (ranks / max(1, (len(order) - 1)) * 255.0).astype(np.uint8)

    color_to_gray = {tuple(unique[i]): int(gray_values[i]) for i in range(len(unique))}

    # Build grayscale image
    idx_img = np.array(q_img, dtype=np.uint8)
    palette = get_palette(q_img)
    h, w = idx_img.shape
    gray_img = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            idx = idx_img[y, x]
            color = tuple(palette[idx]) if idx < len(palette) else (0, 0, 0)
            gray_img[y, x] = color_to_gray[color]
    return Image.fromarray(gray_img, mode='L')



def grayscale_weighted_lab(q_img, target_size):
    """Method D: Lab L + chroma weighting for better color separation"""
    idx_img = np.array(q_img, dtype=np.uint8)
    palette = get_palette(q_img)
    if palette.size == 0:
        return q_img.convert('L')

    bgr_pal = palette[:, ::-1].reshape(-1, 1, 3)
    lab_pal = cv2.cvtColor(bgr_pal, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    Lp = lab_pal[:, 0].astype(np.float32)
    ap = lab_pal[:, 1].astype(np.float32) - 128
    bp = lab_pal[:, 2].astype(np.float32) - 128
    chroma = np.sqrt(ap ** 2 + bp ** 2)
    weighted_pal = Lp + (chroma * 0.3)
    weighted_img = weighted_pal[idx_img]
    weighted_img = weighted_img - weighted_img.min()
    maxv = weighted_img.max()
    if maxv > 0:
        weighted_img = weighted_img / maxv * 255.0
    gray_img = np.clip(weighted_img, 0, 255).astype(np.uint8)
    return Image.fromarray(gray_img, mode='L')



# --- FIXED Palette builder -------------------------------------------------
def build_palette_at_target_size(q_img, gray_img, target_size):
    """
    Build palette directly at target_size (64, 128, or 256) for P-mode images.
    Assumes q_img is a quantized (mode 'P') image. It maps palette indices to
    representative gray values (median across pixels using that index), then
    assigns RGB colors from the image palette into target_size slots according
    to scaled gray values.
    """
    # Expect P-mode (palette) image
    idx_img = np.array(q_img, dtype=np.uint8)
    g_arr = np.array(gray_img, dtype=np.uint8)

    pal = get_palette(q_img)
    if pal is not None and len(pal) > 0:
        palette = np.array(pal, dtype=np.uint8).reshape(-1, 3)
    else:
        palette = np.zeros((0, 3), dtype=np.uint8)

    if palette.size == 0:
        return [(0, 0, 0)] * target_size

    h, w = idx_img.shape

    # Step 1: For each palette index used, collect its gray values and take median
    from collections import defaultdict
    idx_to_grays = defaultdict(list)
    for y in range(h):
        for x in range(w):
            idx = int(idx_img[y, x])
            if idx < len(palette):
                idx_to_grays[idx].append(int(g_arr[y, x]))

    idx_to_repr_gray = {}
    for pi, grays in idx_to_grays.items():
        if grays:
            idx_to_repr_gray[pi] = int(np.median(grays))

    # Step 2: Map representative gray (0..255) to target palette slot 0..target_size-1
    slot_to_color = {}
    for pi, gray in idx_to_repr_gray.items():
        slot = int(gray * (target_size - 1) / 255.0)
        if slot not in slot_to_color:
            # Assign the first color that arrives to this slot
            rgb = tuple(int(c) for c in palette[pi])
            slot_to_color[slot] = rgb

    # Step 3: Build target_size palette and mark filled slots
    out_pal = [(0, 0, 0)] * target_size
    filled = [False] * target_size
    for slot, color in slot_to_color.items():
        out_pal[slot] = color
        filled[slot] = True

    # Step 4: Fill gaps by forward fill then backward fill
    last_color = (0, 0, 0)
    for i in range(target_size):
        if filled[i]:
            last_color = out_pal[i]
        else:
            out_pal[i] = last_color

    last_color = out_pal[target_size - 1]
    for i in range(target_size - 1, -1, -1):
        if filled[i]:
            last_color = out_pal[i]
        elif out_pal[i] == (0, 0, 0):
            out_pal[i] = last_color

    return out_pal


def downsample_palette_256_to_size(palette_256, target_size):
    """
    Downsample a 256-entry palette to target_size (64 or 128) by sampling evenly.
    Returns a palette of exactly target_size entries.
    """
    if target_size >= 256:
        return palette_256

    # Sample evenly across the 256 entries
    indices = np.linspace(0, 255, target_size, dtype=int)
    downsampled = [palette_256[i] for i in indices]
    return downsampled


# --- Quantized grayscale: reduce to unique values --------------------------

def quantize_grayscale_to_palette_colors(q_img, gray_img):
    """
    Reduce grayscale to only the unique gray values that correspond
    to unique colors in the quantized image.
    """
    q_arr = np.array(q_img)[..., :3]
    g_arr = np.array(gray_img)
    h, w, _ = q_arr.shape

    # Build mapping from color to median gray
    from collections import defaultdict
    color_to_grays = defaultdict(list)
    for y in range(h):
        for x in range(w):
            color = tuple(q_arr[y, x])
            gray = int(g_arr[y, x])
            color_to_grays[color].append(gray)

    color_to_repr_gray = {}
    for color, grays in color_to_grays.items():
        color_to_repr_gray[color] = int(np.median(grays))

    # Create quantized grayscale: map each color to its representative gray
    quantized_gray = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            color = tuple(q_arr[y, x])
            quantized_gray[y, x] = color_to_repr_gray[color]

    return Image.fromarray(quantized_gray, mode="L")


# --- Palette utilities -----------------------------------------------------

def save_palette_image(pal, path):
    w = len(pal)
    img = Image.new("RGB", (w, 1))
    for i, c in enumerate(pal):
        img.putpixel((i, 0), c)
    img = img.resize((w, 32), Image.NEAREST)
    img.save(path)
    return img


def get_palette_row(palette_img, y=0) -> np.ndarray:
    w, h = palette_img.size
    y = max(0, min(h - 1, y))
    row_pixels = np.array(palette_img)[y, :, :3]
    if row_pixels.ndim == 1:
        row_pixels = np.expand_dims(row_pixels, axis=0)
    return row_pixels.astype(np.uint8)


def apply_palette_to_greyscale(palette_img: Image.Image, grey_img: Image.Image) -> Image.Image:
    """Apply palette row to a greyscale image, preserving alpha if present.

    Accepts grey_img in modes:
      - 'L' (grayscale)
      - 'LA' (grayscale with alpha)
      - 'RGB'/'RGBA' (uses the first channel as greyscale index; preserves alpha if present)
    Returns RGB if no alpha, RGBA if alpha present.
    """
    palette_row = get_palette_row(palette_img)
    pw = palette_row.shape[0]

    if pw == 256:
        lut = palette_row
    else:
        # Interpolate along the row to 256 entries
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

def _bayer_matrix_8x8() -> np.ndarray:
    """Return a normalized 8x8 Bayer matrix in range [0,1). for BC7"""
    # Classic 8x8 Bayer threshold map
    bm = np.array([
        [0, 48, 12, 60, 3, 51, 15, 63],
        [32, 16, 44, 28, 35, 19, 47, 31],
        [8, 56, 4, 52, 11, 59, 7, 55],
        [40, 24, 36, 20, 43, 27, 39, 23],
        [2, 50, 14, 62, 1, 49, 13, 61],
        [34, 18, 46, 30, 33, 17, 45, 29],
        [10, 58, 6, 54, 9, 57, 5, 53],
        [42, 26, 38, 22, 41, 25, 37, 21]
    ], dtype=np.float32)
    return (bm + 0.5) / 64.0


def _ordered_dither_indices(indices: np.ndarray, palette_size: int, amount: float) -> np.ndarray:
    """Apply ordered dithering to an indices image (0..palette_size-1)."""
    if indices is None or indices.size == 0 or palette_size <= 1 or amount <= 0.0:
        return indices
    h, w = indices.shape[:2]
    # Normalize to [0,1]
    norm = indices.astype(np.float32) / float(max(1, palette_size - 1))
    bm = _bayer_matrix_8x8()
    # Tile and crop
    tile_h = (h + 7) // 8
    tile_w = (w + 7) // 8
    thresh = np.tile(bm, (tile_h, tile_w))[:h, :w]
    # Mix small blue-noise-like threshold; amount scales perturbation around 0.5
    perturbed = np.clip(norm + amount * (thresh - 0.5), 0.0, 1.0)
    # Re-quantize back to indices
    out = np.round(perturbed * float(max(1, palette_size - 1))).astype(np.int32)
    out = np.clip(out, 0, palette_size - 1).astype(np.uint8)
    return out

def _median_filter_indices(indices: np.ndarray, size: int) -> np.ndarray:
    if size is None or size <= 1:
        return indices
    try:
        img = Image.fromarray(indices.astype('uint8'), 'L')
        filt = img.filter(ImageFilter.MedianFilter(size=int(size)))
        return np.array(filt).astype(np.uint8)
    except Exception:
        return indices

def _gaussian_blur_indices(indices: np.ndarray, radius: float) -> np.ndarray:
    if radius is None or radius <= 0.0:
        return indices
    try:
        img = Image.fromarray(indices.astype('uint8'), 'L')
        filt = img.filter(ImageFilter.GaussianBlur(radius=float(radius)))
        return np.array(filt).astype(np.uint8)
    except Exception:
        return indices


def apply_smooth_dither(indices: np.ndarray, palette_size: int,
                        method: str | None = None,
                        median_size: int | None = None,
                        blur_radius: float | None = None,
                        dither_amount: float | None = None) -> np.ndarray:
    """Optionally smooth/AA/dither greyscale indices image.

    Methods:
      - 'none': no change
      - 'median': median filter (default mild)
      - 'gaussian': gaussian blur (mild)
      - 'dither': ordered 8x8 Bayer dither on normalized indices
      - 'median_dither', 'gaussian_dither': combination
    All parameters have safe mild defaults if None and can be sourced from cfg.
    """
    try:
        if method is None:
            method = str(cfg.get(cfg.ci_greyscale_post_method)) if hasattr(cfg, 'ci_greyscale_post_method') else 'median'
        method = (method or 'median').lower()
        if median_size is None:
            median_size = int(cfg.get(cfg.ci_greyscale_median_size)) if hasattr(cfg, 'ci_greyscale_median_size') else 3
        if blur_radius is None:
            blur_radius = float(cfg.get(cfg.ci_greyscale_blur_radius)) / 10.0 if hasattr(cfg, 'ci_greyscale_blur_radius')  else 0.6
        if dither_amount is None:
            dither_amount = float(cfg.get(cfg.ci_greyscale_dither_amount)) / 100.0 if hasattr(cfg, 'ci_greyscale_dither_amount')  else 0.2
    except Exception:
        method = 'median'
        median_size = 3
        blur_radius = 0.6
        dither_amount = 0.2

    out = indices.astype(np.uint8, copy=True)

    def do_dither(arr):
        try:
            return _ordered_dither_indices(arr, palette_size, max(0.0, float(dither_amount)))
        except Exception:
            return arr

    if method == 'none':
        return out
    elif method == 'median':
        out = _median_filter_indices(out, median_size)
    elif method == 'gaussian':
        out = _gaussian_blur_indices(out, blur_radius)
    elif method == 'dither':
        out = do_dither(out)
    elif method == 'median_dither':
        out = do_dither(_median_filter_indices(out, median_size))
    elif method == 'gaussian_dither':
        out = do_dither(_gaussian_blur_indices(out, blur_radius))
    else:
        # fallback: median
        out = _median_filter_indices(out, median_size)

    return np.clip(out, 0, max(1, palette_size - 1)).astype(np.uint8)