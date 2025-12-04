import imagequant
import numpy as np
from PIL import Image
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
                    dithering_level=1.0,
                    max_colors=final_colors,
                    min_quality=85,
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

