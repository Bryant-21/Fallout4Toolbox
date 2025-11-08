import os
import subprocess
from collections import Counter

import imagequant
import numpy as np
from PIL import Image
from PIL.Image import Quantize, Palette
from skimage.color import rgb2lab, deltaE_ciede2000

from src.utils.appconfig import cfg, TEXCONV_EXE
from src.utils.logging_utils import logger


def load_image(path, format='RGB'):
    """Load an image path into a PIL Image.

    - For non-DDS formats, uses PIL directly and converts to requested 'format' (default RGB).
    - For .dds, uses texconv.exe to convert to a temporary PNG, then loads it and converts to requested 'format'.

    Args:
        path (str): Input image path.
        format (str): Desired PIL mode for the returned image, e.g., 'RGB' or 'RGBA'.

    Returns:
        PIL.Image.Image: Loaded image in the requested mode.

    Raises:
        Exception: If texconv is required but not set or conversion fails.
    """
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext != '.dds':
            # Use PIL for regular formats
            with Image.open(path) as im:
                return im.convert(format)

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run texconv to output PNG into tmpdir
            # Use -ft PNG to force PNG, single mip (-m 1), sRGB
            cmd = [
                TEXCONV_EXE,
                '-ft', 'PNG',
                '-y',
                '-m', '1',
                '-srgb',
                path,
                '-o', tmpdir
            ]
            logger.debug(f"Running texconv for DDS load: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.error(f"texconv failed while reading DDS: {result.stderr}")
                raise Exception(f"texconv failed while reading DDS: {result.stderr}")

            # texconv outputs a PNG with same basename but .PNG extension
            base = os.path.splitext(os.path.basename(path))[0]
            # texconv may output uppercase .PNG; try both cases
            cand_paths = [
                os.path.join(tmpdir, base + '.PNG'),
                os.path.join(tmpdir, base + '.png')
            ]
            out_png = next((p for p in cand_paths if os.path.exists(p)), None)
            if not out_png:
                # Sometimes texconv may create subfolders; scan tmpdir for first PNG
                for fn in os.listdir(tmpdir):
                    if fn.lower().endswith('.png'):
                        out_png = os.path.join(tmpdir, fn)
                        break
            if not out_png:
                raise Exception("texconv did not produce a PNG output when reading DDS")

            with Image.open(out_png) as im:
                return im.convert(format)
    except subprocess.TimeoutExpired:
        logger.error("texconv timed out while reading DDS")
        raise Exception("texconv timed out while reading DDS")
    except Exception as e:
        logger.error(f"Failed to load image '{path}': {e}")
        raise


def quantize_image(image, method, use_lower_quant=False):
    """Quantize image using the specified method

    Implements optional two-stage strategy (1): over-quantize per-image to preserve rare colors,
    leaving global reduction to later steps. Controlled by cfg.ci_advanced_quant.
    Also biases libimagequant toward quality over speed when available.
    """
    info = {'method': method}
    logger.debug(f"Quantizing with method: {method}")

    # Determine palette sizes
    final_colors = int(cfg.get(cfg.ci_default_palette_size))
    use_advanced = bool(cfg.get(cfg.ci_advanced_quant)) if hasattr(cfg, 'ci_advanced_quant') else False
    # Optional lower-per-image quantization factor (e.g., 0.5 -> 128 when final is 256)
    try:
        lower_factor = float(cfg.get(cfg.ci_lower_quant_factor)) if cfg.get(cfg.ci_lower_quant_factor) and use_lower_quant else 1.0
    except Exception:
        lower_factor = 1.0
    lower_factor = 1.0 if lower_factor is None else float(lower_factor)

    if lower_factor < 1.0:
        # Explicitly quantize per-image to fewer colors than the final palette size
        intermediate_colors = max(2, int(round(final_colors * lower_factor)))
        info['intermediate_colors'] = intermediate_colors
        info['lower_factor'] = lower_factor
    else:
        # Over-quantize to preserve rare colors: 2x final, capped (only if not lowering)
        intermediate_colors = max(final_colors, min(1024, min(256, final_colors * 2))) if use_advanced else final_colors
        if use_advanced and intermediate_colors != final_colors:
            info['intermediate_colors'] = intermediate_colors

    try:
        if method == "median_cut":
            quantized = image.quantize(colors=intermediate_colors, method=Quantize.MEDIANCUT)
            info['description'] = "Median Cut - Good color relationships, can be blocky"

        elif method == "max_coverage":
            quantized = image.quantize(colors=intermediate_colors, method=Quantize.MAXCOVERAGE)
            info['description'] = "Max Coverage - Maximizes color variety"

        elif method == "fast_octree":
            quantized = image.quantize(colors=intermediate_colors, method=Quantize.FASTOCTREE)
            info['description'] = "Fast Octree - Fast, good for photos"

        elif method == "libimagequant":
            try:
                # Favor quality over speed by allowing more colors and enabling strong dithering.
                # Pillow's LIQ wrapper doesn't expose quality/speed knobs; using higher color budget
                # and Floyd-Steinberg dithering biases toward quality.
                quantized = imagequant.quantize_pil_image(
                    image,
                    dithering_level=1.0,
                    max_colors=intermediate_colors,
                    min_quality=0,  # from 0 to 100
                    max_quality=100,  # from 0 to 100
                )
                info['description'] = "LibImageQuant - High quality (favoring quality over speed)"
            except Exception as e:
                logger.warning(f"LibImageQuant failed with method {method}: {str(e)}")
                quantized = image.quantize(colors=intermediate_colors, method=Quantize.MEDIANCUT)
                info['description'] = "LibImageQuant (fallback to Median Cut)"

        elif method == "kmeans_adaptive":
            # Use a larger k when advanced enabled so k-means can place centroids on rare colors
            quantized = image.quantize(colors=intermediate_colors, method=Quantize.FASTOCTREE, kmeans=intermediate_colors)
            info['description'] = "K-means Adaptive - Adaptive color distribution"

        elif method == "uniform":
            # For uniform method, ensure we get close to target colors
            uniform_img = image.convert("P", palette=Palette.ADAPTIVE, colors=intermediate_colors)
            quantized = uniform_img.convert("RGB").quantize(colors=intermediate_colors)
            info['description'] = "Uniform - Helps with color banding"

        else:
            quantized = image.quantize(colors=intermediate_colors, method=Quantize.MEDIANCUT)
            info['description'] = "Median Cut (default)"

        # Verify we have a reasonable number of colors
        quantized_rgb = quantized.convert('RGB')
        quantized_array = np.array(quantized_rgb)
        unique_colors = np.unique(quantized_array.reshape(-1, 3), axis=0)
        info['initial_color_count'] = len(unique_colors)

        logger.debug(
            f"Quantization completed with {method} (advanced={'on' if use_advanced else 'off'}), "
            f"requested={intermediate_colors}, produced {len(unique_colors)} colors"
        )
        return quantized, info

    except Exception as e:
        logger.error(f"Quantization failed with method {method}: {str(e)}")
        raise


def pad_colors_to_target(existing_colors, original_image, target_size):
    """Pad the color palette to reach exactly target_size colors"""
    logger.debug(f"Padding colors from {len(existing_colors)} to {target_size}")

    existing_colors_set = set(tuple(color) for color in existing_colors)
    padded_colors = existing_colors.copy()

    # Strategy 1: Add gradient colors between existing colors
    if len(padded_colors) > 1:
        try:
            padded_colors = add_gradient_colors(padded_colors, existing_colors_set, target_size)
            if len(padded_colors) >= target_size:
                return padded_colors[:target_size]
        except Exception as e:
            logger.warning(f"Gradient padding failed: {e}")

    # Strategy 2: Add colors from the original image that aren't in the palette
    if len(padded_colors) < target_size:
        try:
            padded_colors = add_missing_colors_from_original(padded_colors, original_image,
                                                                  existing_colors_set, target_size)
            if len(padded_colors) >= target_size:
                return padded_colors[:target_size]
        except Exception as e:
            logger.warning(f"Original color padding failed: {e}")

    # Strategy 3: Add evenly spaced colors in RGB space
    if len(padded_colors) < target_size:
        try:
            padded_colors = add_rgb_grid_colors(padded_colors, existing_colors_set, target_size)
            if len(padded_colors) >= target_size:
                return padded_colors[:target_size]
        except Exception as e:
            logger.warning(f"RGB grid padding failed: {e}")

    # Final fallback: duplicate existing colors to reach target_size
    if len(padded_colors) < target_size:
        logger.warning(f"Using fallback padding, duplicating colors to reach {target_size}")
        while len(padded_colors) < target_size:
            for color in existing_colors:
                if len(padded_colors) >= target_size:
                    break
                # Add a slightly modified version of the color
                modified_color = modify_color_slightly(color)
                if tuple(modified_color) not in existing_colors_set:
                    padded_colors = np.vstack([padded_colors, modified_color])
                    existing_colors_set.add(tuple(modified_color))

    return padded_colors[:target_size]


def add_gradient_colors(existing_colors, existing_colors_set, target_size):
    """Add gradient colors between existing colors"""
    padded_colors = existing_colors.copy()
    num_existing = len(existing_colors)

    for i in range(num_existing):
        for j in range(i + 1, min(i + 5, num_existing)):  # Limit to nearby colors
            if len(padded_colors) >= target_size:
                break

            color1 = existing_colors[i]
            color2 = existing_colors[j]

            # Add intermediate colors
            for k in range(1, 4):  # Add 3 intermediate colors
                if len(padded_colors) >= target_size:
                    break

                ratio = k / 4.0
                new_color = (
                    int(color1[0] * (1 - ratio) + color2[0] * ratio),
                    int(color1[1] * (1 - ratio) + color2[1] * ratio),
                    int(color1[2] * (1 - ratio) + color2[2] * ratio)
                )

                if new_color not in existing_colors_set:
                    padded_colors = np.vstack([padded_colors, new_color])
                    existing_colors_set.add(new_color)

    return padded_colors


def add_missing_colors_from_original(existing_colors, original_image, existing_colors_set, target_size):
    """Add frequent colors from original image that are missing from palette"""
    padded_colors = existing_colors.copy()

    # Sample colors from original image
    original_array = np.array(original_image)
    height, width = original_array.shape[:2]

    # Take a random sample of pixels
    sample_size = min(10000, height * width)
    indices = np.random.choice(height * width, sample_size, replace=False)
    sampled_colors = original_array.reshape(-1, 3)[indices]

    # Count color frequencies in sample
    color_counter = Counter(tuple(color) for color in sampled_colors)

    # Add most frequent colors that aren't in the existing palette
    for color, count in color_counter.most_common():
        if len(padded_colors) >= target_size:
            break
        if color not in existing_colors_set:
            padded_colors = np.vstack([padded_colors, color])
            existing_colors_set.add(color)

    return padded_colors


def add_rgb_grid_colors(existing_colors, existing_colors_set, target_size):
    """Add evenly spaced colors from RGB space"""
    padded_colors = existing_colors.copy()

    # Create a grid of colors in RGB space
    steps = 6  # 6^3 = 216 colors total
    for r in range(0, 256, 256 // steps):
        for g in range(0, 256, 256 // steps):
            for b in range(0, 256, 256 // steps):
                if len(padded_colors) >= target_size:
                    break

                grid_color = (r, g, b)
                if grid_color not in existing_colors_set:
                    padded_colors = np.vstack([padded_colors, grid_color])
                    existing_colors_set.add(grid_color)

    return padded_colors


def modify_color_slightly(color, max_change=10):
    """Create a slightly modified version of a color"""
    new_color = [
        max(0, min(255, color[0] + np.random.randint(-max_change, max_change + 1))),
        max(0, min(255, color[1] + np.random.randint(-max_change, max_change + 1))),
        max(0, min(255, color[2] + np.random.randint(-max_change, max_change + 1)))
    ]
    return np.array(new_color, dtype=np.uint8)


def rebuild_image_with_padded_colors(quantized_array, padded_colors):
    """Rebuild the quantized image using the padded color palette"""
    logger.debug("Rebuilding image with padded color palette")

    # For simplicity, we'll keep the original quantized image
    # The actual color replacement would be complex and might change the image appearance
    # In a real implementation, you'd want to re-quantize or map colors properly

    return quantized_array


def analyze_color_distribution(quantized_array):
    """Analyze how often each color appears in the image"""
    logger.debug("Analyzing color distribution")
    height, width = quantized_array.shape[:2]
    color_tuples = [tuple(quantized_array[i, j]) for i in range(height) for j in range(width)]
    distribution = Counter(color_tuples)
    logger.debug(f"Color distribution analyzed: {len(distribution)} unique colors")
    return distribution


def convert_to_dds(input_path, output_path, is_palette=False, palette_width=256, palette_height=8):
    """Convert image to DDS format using texconv.exe"""
    logger.debug(f"Converting to DDS: {input_path} -> {output_path}")
    try:
        if is_palette:
            # Use provided Palette dimensions
            cmd = [
                TEXCONV_EXE,
                '-f', 'BC7_UNORM',
                '-y',
                '-m', '1',
                '-w', str(palette_width),
                '-h', str(palette_height),
                '-srgb',
                input_path,
                '-o', os.path.dirname(output_path)
            ]
        else:
            cmd = [
                TEXCONV_EXE,
                '-f', 'BC7_UNORM',
                '-y',
                '-m', '1',
                '-srgb',
                input_path,
                '-o', os.path.dirname(output_path)
            ]

        logger.debug(f"Running texconv command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.error(f"texconv failed: {result.stderr}")
            raise Exception(f"texconv failed: {result.stderr}")
        else:
            logger.debug("DDS conversion successful")

    except subprocess.TimeoutExpired:
        logger.error("texconv timed out")
        raise Exception("texconv timed out")
    except Exception as e:
        logger.error(f"DDS conversion error: {str(e)}")
        raise Exception(f"DDS conversion error: {str(e)}")


def next_power_of_2(n):
    """Calculate the next power of 2 that is greater than or equal to n"""
    if n <= 1:
        return 1
    # Calculate the position of the most significant bit
    power = 1
    while power < n:
        power *= 2
    return power


def rgb_to_lab_array(colors):
    rgb = np.array(colors, dtype=np.float32) / 255.0
    lab = rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)
    return lab

def hue_angle_from_ab(a, b):
    return (np.degrees(np.arctan2(b, a)) + 360.0) % 360.0

def unwrap_hue(hues):
    hues_sorted = np.sort(hues)
    diffs = np.diff(np.concatenate([hues_sorted, hues_sorted[:1] + 360]))
    max_gap_idx = np.argmax(diffs)
    rotation = hues_sorted[max_gap_idx]
    unwrapped = (hues - rotation) % 360
    return unwrapped, rotation

def path_cost(order, lab_current):
    # CIE76 ΔE = Euclidean distance in CIELAB
    diffs = lab_current[order[1:]] - lab_current[order[:-1]]
    return np.sum(np.linalg.norm(diffs, axis=1))

def calculate_path_continuity(sorted_lab):
    diffs = sorted_lab[1:] - sorted_lab[:-1]
    dists = np.linalg.norm(diffs, axis=1)
    return float(np.sum(dists)), float(np.max(dists)) if len(dists) > 0 else 0.0

def reduce_colors_lab_de00_with_hue_balance(unique_colors: np.ndarray, counts: np.ndarray, target_size: int, coverage: np.ndarray | None = None):
    """Reduce a color set down to <= target_size using LAB/ΔE00 with hue balancing.

    Args:
        unique_colors: (N,3) uint8 RGB array of unique colors.
        counts: (N,) float/int array of per-color frequencies.
        target_size: desired max palette size.
        coverage: (N,) optional counts of how many distinct images contain the color; if None, uses ones.

    Returns:
        kept_reps: list[tuple[int,int,int]] of selected representative colors (<= target_size)
        color_map: dict mapping original color tuple -> representative tuple
        pad_candidates: list of additional representative candidates not selected (best-first)
    """
    if unique_colors is None or len(unique_colors) == 0:
        return [], {}, []

    colors = unique_colors.astype(np.uint8)
    freqs = counts.astype(np.float64)

    labs = rgb_to_lab_array(colors.tolist()).astype(np.float64)
    a = labs[:, 1]
    b = labs[:, 2]
    chroma = np.sqrt(a * a + b * b)
    Cmax = max(1e-6, float(np.max(chroma)))
    hues = np.array([hue_angle_from_ab(ai, bi) for ai, bi in zip(a, b)], dtype=np.float64)

    # Effective weight with chroma/coverage boosts
    alpha = 0.85
    k_chroma = 0.6
    k_cov = 0.25
    coverage_arr = coverage.astype(np.float64) if coverage is not None else np.ones_like(freqs)
    eff_w = np.power(freqs, alpha) * (1.0 + k_chroma * (chroma / Cmax)) + k_cov * coverage_arr

    def de00_to_centers(lab_point: np.ndarray, centers: np.ndarray) -> np.ndarray:
        lp = lab_point.reshape(1, 1, 3)
        ct = centers.reshape(-1, 1, 3)
        return deltaE_ciede2000(ct, lp).reshape(-1)

    thresholds = list(range(0, 21, 1))
    best_result = None
    for thresh in thresholds:
        kept_labs = []
        clusters = []
        for i, lab_i in enumerate(labs):
            if not kept_labs:
                kept_labs.append(lab_i.copy())
                clusters.append([i])
                continue
            centers = np.vstack(kept_labs)
            d = de00_to_centers(lab_i, centers)
            j = int(np.argmin(d))
            if d[j] <= thresh:
                clusters[j].append(i)
                members = clusters[j]
                w = eff_w[members]
                kept_labs[j] = np.sum(labs[members] * w[:, None], axis=0) / max(1e-9, np.sum(w))
            else:
                kept_labs.append(lab_i.copy())
                clusters.append([i])
        num_clusters = len(clusters)
        best_result = (clusters, np.vstack(kept_labs) if kept_labs else np.empty((0, 3)))
        if num_clusters <= target_size:
            break

    clusters, kept_labs = best_result

    reps = []
    rep_index_of_cluster = []
    cluster_scores = []
    cluster_median_chroma = []
    cluster_coverage_sum = []
    cluster_hue_bin = []

    hue_bins = 24
    bin_width = 360.0 / hue_bins

    def hue_to_bin(deg: float) -> int:
        return int(np.floor(deg / bin_width)) % hue_bins

    for members in clusters:
        if len(members) == 0:
            continue
        if len(members) == 1:
            k = members[0]
            reps.append(tuple(int(x) for x in colors[k]))
            rep_index_of_cluster.append(k)
            cluster_scores.append(float(eff_w[k]))
            cluster_median_chroma.append(float(chroma[k]))
            cluster_coverage_sum.append(float(coverage_arr[k]))
            cluster_hue_bin.append(hue_to_bin(hues[k]))
            continue
        member_idx = np.array(members, dtype=int)
        member_labs = labs[member_idx]
        member_w = eff_w[member_idx]
        costs = []
        for m in range(member_labs.shape[0]):
            d = deltaE_ciede2000(member_labs[m].reshape(1, 1, 3), member_labs.reshape(-1, 1, 3)).reshape(-1)
            costs.append(float(np.dot(d, member_w)))
        min_idx = int(np.argmin(np.array(costs)))
        k = int(member_idx[min_idx])
        reps.append(tuple(int(x) for x in colors[k]))
        rep_index_of_cluster.append(k)
        cluster_scores.append(float(np.sum(member_w)))
        cluster_median_chroma.append(float(np.median(chroma[member_idx])))
        cluster_coverage_sum.append(float(np.sum(coverage_arr[member_idx])))
        cluster_hue_bin.append(hue_to_bin(hues[k]))

    n_clusters = len(reps)

    kept_indices = list(range(n_clusters))
    pad_candidates = []
    if n_clusters > target_size:
        max_per_bin = max(1, int(np.round(target_size * 0.15)))
        high_chroma_bins = set()
        for i in range(n_clusters):
            if cluster_median_chroma[i] >= 20.0:
                high_chroma_bins.add(cluster_hue_bin[i])
        totals = [float(np.sum(freqs[clusters[i]])) for i in range(n_clusters)]
        order = sorted(range(n_clusters), key=lambda i: (
            -cluster_scores[i], -cluster_median_chroma[i], -cluster_coverage_sum[i], -totals[i]
        ))
        selected = []
        used_bins = Counter()
        best_in_bin = {}
        for i in order:
            b = cluster_hue_bin[i]
            if b in high_chroma_bins and b not in best_in_bin:
                best_in_bin[b] = i
        for b, i in best_in_bin.items():
            selected.append(i)
            used_bins[b] += 1
        for i in order:
            if len(selected) >= target_size:
                break
            b = cluster_hue_bin[i]
            if i in selected:
                continue
            if used_bins[b] >= max_per_bin:
                continue
            selected.append(i)
            used_bins[b] += 1
        kept_indices = selected[:target_size]
        pad_candidates = [reps[i] for i in order if i not in kept_indices]
    else:
        kept_indices = list(range(n_clusters))
        pad_candidates = []

    kept_reps = [reps[i] for i in kept_indices]
    kept_rep_indices = [rep_index_of_cluster[i] for i in kept_indices]
    kept_rep_labs = labs[kept_rep_indices]
    kept_rep_bins = [cluster_hue_bin[i] for i in kept_indices]

    color_map = {}
    for idx in range(len(colors)):
        lab_i = labs[idx]
        bin_i = hue_to_bin(hues[idx])
        same_bin_indices = [j for j, b in enumerate(kept_rep_bins) if b == bin_i]
        if same_bin_indices:
            centers = kept_rep_labs[same_bin_indices]
            d = deltaE_ciede2000(centers.reshape(-1, 1, 3), lab_i.reshape(1, 1, 3)).reshape(-1)
            jrel = int(np.argmin(d))
            j = same_bin_indices[jrel]
        else:
            d = deltaE_ciede2000(kept_rep_labs.reshape(-1, 1, 3), lab_i.reshape(1, 1, 3)).reshape(-1)
            j = int(np.argmin(d))
        color_map[tuple(int(x) for x in colors[idx])] = kept_reps[j]

    return kept_reps, color_map, pad_candidates


def remap_rgb_array_to_representatives(arr_rgb: np.ndarray, color_map: dict) -> np.ndarray:
    """Remap each pixel in an RGB array to its representative color using color_map."""
    if arr_rgb.size == 0 or not color_map:
        return arr_rgb
    flat = arr_rgb.reshape(-1, 3)
    remapped = flat.copy()
    # Apply per-unique mapping
    unique, _ = np.unique(flat, axis=0, return_counts=True)
    lookup = {tuple(map(int, k)): np.array(v, dtype=np.uint8) for k, v in color_map.items()}
    for uc in unique:
        t = tuple(int(x) for x in uc)
        rep = lookup.get(t)
        if rep is None:
            continue
        mask = (flat[:, 0] == t[0]) & (flat[:, 1] == t[1]) & (flat[:, 2] == t[2])
        remapped[mask] = rep
    return remapped.reshape(arr_rgb.shape)


def perceptual_color_sort(colors):
    """
    Linear perceptual color sort that creates a smooth gradient by:
    - Unwrapping hue to avoid wrap-around artifacts
    - Choosing an initial sort based on hue dominance
    - Optimizing the path with 2-opt using CIE76 ΔE (Euclidean in CIELAB)
    """
    n = len(colors)
    if n <= 1:
        return list(colors)

    lab = rgb_to_lab_array(colors)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

    # Compute hue and chroma
    hues = np.array([hue_angle_from_ab(ai, bi) for ai, bi in zip(a, b)])
    chromas = np.sqrt(a ** 2 + b ** 2)

    # --- Smart initial ordering ---
    hues_unwrapped, _ = unwrap_hue(hues)
    hue_range = hues_unwrapped.max() - hues_unwrapped.min()
    is_hue_dominant = hue_range > 90  # Threshold for rainbow-like vs. narrow-hue palettes

    if is_hue_dominant:
        # Sort primarily by hue, secondarily by lightness (to order dark→light within same hue)
        initial_order = np.lexsort((L, hues_unwrapped))
    else:
        # Narrow hue range (e.g., yellows, blues, monochrome): sort by lightness, then chroma
        initial_order = np.lexsort((chromas, L))

    sorted_colors = [colors[i] for i in initial_order]
    lab_current = lab[initial_order].copy()

    # --- Path optimization using 2-opt ---
    current_order = list(range(n))
    current_cost = path_cost(current_order, lab_current)

    improved = True
    iterations = 0
    max_polish_iter = min(cfg.get(cfg.ci_palette_color_iteration), n * 5)

    while improved and iterations < max_polish_iter:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 2, n - 1):
                # 2-opt swap: reverse segment from i to j
                new_order = current_order[:i] + current_order[i:j + 1][::-1] + current_order[j + 1:]
                new_cost = path_cost(new_order, lab_current)
                if new_cost < current_cost * 0.999:  # Require small but meaningful improvement
                    current_order, current_cost = new_order, new_cost
                    improved = True
                    break
            if improved:
                break
        iterations += 1

    # Apply final order
    if improved or iterations > 0:
        sorted_colors = [sorted_colors[i] for i in current_order]
        lab_current = lab_current[current_order]

    total_de, max_de = calculate_path_continuity(lab_current)

    method = "hue-guided+2opt" if is_hue_dominant else "L/C-guided+2opt"
    logger.debug(f"Linear gradient sort complete ({method}): {n} colors")
    logger.debug(
        f"Total ΔE: {total_de:.3f}, Average ΔE: {total_de / (n - 1) if n > 1 else 0:.3f}, Max ΔE: {max_de:.3f}")

    start_lab = lab_current[0]
    end_lab = lab_current[-1]
    logger.debug(f"Gradient range: L{start_lab[0]:.1f}→L{end_lab[0]:.1f}")

    start_hue = hue_angle_from_ab(start_lab[1], start_lab[2])
    end_hue = hue_angle_from_ab(end_lab[1], end_lab[2])
    logger.debug(f"Hue range: {start_hue:.1f}° → {end_hue:.1f}°")

    return sorted_colors