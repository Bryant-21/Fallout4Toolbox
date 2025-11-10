import math
import os
from collections import Counter

import numpy as np
from PIL import Image, ImageFilter
from numba import njit, prange
from skimage.color import rgb2lab, deltaE_ciede2000

from src.utils.appconfig import QuantAlgorithm
from src.utils.appconfig import cfg
from src.utils.logging_utils import logger
from utils.dds_utils import load_image
from utils.palette_utils import quantize_image, _get_palette_array


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


@njit(cache=True, fastmath=True)
def next_power_of_2(n):
    """Calculate the next power of 2 that is greater than or equal to n"""
    if n <= 1:
        return 1
    power = 1
    # simple loop; numba will JIT this tight integer loop
    while power < n:
        power *= 2
    return power


def rgb_to_lab_array(colors):
    rgb = np.array(colors, dtype=np.float32) / 255.0
    lab = rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)
    return lab

@njit(fastmath=True, cache=True)
def hue_angle_from_ab(a, b):
    # Use arctan2 for correct quadrant; convert to degrees manually for numba compatibility
    angle = np.arctan2(b, a) * (180.0 / np.pi)
    if angle < 0.0:
        angle += 360.0
    return angle

@njit(fastmath=True, cache=True)
def unwrap_hue(hues):
    hues_sorted = np.sort(hues)
    diffs = np.diff(np.concatenate((hues_sorted, hues_sorted[:1] + 360.0)))
    max_gap_idx = int(np.argmax(diffs))
    rotation = float(hues_sorted[max_gap_idx])
    unwrapped = (hues - rotation) % 360.0
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


def build_palette_from_rgb_array(colors_source: np.ndarray, target_size: int):
    """
    Build a reduced palette from an RGB array by merging perceptually similar colors until the
    number of clusters is <= target_size. Uses LAB + ΔE00 for clustering and returns medoid reps.

    Args:
        colors_source: RGB ndarray, shape (H,W,3) or (N,3), dtype uint8 preferred.
        target_size: desired maximum palette size.

    Returns:
        kept_reps: list of RGB tuples (<= target_size)
        color_map: dict mapping original color tuple -> representative tuple
        pad_candidates: list of additional representative candidates not kept (best-first)
    """
    if colors_source is None:
        return [], {}, []
    arr = np.asarray(colors_source)
    if arr.size == 0:
        return [], {}, []
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[2])
    if arr.shape[1] != 3:
        raise ValueError("colors_source must have shape (*, 3)")
    colors = np.unique(arr.astype(np.uint8), axis=0)
    if len(colors) == 0:
        return [], {}, []

    labs = rgb_to_lab_array(colors.tolist()).astype(np.float64)

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
                kept_labs[j] = np.mean(labs[members], axis=0)
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
    for members in clusters:
        if len(members) == 0:
            continue
        if len(members) == 1:
            k = members[0]
            reps.append(tuple(int(x) for x in colors[k]))
            rep_index_of_cluster.append(k)
            continue
        member_idx = np.array(members, dtype=int)
        member_labs = labs[member_idx]
        costs = []
        for m in range(member_labs.shape[0]):
            d = deltaE_ciede2000(member_labs[m].reshape(1, 1, 3), member_labs.reshape(-1, 1, 3)).reshape(-1)
            costs.append(float(np.sum(d)))
        min_idx = int(np.argmin(np.array(costs)))
        k = int(member_idx[min_idx])
        reps.append(tuple(int(x) for x in colors[k]))
        rep_index_of_cluster.append(k)

    n_clusters = len(reps)
    kept_indices = list(range(n_clusters))
    pad_candidates = []
    if n_clusters > target_size:
        sizes = [len(clusters[i]) for i in range(n_clusters)]
        order = sorted(range(n_clusters), key=lambda i: -sizes[i])
        kept_indices = order[:target_size]
        pad_candidates = [reps[i] for i in order[target_size:]]

    kept_reps = [reps[i] for i in kept_indices]
    kept_rep_indices = [rep_index_of_cluster[i] for i in kept_indices]
    kept_rep_labs = labs[kept_rep_indices]

    color_map = {}
    for idx in range(len(colors)):
        lab_i = labs[idx]
        d = deltaE_ciede2000(kept_rep_labs.reshape(-1, 1, 3), lab_i.reshape(1, 1, 3)).reshape(-1)
        j = int(np.argmin(d))
        color_map[tuple(colors[idx])] = kept_reps[j]

    return kept_reps, color_map, pad_candidates


@njit(fastmath=True, cache=True)
def nearest_palette_index(color: np.ndarray, palette_int16: np.ndarray) -> int:
    """Return index of nearest palette color by Euclidean distance in RGB."""
    # Expect color as uint8 or int16; cast to int16 once
    c0 = int(color[0])
    c1 = int(color[1])
    c2 = int(color[2])
    best = 0
    bestd = np.int64(2_147_483_647)
    for i in range(palette_int16.shape[0]):
        dr = int(palette_int16[i, 0]) - c0
        dg = int(palette_int16[i, 1]) - c1
        db = int(palette_int16[i, 2]) - c2
        d = dr*dr + dg*dg + db*db
        if d < bestd:
            bestd = d
            best = i
    return best


@njit(fastmath=True, cache=True, parallel=True)
def nearest_indices_batch(cand_int16: np.ndarray, palette_int16: np.ndarray) -> np.ndarray:
    """Numba-parallel batch nearest search.

    Args:
        cand_int16: (M,3) int16 candidate colors.
        palette_int16: (P,3) int16 palette.
    Returns:
        (M,) int64 indices of nearest palette entries.
    """
    m = cand_int16.shape[0]
    p = palette_int16.shape[0]
    out = np.empty(m, dtype=np.int64)
    if p == 0:
        for i in prange(m):
            out[i] = 0
        return out
    for i in prange(m):
        c0 = cand_int16[i, 0]
        c1 = cand_int16[i, 1]
        c2 = cand_int16[i, 2]
        best = 0
        bestd = np.int64(2_147_483_647)
        for j in range(p):
            dr = palette_int16[j, 0] - c0
            dg = palette_int16[j, 1] - c1
            db = palette_int16[j, 2] - c2
            d = dr*dr + dg*dg + db*db
            if d < bestd:
                bestd = d
                best = j
        out[i] = best
    return out


def map_rgb_array_to_palette_indices(arr_rgb: np.ndarray, lut_exact: dict, palette_int16: np.ndarray) -> np.ndarray:
    """
    Map an RGB image array to palette indices using an optional exact LUT first, then nearest search.

    - Exact pass is now vectorized via packed 24-bit keys + searchsorted (avoids Python loop/dict lookups).
    - Nearest pass keeps chunking to bound memory; optionally uses a Numba batch kernel for large chunks.

    Args:
        arr_rgb: (H,W,3) uint8 array of image pixels.
        lut_exact: dict mapping exact RGB tuples -> palette index (int). Can be empty.
        palette_int16: (P,3) int16 palette array for nearest search.

    Returns:
        (H,W) uint16 array of palette indices.
    """
    if arr_rgb.size == 0:
        return np.empty(arr_rgb.shape[:2], dtype=np.uint16)

    h, w = arr_rgb.shape[:2]
    flat = arr_rgb.reshape(-1, 3)
    n = flat.shape[0]

    indices = np.zeros((n,), dtype=np.uint16)
    unmatched_mask = np.ones((n,), dtype=bool)

    # -------- Exact LUT pass (vectorized) --------
    if lut_exact:
        # Pack RGB -> uint32 key: r | (g<<8) | (b<<16)
        def pack_rgb_u32(a: np.ndarray) -> np.ndarray:
            a32 = a.astype(np.uint32, copy=False)
            return (a32[:, 0]) | (a32[:, 1] << 8) | (a32[:, 2] << 16)

        keys_flat = pack_rgb_u32(flat)

        # Build sorted key/value arrays from dict once per call
        lut_items = list(lut_exact.items())
        if lut_items:
            lut_keys = np.array([np.uint32(k[0] | (k[1] << 8) | (k[2] << 16)) for k, _ in lut_items], dtype=np.uint32)
            lut_vals = np.array([int(v) for _, v in lut_items], dtype=np.int64)
            order = np.argsort(lut_keys)
            lut_keys = lut_keys[order]
            lut_vals = lut_vals[order]

            idx = np.searchsorted(lut_keys, keys_flat)
            valid = (idx < lut_keys.size)
            valid_idx = idx[valid]
            match = np.zeros_like(valid, dtype=bool)
            if lut_keys.size > 0:
                match_valid = lut_keys[valid_idx] == keys_flat[valid]
                match[valid] = match_valid
            if match.any():
                indices[match] = lut_vals[idx[match]].astype(np.uint16)
                unmatched_mask[match] = False

    # -------- Nearest for unmatched --------
    if unmatched_mask.any():
        cand = flat[unmatched_mask]
        pal = palette_int16
        where_idx = np.where(unmatched_mask)[0]

        # Choose strategy: for very large chunks, prefer Numba kernel to reduce temporary allocations
        use_numba = True
        chunk = 200000  # process per chunk to bound memory
        start = 0
        while start < cand.shape[0]:
            end = min(start + chunk, cand.shape[0])
            sub = cand[start:end].astype(np.int16, copy=False)
            if use_numba:
                nearest = nearest_indices_batch(sub, pal)
            else:
                # Fallback: vectorized broadcasting (more memory)
                c = sub[:, None, :]
                diff = pal[None, :, :] - c
                d2 = np.sum(diff * diff, axis=2)
                nearest = np.argmin(d2, axis=1).astype(np.int64)
            indices[where_idx[start:end]] = nearest.astype(np.uint16)
            start = end

    return indices.reshape(h, w)


def _rgb_to_hsv_vec(rgb_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast vectorized RGB->HSV for arrays of shape (N,3) uint8 in [0,255].
    Returns H,S,V with H in [0,1), S in [0,1], V in [0,1].
    """
    rgb = rgb_uint8.astype(np.float32) / 255.0
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    diff = maxc - minc
    # Saturation
    s = np.zeros_like(v)
    nonzero = maxc > 0
    s[nonzero] = diff[nonzero] / maxc[nonzero]

    # Hue
    h = np.zeros_like(v)
    nz = diff > 1e-6
    # Avoid division by zero by only computing where diff>0
    rc = np.zeros_like(v)
    gc = np.zeros_like(v)
    bc = np.zeros_like(v)
    rc[nz] = (((g - b) / diff) % 6.0)[nz]
    gc[nz] = (((b - r) / diff) + 2.0)[nz]
    bc[nz] = (((r - g) / diff) + 4.0)[nz]

    is_r = (maxc == r) & nz
    is_g = (maxc == g) & nz
    is_b = (maxc == b) & nz
    h[is_r] = rc[is_r]
    h[is_g] = gc[is_g]
    h[is_b] = bc[is_b]
    h = (h / 6.0) % 1.0
    # For grayscale (s≈0) the hue is arbitrary; keep 0
    return h, s, v


def build_coarse_palette_index_map(palette_int16: np.ndarray,
                                   hue_bins: int = 12,
                                   levels_per_bin: int = 5,
                                   min_saturation: float = 0.05) -> np.ndarray:
    """
    Build a remapping from fine palette indices to a coarser set by collapsing indices within hue bins
    down to a limited number of value levels.

    Args:
        palette_int16: (P,3) int16 palette colors (0..255 values expected).
        hue_bins: number of hue bins to group colors.
        levels_per_bin: maximum number of value (brightness) levels preserved per hue bin.
        min_saturation: colors with saturation below this are treated as neutral (single shared bin).

    Returns:
        remap: (P,) uint16 array where remap[i] gives the representative index for original index i.
    """
    P = int(palette_int16.shape[0])
    if P == 0 or levels_per_bin <= 0:
        return np.arange(P, dtype=np.uint16)

    pal_u8 = np.clip(palette_int16, 0, 255).astype(np.uint8, copy=False)
    h, s, v = _rgb_to_hsv_vec(pal_u8)

    # Assign bins: grayscale/low-sat go to bin hue_bins (extra bin)
    bins = np.floor(h * hue_bins).astype(np.int32)
    bins[bins == hue_bins] = hue_bins - 1  # handle h==1 edge
    grey_bin = hue_bins  # extra bin at the end
    bins[s < min_saturation] = grey_bin

    # Prepare remap as identity
    remap = np.arange(P, dtype=np.uint16)

    # Process each bin separately
    total_bins = hue_bins + 1
    for bidx in range(total_bins):
        idxs = np.where(bins == bidx)[0]
        if idxs.size == 0:
            continue
        # Sort indices in this bin by value (brightness), then by saturation to keep more saturated reps
        order = np.lexsort((-s[idxs], v[idxs]))  # primarily by v ascending, then saturated first within ties
        sorted_idxs = idxs[order]
        if sorted_idxs.size <= levels_per_bin:
            # Nothing to collapse, keep identity
            continue
        # Split into equal segments by count and pick representative near the middle of each segment
        segments = np.array_split(sorted_idxs, levels_per_bin)
        reps = []
        for seg in segments:
            if seg.size == 0:
                continue
            # Prefer the most saturated color in the segment to preserve vivid hues (e.g., reds)
            seg_s = s[seg]
            # If all saturations are equal (e.g., greys), fall back to mid by index
            if np.any(seg_s > 0):
                # Choose the most saturated; if multiple, pick one whose value is closest to the segment's median V
                max_s = seg_s.max()
                cand_idx = seg[seg_s >= (max_s - 1e-6)]
                if cand_idx.size > 1:
                    seg_v = v[cand_idx]
                    med_v = np.median(v[seg])
                    pick = cand_idx[np.argmin(np.abs(seg_v - med_v))]
                else:
                    pick = cand_idx[0]
                reps.append(int(pick))
            else:
                # Low-saturation segment: use the middle element (by value order)
                mid = seg[seg.size // 2]
                reps.append(int(mid))
        reps = np.array(reps, dtype=np.int32)
        # Map every index in the bin to the nearest rep by value distance
        v_bin = v[sorted_idxs]
        v_reps = v[reps]
        # For each element, choose rep with minimum |v - v_rep|
        # Vectorized via broadcasting in manageable size (bin-only)
        diff = np.abs(v_bin[:, None] - v_reps[None, :])
        nearest = np.argmin(diff, axis=1)
        remap[sorted_idxs] = reps[nearest].astype(np.uint16)

    return remap


def map_rgb_array_to_palette_indices_coarse(arr_rgb: np.ndarray,
                                            lut_exact: dict,
                                            palette_int16: np.ndarray,
                                            hue_bins: int = 18,
                                            levels_per_hue: int = 6,
                                            min_saturation: float = 0.05) -> np.ndarray:
    """
    Alternative mapping that reduces greyscale complexity by collapsing palette indices per hue.

    Steps:
      1) Use the standard map_rgb_array_to_palette_indices to get fine indices.
      2) Build a remap that limits the number of distinct indices per hue bin to levels_per_hue
         (based on HSV value), then apply it to the index image.

    Args:
        arr_rgb: (H,W,3) uint8 image array.
        lut_exact: dict of exact RGB->index matches.
        palette_int16: (P,3) int16 palette used for nearest matching.
        hue_bins: number of hue groups for collapsing (e.g., 12).
        levels_per_hue: max distinct greyscale values preserved per hue group (e.g., 5).
        min_saturation: threshold to treat colors as neutral (grouped together).

    Returns:
        (H,W) uint16 array of coarsened palette indices.
    """
    fine = map_rgb_array_to_palette_indices(arr_rgb, lut_exact, palette_int16)
    remap = build_coarse_palette_index_map(palette_int16, hue_bins=hue_bins,
                                           levels_per_bin=levels_per_hue,
                                           min_saturation=min_saturation)
    return remap[fine]


def perceptual_color_sort(colors):
    if(cfg.get(cfg.ci_use_faster_sort)):
        return lightness_sort(colors)
    else:
        return perceptual_color_sort_updated(colors)


def adjacency_aware_color_sort(colors: list | np.ndarray,
                                quantized_array: np.ndarray,
                                lambda_de: float = 0.7,
                                connectivity: int = 8) -> list:
    """Order colors so that adjacent indices are more likely to be neighbors in the quantized image while
    still maintaining perceptual smoothness.

    Args:
        colors: list of RGB tuples or (N,3) array of uint8 unique colors.
        quantized_array: HxWx3 uint8 array of the quantized source image.
        lambda_de: blend factor in [0,1]; higher values weight perceptual distance more strongly.
        connectivity: neighborhood (4 or 8) for adjacency.

    Returns:
        A list of RGB tuples in computed order.
    """
    if colors is None or len(colors) == 0:
        return []
    cols = np.array(colors, dtype=np.uint8)
    K = cols.shape[0]
    keys = [tuple(int(c) for c in row) for row in cols]

    # Map pixels to color indices using a packed RGB LUT
    h, w = quantized_array.shape[:2]
    valid_mask = np.ones((h, w), dtype=np.uint8)

    flat = quantized_array.reshape(-1, 3)
    packed = (flat[:, 0].astype(np.uint32) << 16) | (flat[:, 1].astype(np.uint32) << 8) | flat[:, 2].astype(np.uint32)
    lut = {((int(c[0]) << 16) | (int(c[1]) << 8) | int(c[2])): i for i, c in enumerate(cols.tolist())}
    ids = np.fromiter((lut.get(int(v), -1) for v in packed), dtype=np.int32, count=packed.shape[0]).reshape(h, w)

    # Build symmetric co-occurrence matrix from neighbor pairs
    co = np.zeros((K, K), dtype=np.float64)

    def add_pair(a, b):
        if a < 0 or b < 0 or a == b:
            return
        co[a, b] += 1.0
        co[b, a] += 1.0

    if connectivity == 8:
        neighbors = [(0, 1), (1, 0), (1, 1), (1, -1)]
    else:
        neighbors = [(0, 1), (1, 0)]

    vm = valid_mask
    for y in range(h):
        for x in range(w):
            if vm[y, x] == 0:
                continue
            a = ids[y, x]
            if a < 0:
                continue
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and vm[ny, nx] != 0:
                    b = ids[ny, nx]
                    add_pair(a, b)

    # Normalize adjacency scores to [0,1]
    if np.max(co) > 0:
        A = co / np.max(co)
    else:
        # No adjacency info; fall back
        return perceptual_color_sort(colors)

    # Compute perceptual distances (ΔE00) between colors; normalize to [0,1]
    labs = rgb2lab(cols.reshape(-1, 1, 3)).reshape(-1, 3)
    D = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        di = deltaE_ciede2000(labs[i].reshape(1, 1, 3), labs.reshape(-1, 1, 3)).reshape(-1)
        D[i] = di
    if np.max(D) > 0:
        Dn = D / np.max(D)
    else:
        Dn = D

    # Greedy path construction with blended cost: cost = λ*Dn + (1-λ)*(1-A)
    remaining = set(range(K))
    # Start with lowest L to align with greyscale intuition
    start = int(np.argmin(labs[:, 0]))
    order = [start]
    remaining.remove(start)
    while remaining:
        i = order[-1]
        rem = np.array(sorted(list(remaining)))
        costs = lambda_de * Dn[i, rem] + (1.0 - lambda_de) * (1.0 - A[i, rem])
        j = int(rem[int(np.argmin(costs))])
        order.append(j)
        remaining.remove(j)

    return [keys[i] for i in order]

def adjacency_from_p_mode(p_img: Image.Image, connectivity=8):
    """Build adjacency (co-occurrence) matrix from P-mode quantized image."""
    assert p_img.mode == "P"
    pal = _get_palette_array(p_img)
    arr = np.array(p_img)
    h, w = arr.shape
    K = len(pal)

    co = np.zeros((K, K), dtype=np.float64)

    if connectivity == 8:
        neighbors = [(0, 1), (1, 0), (1, 1), (1, -1)]
    else:
        neighbors = [(0, 1), (1, 0)]

    for y in range(h):
        for x in range(w):
            a = arr[y, x]
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    b = arr[ny, nx]
                    if a != b:
                        co[a, b] += 1
                        co[b, a] += 1

    return co

def adjacency_aware_color_sort_pmode(p_img: Image.Image, lambda_de=0.4):
    """Sort colors based on spatial adjacency and perceptual similarity."""
    pal, co = adjacency_from_p_mode(p_img)
    K = len(pal)
    labs = rgb2lab(pal.reshape(-1, 1, 3)).reshape(-1, 3)

    # Normalize adjacency (after applying frequency weighting in adjacency_from_p_mode)
    A = co / np.max(co) if np.max(co) > 0 else co

    # Compute perceptual distances (ΔE00)
    D = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        D[i] = deltaE_ciede2000(labs[i].reshape(1, 1, 3), labs.reshape(-1, 1, 3)).reshape(-1)
    Dn = D / np.max(D) if np.max(D) > 0 else D

    # Start with darkest color
    start = int(np.argmin(labs[:, 0]))
    remaining = set(range(K))
    order = [start]
    remaining.remove(start)

    # Greedy adjacency-perceptual path
    while remaining:
        i = order[-1]
        rem = np.array(sorted(list(remaining)))
        cost = lambda_de * Dn[i, rem] + (1 - lambda_de) * (1 - A[i, rem])
        j = int(rem[int(np.argmin(cost))])
        order.append(j)
        remaining.remove(j)

    # ✅ Return list of (R,G,B) tuples
    return [tuple(map(int, pal[i])) for i in order]


def old_perceptual_color_sort(colors):
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

@njit(fastmath=True, cache=True, parallel=True)
def delta_e_matrix_numba(lab):
    n = lab.shape[0]
    D = np.empty((n, n), dtype=np.float32)
    for i in prange(n):
        li0 = lab[i, 0]
        li1 = lab[i, 1]
        li2 = lab[i, 2]
        for j in range(n):
            dl = li0 - lab[j, 0]
            da = li1 - lab[j, 1]
            db = li2 - lab[j, 2]
            D[i, j] = (dl*dl + da*da + db*db) ** 0.5
    return D


def lightness_sort(colors):
    n = len(colors)
    if n <= 1:
        return list(colors)

    lab = rgb_to_lab_array(colors)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

    # Calculate hues and handle grayscale gracefully
    hues = np.zeros_like(L)
    chromas = np.sqrt(a ** 2 + b ** 2)
    non_gray = chromas > 0.1  # Avoid hue calculation for near-grayscale
    hues[non_gray] = np.array([
        hue_angle_from_ab(ai, bi)
        for ai, bi in zip(a[non_gray], b[non_gray])
    ])

    # ALWAYS prioritize lightness for linear gradients
    initial_order = np.lexsort((hues, L))  # Primary: L*, Secondary: hue

    sorted_colors = [colors[i] for i in initial_order]
    lab_current = lab[initial_order]

    # Calculate adjacent ΔE for logging (no 2-opt to preserve lightness order)
    total_de = 0.0
    max_de = 0.0
    for i in range(n - 1):
        dl = lab_current[i, 0] - lab_current[i + 1, 0]
        da = lab_current[i, 1] - lab_current[i + 1, 1]
        db = lab_current[i, 2] - lab_current[i + 1, 2]
        de = math.sqrt(dl * dl + da * da + db * db)
        total_de += de
        if de > max_de:
            max_de = de

    avg_de = total_de / (n - 1) if n > 1 else 0
    logger.debug(f"Linear gradient sort complete (lightness-primary): {n} colors")
    logger.debug(f"Total ΔE: {total_de:.3f}, Avg ΔE: {avg_de:.3f}, Max ΔE: {max_de:.3f}")

    return sorted_colors

def perceptual_color_sort_updated(colors):
    n = len(colors)
    if n <= 1:
        return list(colors)

    lab = rgb_to_lab_array(colors)

    print("Color analysis:")
    for i, color in enumerate(colors):
        l, a, b = lab[i]
        hue = hue_angle_from_ab(a, b)
        print(f"Color {i}: RGB{color} -> Lab({l:.1f}, {a:.1f}, {b:.1f}) -> Hue: {hue:.1f}°")

    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

    hues = np.array([hue_angle_from_ab(ai, bi) for ai, bi in zip(a, b)])
    chromas = np.sqrt(a**2 + b**2)

    hues_unwrapped, _ = unwrap_hue(hues)
    hue_range = hues_unwrapped.max() - hues_unwrapped.min()
    is_hue_dominant = hue_range > 90

    if is_hue_dominant:
        initial_order = np.lexsort((L, hues_unwrapped))
    else:
        initial_order = np.lexsort((chromas, L))

    sorted_colors = [colors[i] for i in initial_order]
    lab_current = lab[initial_order]

    # --- Precompute distance matrix (n x n) ---
    dist_matrix = delta_e_matrix_numba(lab_current)

    # --- 2-opt with O(1) cost diff calculation ---
    order = np.arange(n)
    current_cost = np.sum(dist_matrix[order[:-1], order[1:]])

    improved = True
    iterations = 0
    max_polish_iter = min(cfg.get(cfg.ci_palette_color_iteration), n * 5)

    while improved and iterations < max_polish_iter:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 2, n - 1):

                # Cost difference of swapping (i-1,i) and (j,j+1)
                before = dist_matrix[order[i-1], order[i]] + dist_matrix[order[j], order[j+1]]
                after  = dist_matrix[order[i-1], order[j]] + dist_matrix[order[i], order[j+1]]

                if after < before:  # real improvement detected
                    order[i:j+1] = order[i:j+1][::-1]  # reverse segment
                    current_cost += (after - before)
                    improved = True
                    break
            if improved:
                break
        iterations += 1

    # Apply result order
    sorted_colors = [sorted_colors[i] for i in order]
    lab_current = lab_current[order]

    total_de = np.sum(dist_matrix[order[:-1], order[1:]])
    max_de = np.max(dist_matrix[order[:-1], order[1:]])

    logger.debug(f"Linear gradient sort complete (fast 2-opt, same quality): {n} colors")
    logger.debug(f"Total ΔE: {total_de:.3f}, Avg ΔE: {total_de/(n-1):.3f}, Max ΔE: {max_de:.3f}")

    return sorted_colors


@njit(fastmath=True, cache=True)
def two_opt_with_delta(order, D, max_iter):
    m = len(order)
    if m < 4 or max_iter <= 0:
        return order
    for _ in range(max_iter):
        improved = False
        for i in range(1, m-2):
            a = order[i-1]
            b = order[i]
            for j in range(i+1, m-1):
                c = order[j]
                d = order[j+1]
                delta = -D[a,b] - D[c,d] + D[a,c] + D[b,d]
                if delta < -1e-6:
                    # reverse section i:j+1 (Numba-safe manual reverse)
                    left = i
                    right = j
                    while left < right:
                        order[left], order[right] = order[right], order[left]
                        left += 1
                        right -= 1
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return order

def faster_perceptual_color_sort(colors):
    """
    Linear perceptual color sort that creates a smooth gradient by:
    - Unwrapping hue to avoid wrap-around artifacts
    - Choosing an initial sort based on hue dominance (your existing policy)
    - Optimizing with 2-opt using precomputed CIE76 distances and O(1) delta updates
    """
    n = len(colors)
    if n <= 1:
        return list(colors)

    # Convert once
    lab = rgb_to_lab_array(colors)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

    # Keep your hue/chroma logic; if you prefer, you can vectorize hue as below without changing behavior:
    # hues = (np.degrees(np.arctan2(b, a)) + 360.0) % 360.0
    hues = np.array([hue_angle_from_ab(ai, bi) for ai, bi in zip(a, b)])
    chromas = np.sqrt(a * a + b * b)

    # --- Initial ordering (unchanged policy), but try both L secondary directions and pick cheaper ---
    hues_unwrapped, _ = unwrap_hue(hues)
    hue_range = float(hues_unwrapped.max() - hues_unwrapped.min())
    is_hue_dominant = hue_range > 90.0

    if is_hue_dominant:
        cand1 = np.lexsort((L, hues_unwrapped))           # lightness ascending within hue
        cand2 = np.lexsort((-L, hues_unwrapped))          # lightness descending within hue
    else:
        cand1 = np.lexsort((chromas, L))                  # L asc within narrow hue
        cand2 = np.lexsort((chromas, -L))                 # L desc within narrow hue

    # Precompute Euclidean (CIE76) pairwise distances once on the fixed lab array
    D = delta_e_matrix_numba(lab)

    def path_cost_from_order(idx):
        if len(idx) <= 1:
            return 0.0
        return float(np.sum(D[idx[:-1], idx[1:]]))

    c1 = path_cost_from_order(cand1)
    c2 = path_cost_from_order(cand2)
    initial_order = cand1 if c1 <= c2 else cand2

    # Materialize sorted copies
    sorted_colors = [colors[i] for i in initial_order]
    lab_current = lab[initial_order].copy()

    # Rebuild D relative to lab_current to simplify indices 0..n-1 for the 2-opt step
    D_curr = delta_e_matrix_numba(lab_current)

    # 2-opt with O(1) delta evaluation, first-improvement, absolute-epsilon acceptance
    order = list(range(n))

    # Decide polishing budget (keep your config, but it’s much cheaper now)
    try:
        max_polish_iter = int(min(cfg.get(cfg.ci_palette_color_iteration), n * 5))
    except Exception:
        max_polish_iter = min(16, max(1, n // 4))

    if max_polish_iter > 0:
        order = np.array(list(range(n)), dtype=np.int32)
        order = two_opt_with_delta(order, D_curr, max_polish_iter)
        order = order.tolist()
        # Apply final order
        if order != list(range(n)):
            sorted_colors = [sorted_colors[i] for i in order]
            lab_current = lab_current[order]

    # Metrics & logging
    total_de = float(np.sum(np.linalg.norm(lab_current[1:] - lab_current[:-1], axis=1)))
    max_de = float(np.max(np.linalg.norm(lab_current[1:] - lab_current[:-1], axis=1))) if n > 1 else 0.0

    method = ("hue-guided" if is_hue_dominant else "L/C-guided") + ("+2opt" if max_polish_iter > 0 else "")
    logger.debug(f"Linear gradient sort complete ({method}): {n} colors")
    logger.debug(f"Total ΔE: {total_de:.3f}, Average ΔE: {total_de / (n - 1) if n > 1 else 0:.3f}, Max ΔE: {max_de:.3f}")

    start_lab = lab_current[0]
    end_lab = lab_current[-1]
    logger.debug(f"Gradient range: L{start_lab[0]:.1f}→L{end_lab[0]:.1f}")
    start_hue = hue_angle_from_ab(start_lab[1], start_lab[2])
    end_hue = hue_angle_from_ab(end_lab[1], end_lab[2])
    logger.debug(f"Hue range: {start_hue:.1f}° → {end_hue:.1f}°")

    return sorted_colors

# === Row-building and composition helpers consolidated for widgets and generator ===
def extract_existing_palette_rows(path: str, row_height: int, skip_grey_rows: bool = True) -> tuple[list[np.ndarray], int]:
    """Extract existing non-grey palette rows from a palette image file.

    - Returns (rows, palette_size). Each row is (palette_size, 3) uint8.
    - If no non-grey rows are detected, returns a single row sampled from the middle.
    """
    pil = load_image(path)
    arr = np.array(pil)
    h, w = arr.shape[:2]
    palette_size = w
    if row_height <= 0:
        row_height = 1
    num_blocks = max(1, h // row_height)

    rows: list[np.ndarray] = []
    for block in range(num_blocks):
        start_row = block * row_height
        y = min(start_row, h - 1)
        row = arr[y, :w, :]
        if skip_grey_rows:
            eq = (row[:, 0] == row[:, 1]) & (row[:, 1] == row[:, 2])
            frac_grey = float(np.mean(eq))
            if frac_grey >= 0.9:
                continue
        rows.append(row[:palette_size].astype(np.uint8))

    if not rows:
        # Fallback: choose middle row
        y = h // 2
        row = arr[y, :w, :]
        rows = [row[:palette_size].astype(np.uint8)]
    return rows, palette_size


def _positions_by_greyscale_index(greyscale_indices: np.ndarray, palette_size: int) -> dict[int, np.ndarray]:
    pos = {}
    for g in range(palette_size):
        positions = np.argwhere(greyscale_indices == g)
        pos[g] = positions
    return pos


def _choose_dominant_color(colors_at_positions: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (chosen_color_rgb, values_counts_sorted) where values_counts_sorted is a tuple
    (values_sorted, counts_sorted). If no colors, returns (None, None)."""
    if colors_at_positions is None or colors_at_positions.size == 0:
        return None, None
    tuples = [tuple(c) for c in colors_at_positions]
    if not tuples:
        return None, None
    values, counts = np.unique(np.array(tuples), axis=0, return_counts=True)
    order = np.argsort(-counts)
    values_sorted = values[order]
    counts_sorted = counts[order]
    chosen = values_sorted[0]
    return chosen, (values_sorted, counts_sorted)


def build_row_from_arrays(greyscale_indices: np.ndarray,
                          rgb_array: np.ndarray,
                          base_row: np.ndarray,
                          palette_size: int,
                          log_top_k: int = 3,
                          context_label: str | None = None) -> np.ndarray:
    """Build a single palette row by mapping greyscale indices to dominant colors from an RGB array.

    Also logs deviations from base_row using the consolidated format.
    """
    height_g, width_g = greyscale_indices.shape
    # Ensure dimensions match; if not, resize nearest
    if rgb_array.shape[:2] != (height_g, width_g):
        try:
            img = Image.fromarray(rgb_array.astype('uint8'), 'RGB')
            img = img.resize((width_g, height_g), Image.Resampling.NEAREST)
            rgb_array = np.array(img)
        except Exception:
            pass

    positions_by_g = _positions_by_greyscale_index(greyscale_indices, palette_size)

    row_palette = np.copy(base_row)
    deviation_count = 0

    for g in range(palette_size):
        positions = positions_by_g.get(g)
        if positions is None or positions.size == 0:
            continue
        colors = rgb_array[positions[:, 0], positions[:, 1]]
        chosen, vc = _choose_dominant_color(colors)
        if chosen is None:
            continue
        base_color = row_palette[g]

        # Decide whether to keep original base color if the new dominant color is very close
        chosen_final = chosen
        do_log = False
        de_val = None
        try:
            use_orig = bool(cfg.get(cfg.ci_palette_use_original_colors)) if hasattr(cfg, 'ci_palette_use_original_colors') else False
        except Exception:
            use_orig = False
        if use_orig and not np.array_equal(chosen, base_color):
            try:
                # Compute ΔE00 between base and chosen
                cols = np.array([base_color, np.array(chosen, dtype=np.uint8)], dtype=np.uint8)
                labs = rgb2lab(cols.reshape(-1, 1, 3)).reshape(-1, 3)
                de_val = float(deltaE_ciede2000(labs[0].reshape(1, 1, 3), labs[1].reshape(1, 1, 3)).reshape(-1)[0])
            except Exception:
                de_val = None
            try:
                thr = float(cfg.get(cfg.ci_palette_original_max_de)) if hasattr(cfg, 'ci_palette_original_max_de') else 2.0
            except Exception:
                thr = 2.0
            if de_val is not None and de_val <= thr:
                # Keep the original palette color; do not count as deviation
                chosen_final = base_color
                do_log = False
            else:
                # Replace and log deviation
                chosen_final = chosen
                do_log = True
        else:
            # Either using originals is disabled or colors exactly match
            do_log = not np.array_equal(chosen, base_color)

        if do_log and cfg.get(cfg.ci_extra_logging):
            deviation_count += 1
            try:
                if vc is not None and log_top_k > 0:
                    values_sorted, counts_sorted = vc
                    top_k = int(min(log_top_k, len(values_sorted)))
                    top_entries = [
                        (values_sorted[j].tolist(), int(counts_sorted[j])) for j in range(top_k)
                    ]
                else:
                    top_entries = []
                label = f"{context_label}: " if context_label else ""
                if de_val is not None:
                    logger.debug(
                        f"{label}Index {g:3d}: base={base_color.tolist()} -> new={np.array(chosen).tolist()} | "
                        f"ΔE00={de_val:.2f} | pixels={int(colors.shape[0])} | top={top_entries}"
                    )
                else:
                    logger.debug(
                        f"{label}Index {g:3d}: base={base_color.tolist()} -> new={np.array(chosen).tolist()} | "
                        f"pixels={int(colors.shape[0])} | top={top_entries}"
                    )
            except Exception:
                pass
        row_palette[g] = chosen_final

    try:
        if context_label:
            if deviation_count > 0:
                logger.debug(f"Row summary for {context_label}: deviations={deviation_count} of {palette_size}")
        else:
            if deviation_count > 0:
                logger.debug(f"Row summary: deviations={deviation_count} of {palette_size}")

    except Exception:
        pass

    return row_palette


def build_row_from_pair(greyscale_path: str,
                        color_path: str,
                        base_row: np.ndarray,
                        palette_size: int,
                        quant_method: QuantAlgorithm,
                        log_top_k: int = 3) -> np.ndarray:
    """Load images from disk (greyscale, color), quantize color, and build a palette row.
    Logs with a header and per-index deviations using consolidated logic.
    """
    # Load greyscale indices
    pil_g = load_image(greyscale_path, format='RGB')
    arr_g = np.array(pil_g)
    greyscale_indices = np.clip(arr_g[:, :, 0].astype(np.int32), 0, palette_size - 1)

    # Load+quantize color image
    pil_img = load_image(color_path, format='RGB')
    ex_quantized, _ = quantize_image(pil_img, quant_method)
    ex_quant_rgb = ex_quantized.convert('RGB')
    arr_rgb = np.array(ex_quant_rgb)

    # Header
    try:
        logger.debug(
            f"Building LUT row: grey={os.path.basename(greyscale_path)}, color={os.path.basename(color_path)}, palette_size={palette_size}"
        )
    except Exception:
        pass

    return build_row_from_arrays(
        greyscale_indices=greyscale_indices,
        rgb_array=arr_rgb,
        base_row=base_row,
        palette_size=palette_size,
        log_top_k=log_top_k,
        context_label=os.path.basename(color_path)
    )


def compose_palette_image(rows: list[np.ndarray], row_height: int, palette_size: int, pad_mode: str = 'none') -> Image.Image:
    """Compose a palette image from a list of row arrays (each (palette_size,3)).

    pad_mode:
      - 'none': pad with zeros
      - 'gradient': fill remaining rows to power-of-two height with a greyscale gradient
    """
    existing_rows = rows or []
    num_blocks = len(existing_rows)
    required_height = row_height * num_blocks
    palette_height = int(next_power_of_2(required_height))
    palette_width = palette_size

    palette_image_array = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
    for block_idx, block_array in enumerate(existing_rows):
        start_row = block_idx * row_height
        end_row = start_row + row_height
        for row in range(start_row, end_row):
            for col in range(palette_width):
                if col < palette_size:
                    palette_image_array[row, col] = block_array[col]
                else:
                    palette_image_array[row, col] = [0, 0, 0]

    filled_rows = row_height * num_blocks
    if pad_mode == 'gradient' and filled_rows < palette_height:
        for row in range(filled_rows, palette_height):
            for col in range(palette_width):
                grey_value = int(col * (255 / (palette_width - 1))) if palette_width > 1 else 0
                palette_image_array[row, col] = [grey_value, grey_value, grey_value]

    return Image.fromarray(palette_image_array.astype('uint8'), 'RGB')


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    sigma = float(max(0.0, sigma))
    if sigma == 0.0:
        return np.array([1.0], dtype=np.float32)
    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k.astype(np.float32)


def _convolve1d_reflect(arr: np.ndarray, kernel: np.ndarray, axis: int = 0) -> np.ndarray:
    if arr.ndim == 1:
        a = arr
        pad = len(kernel) // 2
        if pad == 0:
            return a.copy()
        a_pad = np.pad(a, (pad, pad), mode='reflect')
        out = np.convolve(a_pad, kernel[::-1], mode='valid')
        return out.astype(a.dtype)
    # Move axis to front
    arr_move = np.moveaxis(arr, axis, 0)
    out = np.empty_like(arr_move, dtype=np.float32)
    pad = len(kernel) // 2
    for idx in range(arr_move.shape[1] if arr_move.ndim > 1 else 1):
        # Slice all remaining axes at idx using ... trick
        line = arr_move[:, idx] if arr_move.ndim > 1 else arr_move
        line_pad = np.pad(line, (pad, pad), mode='reflect')
        conv = np.convolve(line_pad.astype(np.float32), kernel[::-1], mode='valid')
        out[:, idx] = conv.astype(np.float32)
    out = np.moveaxis(out, 0, axis)
    return out.astype(arr.dtype)


def upscale_and_smooth_lut(sorted_colors: list | np.ndarray, target_size: int = 256, sigma: float = 1.0) -> np.ndarray:
    """Upscale a color LUT (list/array of RGB) to target_size using linear interpolation per channel
    followed by light 1D Gaussian smoothing along the LUT axis.

    Args:
      sorted_colors: sequence of RGB colors (N,3), uint8 preferred.
      target_size: desired output length (default 256).
      sigma: Gaussian sigma in color index units (default 1.0). If 0, no smoothing.

    Returns:
      colors_up: (target_size,3) uint8 array.
    """
    if sorted_colors is None:
        return np.zeros((target_size, 3), dtype=np.uint8)
    cols = np.array(sorted_colors, dtype=np.float32)
    if cols.ndim != 2 or cols.shape[1] != 3 or cols.shape[0] == 0:
        return np.zeros((target_size, 3), dtype=np.uint8)
    n = int(cols.shape[0])
    if n == target_size:
        out = cols.copy()
    elif n == 1:
        out = np.tile(cols[0:1], (target_size, 1))
    else:
        x_old = np.linspace(0.0, 1.0, n, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, int(target_size), dtype=np.float32)
        out = np.zeros((int(target_size), 3), dtype=np.float32)
        for c in range(3):
            out[:, c] = np.interp(x_new, x_old, cols[:, c])
    # Smoothing
    sig = float(sigma)
    if sig > 0.0:
        k = _gaussian_kernel_1d(sig)
        # Convolve along axis 0 (over entries), each channel independently
        for c in range(3):
            out[:, c] = _convolve1d_reflect(out[:, c], k, axis=0)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)

