import os
import re
from collections import Counter

import numpy as np
from PIL import Image
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget, QLabel, QFileDialog, QMessageBox
from qfluentwidgets import (
    FluentIcon as FIF,
    PrimaryPushButton,
    PushSettingCard,
    SwitchSettingCard
)
from skimage.color import deltaE_ciede2000

from help.bulkpalette_help import BulkPaletteHelp
from settings.palette_settings import PaletteSettings
from src.palette.palette_engine import (
    quantize_image,
    perceptual_color_sort,
    next_power_of_2,
    pad_colors_to_target,
    convert_to_dds,
    load_image,
    rgb_to_lab_array,
    hue_angle_from_ab,
)
from src.utils.appconfig import cfg
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger
from utils.cards import TextSettingCard

SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tif', '.tiff', '.dds'}


class BulkPaletteWorker(QThread):
    progress = Signal(int, str)
    error = Signal(str)
    completed = Signal(dict)

    def __init__(self, directory, suffix_filter, output_dir, generate_dds=False, working_resolution=None, palette_size=256, exclude_filter: str = "", group_name: str = ""):
        super().__init__()
        self.directory = directory
        self.suffix_filter = suffix_filter
        self.exclude_filter = (exclude_filter or "").strip()
        self.output_dir = output_dir or directory
        self.generate_dds = generate_dds
        self.working_resolution = working_resolution
        self.palette_size = palette_size
        self.group_name = self._sanitize_for_filename((group_name or "").strip())

    def run(self):
        try:
            self.progress.emit(1, "Scanning directory...")
            image_paths = self._collect_images()
            if not image_paths:
                raise Exception("No images found matching the include filter in the selected directory.")

            # Collect original extensions present to decide output formats (match input types per file)
            original_exts = [os.path.splitext(p)[1].lower() for p in image_paths]
            originals_have_dds = any(ext == '.dds' for ext in original_exts)
            # If any DDS will be written, ensure texconv is available
            if originals_have_dds:
                texconv_path = cfg.get(cfg.texconv_file)
                if not texconv_path or not os.path.isfile(texconv_path):
                    raise Exception("texconv.exe path is not set. Please configure it in Settings before writing DDS outputs.")

            self.progress.emit(5, f"Found {len(image_paths)} images. Loading and quantizing...")

            # Step 1: Quantize each image and collect color frequencies
            per_image_quant = []
            global_counter = Counter()
            color_image_coverage = Counter()  # counts in how many distinct images a color appears
            sample_img_for_padding = None

            for idx, path in enumerate(image_paths):
                img = load_image(path, cfg.get(cfg.texconv_file))
                if self.working_resolution and self.working_resolution > 0:
                    img = self._downscale_keep_aspect(img, self.working_resolution)
                if sample_img_for_padding is None:
                    sample_img_for_padding = img

                quantized_img, qinfo = quantize_image(img, cfg.get(cfg.ci_default_quant_method))
                q_rgb = quantized_img.convert('RGB')
                q_arr = np.array(q_rgb)
                h, w = q_arr.shape[:2]

                colors, counts = np.unique(q_arr.reshape(-1, 3), axis=0, return_counts=True)
                local_counter = Counter({tuple(c): int(n) for c, n in zip(colors, counts)})
                global_counter.update(local_counter)
                # Update image coverage (each color counted once per image)
                for lc in local_counter.keys():
                    color_image_coverage[tuple(int(x) for x in lc)] += 1

                per_image_quant.append({
                    'path': path,
                    'array_rgb': q_arr,
                    'size': (w, h),
                    'local_counter': local_counter
                })

                pct = 5 + int((idx + 1) / max(1, len(image_paths)) * 45)
                self.progress.emit(pct, f"Quantized {os.path.basename(path)} with {len(local_counter)} colors")

            self.progress.emit(50, f"Merging {len(global_counter)} colors down to ≤ {self.palette_size}...")

            # Step 2: Build global palette ≤ palette_size, merging by proximity preferring frequent colors
            palette_colors, color_map, pad_candidates = self._build_global_palette(global_counter, target_size=self.palette_size, color_coverage=color_image_coverage)

            # Step 3: Ensure exactly palette_size (pad if needed), and sort perceptually
            colors_np = np.array(palette_colors, dtype=np.uint8)
            if len(colors_np) < self.palette_size:
                # Prefer to pad using high-scoring dropped clusters to preserve diversity
                try:
                    if pad_candidates:
                        seen = {tuple(map(int, c)) for c in colors_np.tolist()}
                        to_add = []
                        for c in pad_candidates:
                            t = tuple(int(x) for x in c)
                            if t not in seen:
                                to_add.append(t)
                                seen.add(t)
                                if len(colors_np) + len(to_add) >= self.palette_size:
                                    break
                        if to_add:
                            colors_np = np.vstack([colors_np, np.array(to_add, dtype=np.uint8)])
                    # If still short, fall back to existing padding strategies
                    if len(colors_np) < self.palette_size:
                        colors_np = pad_colors_to_target(colors_np, sample_img_for_padding, self.palette_size)
                except Exception as e:
                    logger.warning(f"Padding palette failed: {e}")
                    # Fallback pad by repeating
                    while len(colors_np) < self.palette_size:
                        colors_np = np.vstack([colors_np, colors_np[:max(1, self.palette_size - len(colors_np))]])
                    colors_np = colors_np[:self.palette_size]

            # sort perceptually
            sorted_colors = perceptual_color_sort([tuple(c) for c in colors_np])
            palette_np = np.array(sorted_colors, dtype=np.uint8)

            # Step 4: Remap each quantized image colors to palette indices, and save greyscale
            self.progress.emit(70, "Remapping images to palette indices...")
            palette_index_lut = {tuple(color): i for i, color in enumerate(palette_np.tolist())}

            # Precompute palette for nearest search
            pal_float = palette_np.astype(np.int16)

            results_greyscale = []
            for i, item in enumerate(per_image_quant):
                arr = item['array_rgb']

                # Constrain mapping to each image's OWN colors only.
                # Build a per-image exact LUT: local quantized color -> global sorted palette index
                lut_exact = {}
                local_colors = item.get('local_counter', {}).keys()
                for lc in local_colors:
                    rep = color_map.get(tuple(lc))
                    if rep is None:
                        # Fallback: find nearest in palette for unexpected colors (should be rare)
                        nearest_idx = self._nearest_palette_index(np.array(lc, dtype=np.uint8), pal_float)
                        lut_exact[tuple(int(x) for x in lc)] = int(nearest_idx)
                    else:
                        lut_exact[tuple(int(x) for x in lc)] = int(palette_index_lut[tuple(rep)])

                # Apply mapping using helper (handles any stray unmatched colors via nearest search)
                grey_indices = self._map_rgb_array_to_palette_indices(arr, lut_exact, pal_float).astype(np.uint8)

                # Determine output extension to match source
                src_ext = os.path.splitext(item['path'])[1].lower()
                rel_name = os.path.splitext(os.path.basename(item['path']))[0]

                # Incorporate optional group name into base filename
                base_with_group = rel_name
                if self.group_name:
                    base_with_group = f"{rel_name}_{self.group_name}"

                # Save greyscale (L) matching input type; for DDS, convert via temp PNG
                grey_base = f"{base_with_group}_greyscale"
                color_base = f"{base_with_group}_quant"

                if src_ext == '.dds':
                    # write temporary PNGs then convert
                    tmp_grey_png = os.path.join(self.output_dir, grey_base + '.png')
                    Image.fromarray(grey_indices, mode='L').save(tmp_grey_png)
                    grey_out = os.path.join(self.output_dir, grey_base + '.dds')
                    try:
                        convert_to_dds(tmp_grey_png, grey_out, cfg.get(cfg.texconv_file), is_palette=False)
                    finally:
                        try:
                            os.remove(tmp_grey_png)
                        except Exception:
                            pass

                    color_arr = palette_np[grey_indices]
                    tmp_color_png = os.path.join(self.output_dir, color_base + '.png')
                    Image.fromarray(color_arr.astype(np.uint8), mode='RGB').save(tmp_color_png)
                    color_out = os.path.join(self.output_dir, color_base + '.dds')
                    try:
                        convert_to_dds(tmp_color_png, color_out, cfg.get(cfg.texconv_file), is_palette=False)
                    finally:
                        try:
                            os.remove(tmp_color_png)
                        except Exception:
                            pass
                else:
                    # non-DDS: save directly using the same extension
                    grey_out = os.path.join(self.output_dir, grey_base + src_ext)
                    Image.fromarray(grey_indices, mode='L').save(grey_out)
                    color_arr = palette_np[grey_indices]
                    color_out = os.path.join(self.output_dir, color_base + src_ext)
                    Image.fromarray(color_arr.astype(np.uint8), mode='RGB').save(color_out)

                results_greyscale.append({
                    'source': item['path'],
                    'greyscale_path': grey_out,
                    'color_path': color_out,
                })

                pct = 70 + int((i + 1) / max(1, len(per_image_quant)) * 20)
                self.progress.emit(pct, f"Saved greyscale and colorized for {os.path.basename(item['path'])}")

            # Step 5: Save palette image matching input type rule
            self.progress.emit(92, "Saving palette image...")
            palette_width = self.palette_size
            palette_row_height = cfg.get(cfg.ci_palette_row_height) if hasattr(cfg, 'ci_palette_row_height') else 8
            palette_height = next_power_of_2(palette_row_height)
            pal_img_arr = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
            for row in range(palette_height):
                pal_img_arr[row, :palette_width] = palette_np

            palette_img = Image.fromarray(pal_img_arr, 'RGB')

            # Decide palette filename and extension
            folder_name = os.path.basename(os.path.normpath(self.directory)) or 'output'
            exts_set = set([os.path.splitext(p)[1].lower() for p in image_paths])
            if len(exts_set) == 1:
                palette_ext = next(iter(exts_set))
            else:
                palette_ext = '.png'

            # Use group name if provided, otherwise fall back to folder name
            if self.group_name:
                palette_base = os.path.join(self.output_dir, f"{self.group_name}_palette")
            else:
                palette_base = os.path.join(self.output_dir, f"palette_{folder_name}")
            
            if palette_ext == '.dds':
                # write temp png then convert to DDS
                tmp_png = palette_base + '.png'
                palette_img.save(tmp_png)
                palette_out = palette_base + '.dds'
                try:
                    convert_to_dds(tmp_png, palette_out, cfg.get(cfg.texconv_file), is_palette=True, palette_width=palette_width, palette_height=palette_height)
                finally:
                    try:
                        os.remove(tmp_png)
                    except Exception:
                        pass
            else:
                palette_out = palette_base + palette_ext
                palette_img.save(palette_out)

            self.progress.emit(100, "Bulk palette generation complete")

            self.completed.emit({
                'palette_path': palette_out,
                'palette_colors': palette_np.tolist(),
                'images': results_greyscale
            })

        except Exception as e:
            logger.error(f"BulkPaletteWorker error: {e}", exc_info=True)
            self.error.emit(str(e))

    @staticmethod
    def _sanitize_for_filename(name: str) -> str:
        if not name:
            return ""
        # Remove characters not allowed in Windows filenames and trim spaces/dots
        cleaned = re.sub(r'[<>:"/\\|?*]+', '', name)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip().strip('.')
        return cleaned

    def _collect_images(self):
        paths = []
        suf = (self.suffix_filter or '').lower()
        exf = (self.exclude_filter or '').lower()
        # Parse CSV include and exclude filters into lists of substrings (case-insensitive)
        include_terms = [t.strip() for t in suf.split(',') if t.strip()] if suf else []
        exclude_terms = [t.strip() for t in exf.split(',') if t.strip()] if exf else []

        def is_included(name_lower: str) -> bool:
            # If no include terms provided, include everything by default
            if not include_terms:
                return True
            return any(term in name_lower for term in include_terms)

        def is_excluded(name_lower: str) -> bool:
            if not exclude_terms:
                return False
            return any(term in name_lower for term in exclude_terms)

        # Determine whether to include subdirectories based on config
        include_subdirs_raw = cfg.get(cfg.ci_include_subdirs) if hasattr(cfg, 'ci_include_subdirs') else False
        include_subdirs = include_subdirs_raw.value if hasattr(include_subdirs_raw, 'value') else include_subdirs_raw
        include_subdirs = bool(include_subdirs)
        if include_subdirs:
            for root, _, files in os.walk(self.directory):
                for name in files:
                    base, ext = os.path.splitext(name)
                    if ext.lower() not in SUPPORTED_EXTS:
                        continue
                    # apply include filter (CSV; anywhere in the filename)
                    if not is_included(name.lower()):
                        continue
                    # apply exclude filter (anywhere in filename)
                    if is_excluded(name.lower()):
                        continue
                    paths.append(os.path.join(root, name))
        else:
            for name in os.listdir(self.directory):
                base, ext = os.path.splitext(name)
                if ext.lower() not in SUPPORTED_EXTS:
                    continue
                # apply include filter (CSV; anywhere in the filename)
                if not is_included(name.lower()):
                    continue
                # apply exclude filter (anywhere in filename)
                if is_excluded(name.lower()):
                    continue
                paths.append(os.path.join(self.directory, name))
        return sorted(paths)

    @staticmethod
    def _downscale_keep_aspect(img: Image.Image, max_side: int) -> Image.Image:
        w, h = img.size
        scale = min(1.0, max_side / max(w, h)) if max_side and max(w, h) > max_side else 1.0
        if scale >= 1.0:
            return img
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        return img.resize(new_size, Image.Resampling.LANCZOS)

    def _build_global_palette(self, global_counter: Counter, target_size: int, color_coverage: dict | None = None):
        # Colors sorted by global frequency (desc)
        color_items = list(global_counter.items())
        color_items.sort(key=lambda kv: kv[1], reverse=True)
        colors = np.array([kv[0] for kv in color_items], dtype=np.uint8)
        freqs = np.array([kv[1] for kv in color_items], dtype=np.float64)

        if len(colors) == 0:
            return [], {}, []

        # Per-color image coverage (how many distinct images contain this color)
        coverage = np.array([float((color_coverage or {}).get(tuple(c), 0)) for c in colors], dtype=np.float64)

        # Compute LAB, chroma, hue
        labs = rgb_to_lab_array(colors.tolist()).astype(np.float64)
        L = labs[:, 0]
        a = labs[:, 1]
        b = labs[:, 2]
        chroma = np.sqrt(a * a + b * b)
        Cmax = max(1e-6, float(np.max(chroma)))
        hues = np.array([hue_angle_from_ab(ai, bi) for ai, bi in zip(a, b)], dtype=np.float64)

        # Effective weight to prevent neutral dominance and preserve saturated small features
        alpha = 0.85
        k_chroma = 0.6
        k_cov = 0.25
        eff_w = np.power(freqs, alpha) * (1.0 + k_chroma * (chroma / Cmax)) + k_cov * coverage

        # ΔE00 helper (vectorized against a set of centers)
        def de00_to_centers(lab_point: np.ndarray, centers: np.ndarray) -> np.ndarray:
            # lab_point: (3,), centers: (K,3) -> (K,)
            lp = lab_point.reshape(1, 1, 3)
            ct = centers.reshape(-1, 1, 3)
            # deltaE_ciede2000 expects shapes (..,3)
            return deltaE_ciede2000(ct, lp).reshape(-1)

        # Greedy clustering with increasing ΔE00 threshold until <= target_size
        thresholds = list(range(0, 21, 1))  # ΔE00 0..20
        best_result = None
        for thresh in thresholds:
            kept_labs = []   # running cluster centers (LAB)
            clusters = []    # list of lists of member indices
            weights_sum = [] # sum of eff_w per cluster (for center updates)

            for i, lab_i in enumerate(labs):
                if not kept_labs:
                    kept_labs.append(lab_i.copy())
                    clusters.append([i])
                    weights_sum.append(eff_w[i])
                    continue
                centers = np.vstack(kept_labs)
                d = de00_to_centers(lab_i, centers)
                j = int(np.argmin(d))
                if d[j] <= thresh:
                    clusters[j].append(i)
                    # Update center as weighted mean in LAB using eff_w
                    members = clusters[j]
                    w = eff_w[members]
                    kept_labs[j] = np.sum(labs[members] * w[:, None], axis=0) / max(1e-9, np.sum(w))
                    weights_sum[j] = float(np.sum(w))
                else:
                    kept_labs.append(lab_i.copy())
                    clusters.append([i])
                    weights_sum.append(eff_w[i])

            num_clusters = len(clusters)
            best_result = (clusters, np.vstack(kept_labs) if kept_labs else np.empty((0, 3)))
            if num_clusters <= target_size:
                break

        clusters, kept_labs = best_result

        # Choose a weighted medoid (real member) per cluster using ΔE00 and eff_w
        reps = []
        rep_index_of_cluster = []
        cluster_scores = []  # composite score per cluster (sum eff_w)
        cluster_median_chroma = []
        cluster_coverage_sum = []
        cluster_hue_bin = []

        # Hue binning config
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
                cluster_coverage_sum.append(float(coverage[k]))
                cluster_hue_bin.append(hue_to_bin(hues[k]))
                continue
            member_idx = np.array(members, dtype=int)
            member_labs = labs[member_idx]
            member_w = eff_w[member_idx]
            # Pairwise ΔE00 distances
            # Compute distances from each candidate to all members and compute weighted sum cost
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
            cluster_coverage_sum.append(float(np.sum(coverage[member_idx])))
            cluster_hue_bin.append(hue_to_bin(hues[k]))

        n_clusters = len(reps)

        # Selection/trimming with hue caps and quotas if over target
        kept_indices = list(range(n_clusters))
        pad_candidates = []
        if n_clusters > target_size:
            # Determine per-bin cap and quota
            max_per_bin = max(1, int(np.round(target_size * 0.15)))
            # Identify bins with any high-chroma presence to reserve at least one
            high_chroma_bins = set()
            for i in range(n_clusters):
                if cluster_median_chroma[i] >= 20.0:
                    high_chroma_bins.add(cluster_hue_bin[i])
            # Rank clusters by composite score, then median chroma, then coverage, then total freq
            totals = [float(np.sum(freqs[clusters[i]])) for i in range(n_clusters)]
            order = sorted(range(n_clusters), key=lambda i: (
                -cluster_scores[i], -cluster_median_chroma[i], -cluster_coverage_sum[i], -totals[i]
            ))
            # First, reserve one per high-chroma bin (best cluster in that bin)
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
            # Fill remaining respecting per-bin caps
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
            # Prepare pad candidates from the next best that were not selected (respect caps loosely)
            pad_candidates = [reps[i] for i in order if i not in kept_indices]
        else:
            # Under or equal target: all are kept, pad candidates are empty
            kept_indices = list(range(n_clusters))
            pad_candidates = []

        kept_reps = [reps[i] for i in kept_indices]
        kept_rep_indices = [rep_index_of_cluster[i] for i in kept_indices]
        kept_rep_labs = labs[kept_rep_indices]
        kept_rep_bins = [cluster_hue_bin[i] for i in kept_indices]

        # Build color_map: map every original color to nearest kept rep in LAB, prefer same hue bin
        color_map = {}
        for idx in range(len(colors)):
            lab_i = labs[idx]
            # Prefer same hue bin based on the original color's hue
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
            color_map[tuple(colors[idx])] = kept_reps[j]

        return kept_reps, color_map, pad_candidates

    @staticmethod
    def _nearest_palette_index(color: np.ndarray, palette_int16: np.ndarray) -> int:
        # Euclidean distance in RGB
        diff = palette_int16 - color.astype(np.int16)
        d2 = np.sum(diff * diff, axis=1)
        return int(np.argmin(d2))

    def _map_rgb_array_to_palette_indices(self, arr_rgb: np.ndarray, lut_exact: dict, palette_int16: np.ndarray) -> np.ndarray:
        h, w = arr_rgb.shape[:2]
        flat = arr_rgb.reshape(-1, 3)
        # First try exact mapping for speed
        indices = np.full((flat.shape[0],), 0, dtype=np.uint16)
        unmatched_mask = np.ones((flat.shape[0],), dtype=bool)
        for i in range(flat.shape[0]):
            t = tuple(int(x) for x in flat[i])
            if t in lut_exact:
                indices[i] = lut_exact[t]
                unmatched_mask[i] = False
        # For unmatched, do nearest search
        if unmatched_mask.any():
            cand = flat[unmatched_mask]
            pal = palette_int16
            # Vectorized distance
            # Expand dims: [N,1,3] vs [1,P,3] -> broadcast to [N,P,3]
            # To save memory, process in chunks
            chunk = 100000
            start = 0
            where_idx = np.where(unmatched_mask)[0]
            while start < cand.shape[0]:
                end = min(start + chunk, cand.shape[0])
                c = cand[start:end].astype(np.int16)[:, None, :]
                diff = pal[None, :, :] - c
                d2 = np.sum(diff * diff, axis=2)  # [nchunk, P]
                nearest = np.argmin(d2, axis=1)
                indices[where_idx[start:end]] = nearest.astype(np.uint16)
                start = end
        return indices.reshape(h, w)


class BulkPaletteWidget(BaseWidget):
    def __init__(self, parent=None, text=None):
        super().__init__(parent=parent, text=text, vertical=True)
        self.main_widget = QWidget()
        self.dir_card = None
        self.suffix_card = None
        self.output_dir_card = None
        self.generate_dds_card = None
        self.generate_button = None
        self.progress_bar = None

        self.directory = None
        self.suffix = "_d"
        self.output_dir = None
        self.generate_dds = False

        # Directory selection
        self.dir_card = PushSettingCard(
            self.tr("Texture Directory"),
            FIF.FOLDER,
            self.tr("Directory containing textures"),
            self.tr("No directory selected")
        )
        self.dir_card.clicked.connect(self._choose_directory)
        self.addToFrame(self.dir_card)

        self.subdirs_card= SwitchSettingCard(
            icon=CustomIcons.SUB.icon(),
            title=self.tr("Include Subdirectories"),
            configItem=cfg.ci_include_subdirs
        )
        
        self.addToFrame(self.subdirs_card)

        # Include filter (CSV) selection
        self.suffix_card = TextSettingCard(
            cfg.ci_suffix,
            FIF.FILTER,
            self.tr("Include filter (CSV)"),
            self.tr("Include files whose name contains any of these terms (comma-separated, case-insensitive). Leave empty to include all."),
        )
        self.addToFrame(self.suffix_card)

        # Exclude filter (CSV) selection
        self.exclude_card = TextSettingCard(
            cfg.ci_exclude,
            FIF.CLOSE,
            self.tr("Exclude filter (CSV)"),
            self.tr("Skip files whose name contains any of these terms (comma-separated, case-insensitive). Leave empty to skip none."),
        )
        self.addToFrame(self.exclude_card)

        # Group name (appended to output filenames and used for palette name)
        self.group_card = TextSettingCard(
            cfg.ci_group_name,
            FIF.TAG,
            self.tr("Group name"),
            self.tr("Optional: appended to all output filenames and used to name the palette (e.g., group_palette.dds)."),
        )
        self.addToFrame(self.group_card)

        # Output directory
        self.output_dir_card = PushSettingCard(
            self.tr("Output Directory"),
            CustomIcons.FOLDERRIGHT.icon(),
            self.tr("Where greyscale and palette images will be written"),
            self.tr("Will use input directory if not set")
        )
        self.output_dir_card.clicked.connect(self._choose_output_dir)
        self.addToFrame(self.output_dir_card)

        # Generate button
        self.generate_button = PrimaryPushButton(icon=FIF.RIGHT_ARROW, text=self.tr("Run"))
        self.generate_button.clicked.connect(self._run)
        self.generate_button.setEnabled(True)


        # Progress masked by parent window; no local ProgressBar

        # Info label
        self.info_label = QLabel("")
        self.addToFrame(self.info_label)

        self.addButtonBarToBottom(self.generate_button)

        self.settings_widget = PaletteSettings(self)
        self.settings_drawer.addWidget(self.settings_widget)
        self.help_widget = BulkPaletteHelp(self)
        self.help_drawer.addWidget(self.help_widget)

    def _choose_directory(self):
        directory = QFileDialog.getExistingDirectory(self, self.tr("Select Texture Directory"), "")
        if directory:
            self.directory = directory
            self.dir_card.setContent(directory)

    def _choose_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, self.tr("Select Output Directory"), "")
        if directory:
            self.output_dir = directory
            self.output_dir_card.setContent(directory)

    def _run(self):
        if not self.directory:
            QMessageBox.warning(self, self.tr("Missing Directory"), self.tr("Please choose a texture directory."))
            return

        # Get working resolution from ConfigItem
        wr_val = cfg.get(cfg.ci_default_working_res).value
        if isinstance(wr_val, str) and str(wr_val).lower().startswith('original'):
            wr = None
        else:
            try:
                wr = int(wr_val)
            except Exception:
                wr = None

        # Use parent window mask progress instead of local progress bar
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'show_progress'):
            try:
                p.show_progress()
            except Exception:
                pass
        self.info_label.setText(self.tr("Starting..."))
        self.worker = BulkPaletteWorker(
            directory=self.directory,
            suffix_filter=cfg.get(cfg.ci_suffix),
            output_dir=self.output_dir,
            generate_dds=self.generate_dds,
            working_resolution=wr,
            palette_size=cfg.get(cfg.ci_default_palette_size),
            exclude_filter=cfg.get(cfg.ci_exclude),
            group_name=cfg.get(cfg.ci_group_name)
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.completed.connect(self._on_completed)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, value, msg):
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'update_progress'):
            try:
                p.update_progress(int(value))
            except Exception:
                pass
        self.info_label.setText(msg)

    def _on_completed(self, results):
        pal_path = results.get('palette_path')
        count = len(results.get('images', []))
        self.info_label.setText(self.tr(f"Done. Saved palette to {pal_path}. Converted {count} images."))
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'complete_loader'):
            try:
                p.complete_loader()
            except Exception:
                pass

    def _on_error(self, message):
        QMessageBox.critical(self, self.tr("Bulk Palette Error"), message)
        self.info_label.setText(message)
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'complete_loader'):
            try:
                p.complete_loader()
            except Exception:
                pass
