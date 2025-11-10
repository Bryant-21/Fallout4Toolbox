import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    build_palette_from_rgb_array,
    map_rgb_array_to_palette_indices,
    nearest_palette_index,
)
from src.utils.appconfig import cfg, TEXCONV_EXE
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger
from utils.cards import TextSettingCard

SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tif', '.tiff', '.dds'}


class BulkPaletteWorker(QThread):
    progress = Signal(int, str)
    error = Signal(str)
    completed = Signal(dict)

    def __init__(self, directory, suffix_filter, output_dir, generate_dds=False, working_resolution=None, palette_size=256, exclude_filter: str = "", group_name: str = "", single_palette: bool = True):
        super().__init__()
        self.directory = directory
        self.suffix_filter = suffix_filter
        self.exclude_filter = (exclude_filter or "").strip()
        self.output_dir = output_dir or directory
        self.generate_dds = generate_dds
        self.working_resolution = working_resolution
        self.palette_size = palette_size
        self.group_name = self._sanitize_for_filename((group_name or "").strip())
        self.single_palette = bool(single_palette)

    def run(self):
        try:
            self.progress.emit(1, "Scanning directory...")
            image_paths = self._collect_images()
            if not image_paths:
                raise Exception("No images found matching the include filter in the selected directory.")

            self.progress.emit(5, f"Found {len(image_paths)} images. Loading and quantizing...")

            # Step 1: Quantize each image and collect RGB arrays (no frequency counting)
            per_image_quant = []
            global_unique_colors = set()
            sample_img_for_padding = None

            # Prepare a sample image for padding (load first image synchronously)
            sample_img_for_padding = None
            if image_paths:
                try:
                    _img0 = load_image(image_paths[0])
                    if self.working_resolution and self.working_resolution > 0:
                        _img0 = self._downscale_keep_aspect(_img0, self.working_resolution)
                    sample_img_for_padding = _img0
                except Exception:
                    sample_img_for_padding = None

            def _step1_task(path: str):
                img = load_image(path)
                if self.working_resolution and self.working_resolution > 0:
                    img = self._downscale_keep_aspect(img, self.working_resolution)
                quantized_img, _ = quantize_image(img, cfg.get(cfg.ci_default_quant_method), True)
                q_rgb = quantized_img.convert('RGB')
                q_arr = np.array(q_rgb)
                h, w = q_arr.shape[:2]
                uniq = np.unique(q_arr.reshape(-1, 3), axis=0)
                uniq_set = {tuple(int(x) for x in c) for c in uniq.tolist()}
                return {
                    'path': path,
                    'array_rgb': q_arr,
                    'size': (w, h),
                    'unique_set': uniq_set,
                }

            # Run Step 1 in parallel threads
            results_by_path = {}
            completed = 0
            total = len(image_paths)
            max_workers = max(1, int(cfg.get(cfg.threads_cfg)))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                future_map = {ex.submit(_step1_task, p): p for p in image_paths}
                for fut in as_completed(future_map):
                    path = future_map[fut]
                    try:
                        res = fut.result()
                    except Exception as e:
                        raise Exception(f"Failed processing {os.path.basename(path)}: {e}")
                    # Update global unique color set
                    global_unique_colors.update(res['unique_set'])
                    # Store per-image result
                    results_by_path[path] = {
                        'path': res['path'],
                        'array_rgb': res['array_rgb'],
                        'size': res['size'],
                        'unique_set': res['unique_set'],
                    }
                    completed += 1
                    pct = 5 + int(completed / max(1, total) * 45)
                    self.progress.emit(pct, f"Quantized {os.path.basename(path)} with {len(res['unique_set'])} unique colors")

            # Preserve original order
            per_image_quant = [results_by_path[p] for p in image_paths]

            # Branch based on single vs multiple palette setting
            if not self.single_palette:
                # Multi-palette mode: build a separate palette per image and save outputs
                self.progress.emit(50, f"Building {len(per_image_quant)} per-image palette(s)...")

                def _per_image_task(item):
                    try:
                        logger.debug(f"Starting Per Image Processing {os.path.basename(item['path'])}")
                        # Build per-image palette (<= target size) from the image's RGB array
                        pcolors, cmap, pad_cands = build_palette_from_rgb_array(item['array_rgb'], target_size=self.palette_size)
                        pal_np = np.array(pcolors, dtype=np.uint8)
                        # Pad up to target size if needed
                        if len(pal_np) < self.palette_size:
                            try:
                                if pad_cands:
                                    seen = {tuple(map(int, c)) for c in pal_np.tolist()}
                                    to_add = []
                                    for c in pad_cands:
                                        t = tuple(int(x) for x in c)
                                        if t not in seen:
                                            to_add.append(t)
                                            seen.add(t)
                                            if len(pal_np) + len(to_add) >= self.palette_size:
                                                break
                                    if to_add:
                                        pal_np = np.vstack([pal_np, np.array(to_add, dtype=np.uint8)])
                                if len(pal_np) < self.palette_size:
                                    pal_np = pad_colors_to_target(pal_np, None, self.palette_size)
                            except Exception as e:
                                logger.warning(f"Padding per-image palette failed: {e}")
                                while len(pal_np) < self.palette_size:
                                    pal_np = np.vstack([pal_np, pal_np[:max(1, self.palette_size - len(pal_np))]])
                                pal_np = pal_np[:self.palette_size]
                        # Sort perceptually
                        logger.debug(f"Starting Color Sort {os.path.basename(item['path'])}")
                        pal_sorted = np.array(perceptual_color_sort([tuple(c) for c in pal_np]), dtype=np.uint8)
                        logger.debug(f"Colors Sorted {os.path.basename(item['path'])}")

                        # Map image to indices and save greyscale + colorized + per-image palette
                        arr = item['array_rgb']
                        pal_index_lut = {tuple(color): i for i, color in enumerate(pal_sorted.tolist())}
                        pal_int16 = pal_sorted.astype(np.int16)
                        lut_exact = {}
                        for lc in item['unique_set']:
                            rep = cmap.get(tuple(lc))
                            if rep is None:
                                idx = nearest_palette_index(np.array(lc, dtype=np.uint8), pal_int16)
                                lut_exact[tuple(int(x) for x in lc)] = int(idx)
                            else:
                                lut_exact[tuple(int(x) for x in lc)] = int(pal_index_lut[tuple(rep)])
                        grey_indices = map_rgb_array_to_palette_indices(arr, lut_exact, pal_int16).astype(np.uint8)

                        src_ext = os.path.splitext(item['path'])[1].lower()
                        rel_name = os.path.splitext(os.path.basename(item['path']))[0]
                        base_with_group = f"{rel_name}_{self.group_name}" if self.group_name else rel_name
                        grey_base = f"{base_with_group}_greyscale"
                        color_base = f"{base_with_group}_quant"
                        pal_base = f"{base_with_group}_palette"

                        # Save greyscale and colorized
                        if src_ext == '.dds':
                            tmp_grey_png = os.path.join(self.output_dir, grey_base + '.png')
                            Image.fromarray(grey_indices, mode='L').save(tmp_grey_png)
                            grey_out = os.path.join(self.output_dir, grey_base + '.dds')
                            try:
                                convert_to_dds(tmp_grey_png, grey_out, is_palette=False)
                            finally:
                                try:
                                    os.remove(tmp_grey_png)
                                except Exception:
                                    pass

                            color_arr = pal_sorted[grey_indices]
                            tmp_color_png = os.path.join(self.output_dir, color_base + '.png')
                            Image.fromarray(color_arr.astype(np.uint8), mode='RGB').save(tmp_color_png)
                            color_out = os.path.join(self.output_dir, color_base + '.dds')
                            try:
                                convert_to_dds(tmp_color_png, color_out, is_palette=False)
                            finally:
                                try:
                                    os.remove(tmp_color_png)
                                except Exception:
                                    pass

                            # Save per-image palette as DDS (via temp PNG)
                            palette_width = self.palette_size
                            palette_row_height = cfg.get(cfg.ci_palette_row_height) if hasattr(cfg, 'ci_palette_row_height') else 8
                            palette_height = next_power_of_2(palette_row_height)
                            pal_img_arr = np.zeros((int(palette_height), int(palette_width), 3), dtype=np.uint8)
                            for row in range(int(palette_height)):
                                pal_img_arr[row, :int(palette_width)] = pal_sorted
                            pal_img = Image.fromarray(pal_img_arr, 'RGB')
                            tmp_pal_png = os.path.join(self.output_dir, pal_base + '.png')
                            pal_img.save(tmp_pal_png)
                            pal_out = os.path.join(self.output_dir, pal_base + '.dds')
                            try:
                                convert_to_dds(tmp_pal_png, pal_out, is_palette=True, palette_width=palette_width, palette_height=palette_height)
                            finally:
                                try:
                                    os.remove(tmp_pal_png)
                                except Exception:
                                    pass
                        else:
                            grey_out = os.path.join(self.output_dir, grey_base + src_ext)
                            Image.fromarray(grey_indices, mode='L').save(grey_out)
                            color_arr = pal_sorted[grey_indices]
                            color_out = os.path.join(self.output_dir, color_base + src_ext)
                            Image.fromarray(color_arr.astype(np.uint8), mode='RGB').save(color_out)

                            palette_width = self.palette_size
                            palette_row_height = cfg.get(cfg.ci_palette_row_height) if hasattr(cfg, 'ci_palette_row_height') else 8
                            palette_height = next_power_of_2(palette_row_height)
                            pal_img_arr = np.zeros((int(palette_height), int(palette_width), 3), dtype=np.uint8)
                            for row in range(int(palette_height)):
                                pal_img_arr[row, :int(palette_width)] = pal_sorted
                            pal_img = Image.fromarray(pal_img_arr, 'RGB')
                            pal_out = os.path.join(self.output_dir, pal_base + src_ext)
                            pal_img.save(pal_out)

                        return {
                            'source': item['path'],
                            'greyscale_path': grey_out,
                            'color_path': color_out,
                            'palette_path': pal_out,
                        }
                    except Exception as e:
                        logger.warning(f"Per-image processing failed for {os.path.basename(item['path'])}: {e}")
                        return {
                            'source': item['path'],
                            'error': str(e)
                        }

                total_pi = len(per_image_quant)
                completed_pi = 0
                results_list = [None] * total_pi
                max_workers_pi = max(1, int(cfg.get(cfg.threads_cfg)))
                with ThreadPoolExecutor(max_workers=max_workers_pi) as ex:
                    futures = {ex.submit(_per_image_task, item): idx for idx, item in enumerate(per_image_quant)}
                    for fut in as_completed(futures):
                        idx = futures[fut]
                        res = fut.result()
                        results_list[idx] = res
                        completed_pi += 1
                        pct = 50 + int(completed_pi / max(1, total_pi) * 50)
                        name = os.path.basename(per_image_quant[idx]['path'])
                        self.progress.emit(pct, f"Saved per-image outputs for {name}")

                self.progress.emit(100, "Bulk palette generation (multiple) complete")
                self.completed.emit({
                    'palette_path': '(multiple)',
                    'images': [r for r in results_list if r is not None]
                })
                return
            else:
                # SINGLE-PALETTE MODE
                self.progress.emit(50, f"Merging {len(global_unique_colors)} colors down to ≤ {self.palette_size}...")

                # Step 2: Build global palette ≤ palette_size by merging similar colors
                global_colors_arr = np.array(list(global_unique_colors), dtype=np.uint8) if global_unique_colors else np.zeros((0,3), dtype=np.uint8)
                palette_colors, color_map, pad_candidates = build_palette_from_rgb_array(global_colors_arr, target_size=self.palette_size)

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

                def _step4_task(item):
                    arr = item['array_rgb']
                    # Build per-image exact LUT from unique colors only
                    lut_exact = {}
                    local_colors = item.get('unique_set', set())
                    for lc in local_colors:
                        rep = color_map.get(tuple(lc))
                        if rep is None:
                            nearest_idx = nearest_palette_index(np.array(lc, dtype=np.uint8), pal_float)
                            lut_exact[tuple(int(x) for x in lc)] = int(nearest_idx)
                        else:
                            lut_exact[tuple(int(x) for x in lc)] = int(palette_index_lut[tuple(rep)])
                    grey_indices = map_rgb_array_to_palette_indices(arr, lut_exact, pal_float).astype(np.uint8)

                    src_ext = os.path.splitext(item['path'])[1].lower()
                    rel_name = os.path.splitext(os.path.basename(item['path']))[0]
                    base_with_group = f"{rel_name}_{self.group_name}" if self.group_name else rel_name
                    grey_base = f"{base_with_group}_greyscale"
                    color_base = f"{base_with_group}_quant"

                    if src_ext == '.dds':
                        tmp_grey_png = os.path.join(self.output_dir, grey_base + '.png')
                        Image.fromarray(grey_indices, mode='L').save(tmp_grey_png)
                        grey_out = os.path.join(self.output_dir, grey_base + '.dds')
                        try:
                            convert_to_dds(tmp_grey_png, grey_out, is_palette=False)
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
                            convert_to_dds(tmp_color_png, color_out, is_palette=False)
                        finally:
                            try:
                                os.remove(tmp_color_png)
                            except Exception:
                                pass
                    else:
                        grey_out = os.path.join(self.output_dir, grey_base + src_ext)
                        Image.fromarray(grey_indices, mode='L').save(grey_out)
                        color_arr = palette_np[grey_indices]
                        color_out = os.path.join(self.output_dir, color_base + src_ext)
                        Image.fromarray(color_arr.astype(np.uint8), mode='RGB').save(color_out)

                    return {
                        'source': item['path'],
                        'greyscale_path': grey_out,
                        'color_path': color_out,
                    }

                # Run Step 4 in parallel threads
                total4 = len(per_image_quant)
                completed4 = 0
                max_workers4 = max(1, int(cfg.get(cfg.threads_cfg)))
                with ThreadPoolExecutor(max_workers=max_workers4) as ex:
                    future_map4 = {ex.submit(_step4_task, item): item for item in per_image_quant}
                    for fut in as_completed(future_map4):
                        item = future_map4[fut]
                        try:
                            res = fut.result()
                        except Exception as e:
                            raise Exception(f"Failed saving {os.path.basename(item['path'])}: {e}")
                        results_greyscale.append(res)
                        completed4 += 1
                        pct = 70 + int(completed4 / max(1, total4) * 20)
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
                        convert_to_dds(tmp_png, palette_out, is_palette=True, palette_width=palette_width, palette_height=palette_height)
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

    @staticmethod
    def build_global_palette(colors_source: np.ndarray, target_size: int):
        # Accepts an RGB array (HxWx3 or Nx3) and merges similar colors until <= target_size
        # 1) Deduplicate colors
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

        # 2) Convert to LAB for perceptual distance
        labs = rgb_to_lab_array(colors.tolist()).astype(np.float64)

        # 3) Greedy clustering with increasing ΔE00 threshold
        def de00_to_centers(lab_point: np.ndarray, centers: np.ndarray) -> np.ndarray:
            lp = lab_point.reshape(1, 1, 3)
            ct = centers.reshape(-1, 1, 3)
            return deltaE_ciede2000(ct, lp).reshape(-1)

        thresholds = list(range(0, 21, 1))  # ΔE00 0..20
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
                    # Update center as simple mean (unweighted)
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

        # 4) Choose medoid (member with minimal sum of distances) per cluster
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
            # Keep the largest clusters first
            sizes = [len(clusters[i]) for i in range(n_clusters)]
            order = sorted(range(n_clusters), key=lambda i: -sizes[i])
            kept_indices = order[:target_size]
            pad_candidates = [reps[i] for i in order[target_size:]]

        kept_reps = [reps[i] for i in kept_indices]
        kept_rep_indices = [rep_index_of_cluster[i] for i in kept_indices]
        kept_rep_labs = labs[kept_rep_indices]

        # 5) Build color_map: map every original color to nearest kept rep in LAB
        color_map = {}
        for idx in range(len(colors)):
            lab_i = labs[idx]
            d = deltaE_ciede2000(kept_rep_labs.reshape(-1, 1, 3), lab_i.reshape(1, 1, 3)).reshape(-1)
            j = int(np.argmin(d))
            color_map[tuple(colors[idx])] = kept_reps[j]

        return kept_reps, color_map, pad_candidates



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
            self.tr("Directory containing textures"),
            CustomIcons.FOLDER_IMAGE.icon(stroke=True),
            self.tr("Texture Directory"),
            self.tr("No directory selected")
        )
        self.dir_card.clicked.connect(self._choose_directory)
        self.addToFrame(self.dir_card)

        self.subdirs_card= SwitchSettingCard(
            icon=CustomIcons.SUB.icon(),
            title=self.tr("Include Subdirectories"),
            configItem=cfg.ci_include_subdirs
        )

        self.single_palette_card = SwitchSettingCard(
            icon=CustomIcons.SUB.icon(),
            title=self.tr("Single Palette (On) or Multiple (Off)"),
            configItem=cfg.ci_single_palette
        )
        
        self.addToFrame(self.subdirs_card)
        # Single vs Multiple palette toggle
        self.addToFrame(self.single_palette_card)

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
        # Resolve boolean for single vs multiple palette correctly from ConfigItem
        sp_raw = cfg.get(cfg.ci_single_palette)
        sp_val = sp_raw.value if hasattr(sp_raw, 'value') else sp_raw
        self.worker = BulkPaletteWorker(
            directory=self.directory,
            suffix_filter=cfg.get(cfg.ci_suffix),
            output_dir=self.output_dir,
            generate_dds=self.generate_dds,
            working_resolution=wr,
            palette_size=cfg.get(cfg.ci_default_palette_size),
            exclude_filter=cfg.get(cfg.ci_exclude),
            group_name=cfg.get(cfg.ci_group_name),
            single_palette=bool(sp_val)
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
