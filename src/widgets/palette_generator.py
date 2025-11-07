import json
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from PIL import Image
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QThread
from PySide6.QtCore import Signal
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (QVBoxLayout, QLabel, QFileDialog, QWidget, QMessageBox,
                               QScrollArea,
                               QGridLayout, QStackedWidget)
from qfluentwidgets import (
    PushSettingCard,
    ConfigItem,
    FluentIcon as FIF,
    PrimaryPushButton,
    PushButton, SegmentedWidget
)

from src.help.palette_help import PaletteHelp
from src.palette.palette_engine import perceptual_color_sort, quantize_image, next_power_of_2, \
    analyze_color_distribution, convert_to_dds, pad_colors_to_target, rebuild_image_with_padded_colors, load_image, \
    reduce_colors_lab_de00_with_hue_balance, remap_rgb_array_to_representatives
from src.settings.palette_settings import PaletteSettings
from src.utils.appconfig import cfg
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger


class SinglePaletteGenerationWorker(QThread):
    """Worker thread for generating a perfect Palette for a single image"""
    progress_updated = Signal(int, str)
    result_ready = Signal(dict)
    error_occurred = Signal(str)
    step_complete = Signal(str, object)  # step_name, result_data

    def __init__(self, image_path, output_dir,
                 generate_dds=False, step=None,
                 previous_data=None,
                 extra_image_paths=None, working_resolution=None,
                 produce_color_report=False, produce_metadata_json=False,
                 greyscale_texture_paths=None,
                 palette_row_height=4):
        super().__init__()
        self.image_path = image_path
        self.output_dir = output_dir
        self.generate_dds = generate_dds
        self.step = step  # Specific step to run: 'quantize', 'greyscale', 'palette', 'all'
        self.previous_data = previous_data or {}  # Data from previous steps
        self.extra_image_paths = extra_image_paths or []
        self.working_resolution = working_resolution  # None for Original, else max side target (e.g., 4096)
        self.produce_color_report = produce_color_report
        self.produce_metadata_json = produce_metadata_json
        self.greyscale_texture_paths = greyscale_texture_paths or []
        self.palette_row_height = palette_row_height
        # sRGB compensation removed from UI/logic; keep attribute for backward-compatibility in save paths
        self.srgb_compensation = False

        logger.debug(
            f"Worker initialized: step={step}, method={cfg.get(cfg.ci_default_quant_method)}, palette_size={cfg.get(cfg.ci_default_palette_size)}, image={image_path}, extras={len(self.extra_image_paths)}, gs_tex={len(self.greyscale_texture_paths)}, work_res={working_resolution}, report={self.produce_color_report}, meta={self.produce_metadata_json}")

    def run(self):
        try:
            logger.debug(f"Starting Palette generation step: {self.step}")

            if self.step == 'quantize' or self.step == 'all':
                results = self.run_quantization_step()
                if self.step == 'quantize':
                    self.step_complete.emit('quantize', results)
                    return
                # Store for next steps
                self.previous_data['quantize'] = results

            if self.step == 'greyscale' or self.step == 'all':
                # Use quantized data from previous step if available
                if 'quantize' in self.previous_data:
                    quantize_data = self.previous_data['quantize']
                    # Use the data directly without setting instance attributes
                    results = self.run_greyscale_step(quantize_data)
                else:
                    # No previous data available
                    self.error_occurred.emit(
                        "No quantized data available for greyscale step. Please run quantization first.")
                    return

                if self.step == 'greyscale':
                    self.step_complete.emit('greyscale', results)
                    return
                # Store for next steps
                self.previous_data['greyscale'] = results

            if self.step == 'palette' or self.step == 'all':
                # Use greyscale data from previous step if available
                if 'greyscale' in self.previous_data:
                    greyscale_data = self.previous_data['greyscale']
                    quantize_data = self.previous_data.get('quantize', {})
                    results = self.run_palette_step(greyscale_data, quantize_data)
                else:
                    # No previous data available
                    self.error_occurred.emit(
                        "No greyscale data available for Palette step. Please run previous steps first.")
                    return

                if self.step == 'palette':
                    self.step_complete.emit('palette', results)
                    return
                # Store for next steps
                self.previous_data['palette'] = results

            # If running all steps, save everything
            if self.step == 'all':
                self.save_all_results()

        except Exception as e:
            logger.error(f"Error in Palette generation: {str(e)}", exc_info=True)
            self.error_occurred.emit(str(e))

    def run_quantization_step(self):
        """Run only the quantization step"""
        try:
            logger.debug("Starting quantization step")
            self.progress_updated.emit(10, "Loading image...")

            def downscale_keep_aspect(img, max_side):
                if max_side is None:
                    return img
                w, h = img.size
                current_max = max(w, h)
                if current_max <= max_side:
                    return img
                scale = max_side / float(current_max)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            img = load_image(self.image_path, cfg.get(cfg.texconv_file))
            base_original_img = img
            base_orig_w, base_orig_h = base_original_img.size
            # Apply working resolution downscale to base (no upscale)
            if self.working_resolution:
                original_img = downscale_keep_aspect(base_original_img, self.working_resolution)
            else:
                original_img = base_original_img

            original_array = np.array(original_img)
            height, width = original_array.shape[:2]

            logger.debug(f"Image loaded (processed): {width}x{height} (original {base_orig_w}x{base_orig_h})")

            self.progress_updated.emit(30, f"Quantizing using {cfg.get(cfg.ci_default_quant_method)}...")

            # Quantize to exactly Palette_SIZE colors using selected method
            quantized, quantization_info = quantize_image(original_img, cfg.get(cfg.ci_default_quant_method))
            quantized_rgb = quantized.convert('RGB')
            quantized_array = np.array(quantized_rgb)

            # Get the unique colors from quantized image (and their counts)
            flat = quantized_array.reshape(-1, 3)
            unique_colors, counts = np.unique(flat, axis=0, return_counts=True)
            num_colors = len(unique_colors)

            logger.debug(f"Quantization complete: {num_colors} unique colors")

            target_size = int(cfg.get(cfg.ci_default_palette_size))
            # Determine the actual target we will use for padding/trimming
            # New behavior: when under target_size, pad to the closest power-of-two >= actual count

            # If we have more than target_size colors, reduce using LAB/ΔE00 with hue balancing
            if num_colors > target_size:
                self.progress_updated.emit(60, f"Reducing {num_colors}→{target_size} with LAB/ΔE00 + hue balancing...")
                try:
                    kept_reps, color_map, pad_candidates = reduce_colors_lab_de00_with_hue_balance(unique_colors, counts, target_size)
                    # Remap quantized_array colors to their representatives using shared helper
                    quantized_array = remap_rgb_array_to_representatives(quantized_array, color_map)
                    quantized_rgb = Image.fromarray(quantized_array.astype('uint8'), 'RGB')

                    # Recompute unique colors after remap
                    unique_colors = np.unique(quantized_array.reshape(-1, 3), axis=0)
                    num_colors = len(unique_colors)
                    logger.debug(f"Post-reduction unique colors: {num_colors}")
                except Exception as e:
                    logger.error(f"LAB/ΔE00 reduction failed: {e}")

            # Handle cases where we have fewer than Palette_SIZE colors (either originally or after reduction)
            if num_colors < target_size:
                # New behavior: pad to next power-of-two based on actual color count, not the configured target
                pad_to = int(next_power_of_2(max(1, num_colors)))
                if pad_to == num_colors:
                    logger.info(f"Color count {num_colors} is already a power of two; no padding needed")
                else:
                    logger.warning(f"Quantization has {num_colors} colors, padding to next power-of-two {pad_to} (config target {target_size})")
                    self.progress_updated.emit(70, f"Padding colors from {num_colors} to {pad_to}...")

                    # Pad with additional colors to reach pad_to
                    padded_colors = pad_colors_to_target(unique_colors, original_img, pad_to)
                    unique_colors = padded_colors
                    num_colors = pad_to

                    # Update the quantized image with padded colors (no-op placeholder)
                    quantized_array = rebuild_image_with_padded_colors(quantized_array, unique_colors)
                    quantized_rgb = Image.fromarray(quantized_array.astype('uint8'), 'RGB')

                    quantization_info['color_padding'] = f"Padded from {len(np.unique(flat, axis=0))} to {pad_to} colors"
                    quantization_info['original_color_count'] = int(len(np.unique(flat, axis=0)))

            # If still over target (reduction failed to trim enough), trim by perceptual selection
            if num_colors > target_size:
                self.progress_updated.emit(72, f"Trimming {num_colors}→{target_size} by perceptual selection...")
                # Pick a perceptually sorted order and keep first target_size distinct colors
                sorted_colors = perceptual_color_sort([tuple(c) for c in unique_colors])
                unique_colors = np.array(sorted_colors[:target_size], dtype=np.uint8)
                num_colors = target_size

            # Final verification: ensure we have a sensible palette size
            if num_colors < 1:
                error_msg = f"Quantization produced an invalid palette size: {num_colors}"
                logger.error(error_msg)
                return {'success': False, 'error': error_msg}

            self.progress_updated.emit(100, "Quantization complete!")

            # Quantize any extra images provided (for additional Palette blocks)
            extra_images_data = []
            if self.extra_image_paths:
                self.progress_updated.emit(85, f"Quantizing {len(self.extra_image_paths)} extra image(s)...")
                for p in self.extra_image_paths:
                    try:
                        ex_img = load_image(p, cfg.get(cfg.texconv_file))
                        ex_rgb = ex_img
                        ex_orig_w, ex_orig_h = ex_rgb.size
                        # Apply working resolution downscale (no upscale)
                        ex_proc = downscale_keep_aspect(ex_rgb, self.working_resolution) if self.working_resolution else ex_rgb

                        # Ensure not larger than base processed size
                        def fit_within(img, max_w, max_h):
                            w, h = img.size
                            if w <= max_w and h <= max_h:
                                return img
                            scale = min(max_w / float(w), max_h / float(h))
                            new_w = max(1, int(round(w * scale)))
                            new_h = max(1, int(round(h * scale)))
                            return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                        ex_proc2 = fit_within(ex_proc, width, height)
                        ex_w2, ex_h2 = ex_proc2.size
                        # Quantize using same method and lut size
                        ex_quantized, _ = quantize_image(ex_proc2, cfg.get(cfg.ci_default_quant_method))
                        ex_quant_rgb = ex_quantized.convert('RGB')
                        ex_arr = np.array(ex_quant_rgb)
                        extra_images_data.append({
                            'path': p,
                            'quantized_image': ex_quant_rgb,
                            'quantized_array': ex_arr,
                            'original_dimensions': (ex_orig_w, ex_orig_h),
                            'processed_dimensions': (ex_w2, ex_h2),
                            'dimensions': (ex_w2, ex_h2)
                        })
                    except Exception as ex:
                        logger.warning(f"Failed to process extra image '{p}': {ex}")

            # Quantize greyscale-conversion textures (to later map using base mapping)
            greyscale_textures_data = []
            if self.greyscale_texture_paths:
                self.progress_updated.emit(88, f"Preparing {len(self.greyscale_texture_paths)} greyscale conversion texture(s)...")
                for p in self.greyscale_texture_paths:
                    try:
                        gs_img = load_image(p, cfg.get(cfg.texconv_file))
                        gs_rgb = gs_img
                        gs_orig_w, gs_orig_h = gs_rgb.size
                        gs_proc = downscale_keep_aspect(gs_rgb, self.working_resolution) if self.working_resolution else gs_rgb

                        # Ensure not larger than base processed size
                        def fit_within(img, max_w, max_h):
                            w, h = img.size
                            if w <= max_w and h <= max_h:
                                return img
                            scale = min(max_w / float(w), max_h / float(h))
                            new_w = max(1, int(round(w * scale)))
                            new_h = max(1, int(round(h * scale)))
                            return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                        gs_proc2 = fit_within(gs_proc, width, height)
                        gs_w2, gs_h2 = gs_proc2.size
                        # Quantize with same method/lut size (for stable color set); we'll still map to base
                        gs_quantized, _ = quantize_image(gs_proc2, cfg.get(cfg.ci_default_quant_method))
                        gs_quant_rgb = gs_quantized.convert('RGB')
                        gs_quant_arr = np.array(gs_quant_rgb)
                        greyscale_textures_data.append({
                            'path': p,
                            'processed_color_image': gs_proc2.convert('RGB'),
                            'processed_color_array': np.array(gs_proc2.convert('RGB')),
                            'quantized_array': gs_quant_arr,
                            'original_dimensions': (gs_orig_w, gs_orig_h),
                            'processed_dimensions': (gs_w2, gs_h2)
                        })
                    except Exception as ex:
                        logger.warning(f"Failed to process greyscale texture '{p}': {ex}")

            # Prepare results for quantization step (always)
            results = {
                'success': True,
                'step': 'quantize',
                'original_image': original_img,
                'quantized_image': quantized_rgb,
                'quantized_array': quantized_array,
                'unique_colors': unique_colors,
                'quantization_info': quantization_info,
                'dimensions': (width, height),
                'base_original_dimensions': (base_orig_w, base_orig_h),
                'base_processed_dimensions': (width, height),
                'working_resolution': self.working_resolution,
                'actual_color_count': num_colors,
                'palette_size': int(num_colors),
                'extra_images': extra_images_data,
                'greyscale_textures': greyscale_textures_data
            }

            logger.debug("Quantization step completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error in quantization step: {str(e)}", exc_info=True)
            raise

    def run_greyscale_step(self, quantize_data):
        """Run only the greyscale conversion step using quantized data"""
        try:
            logger.debug("Starting greyscale step")
            self.progress_updated.emit(10, "Preparing greyscale conversion...")

            # Use quantized data from previous step
            quantized_array = quantize_data['quantized_array']
            unique_colors = quantize_data['unique_colors']
            width, height = quantize_data['dimensions']
            palette_size = int(quantize_data.get('palette_size') or len(unique_colors))

            self.progress_updated.emit(20, "Sorting colors and assigning greyscale values...")

            # Sort colors using perceptual sorting to keep similar colors together
            color_tuples = [tuple(color) for color in unique_colors]
            sorted_colors = perceptual_color_sort(color_tuples)

            # Debug: Check the color distribution in the sorted list
            logger.debug("Color distribution in sorted Palette:")
            hue_groups = defaultdict(list)
            for color in sorted_colors:
                r, g, b = color
                # Simple hue classification
                if r > g and r > b:
                    hue_groups['red'].append(color)
                elif g > r and g > b:
                    hue_groups['green'].append(color)
                elif b > r and b > g:
                    hue_groups['blue'].append(color)
                elif r == g == b:
                    hue_groups['gray'].append(color)
                elif r > b and g > b:
                    hue_groups['yellow'].append(color)
                elif g > r and b > r:
                    hue_groups['cyan'].append(color)
                elif r > g and b > g:
                    hue_groups['magenta'].append(color)
                else:
                    hue_groups['other'].append(color)

            for hue, colors_in_group in hue_groups.items():
                logger.debug(f"  {hue}: {len(colors_in_group)} colors")

            # Create mapping: color -> greyscale value
            color_to_grey = {}
            grey_to_color = {}

            for grey_value, color in enumerate(sorted_colors):
                color_to_grey[color] = grey_value
                grey_to_color[grey_value] = np.array(color, dtype=np.uint8)

            logger.debug("Color to greyscale mapping created")

            self.progress_updated.emit(60, "Creating greyscale image...")

            # Create greyscale image by replacing colors with assigned greyscale values
            greyscale_array = np.zeros((height, width), dtype=np.uint8)

            for i in range(height):
                for j in range(width):
                    color = tuple(quantized_array[i, j])
                    greyscale_array[i, j] = color_to_grey[color]

            # Convert to RGB for display
            greyscale_image = Image.fromarray(greyscale_array, 'L')
            # Scale greyscale values to 0-255 for better display
            display_array = (greyscale_array * (255 / (palette_size - 1))).astype(np.uint8)
            greyscale_rgb = Image.fromarray(display_array, 'L').convert('RGB')

            # Process greyscale-conversion textures using base color mapping
            greyscale_textures_results = []
            gs_sources = quantize_data.get('greyscale_textures', [])
            if gs_sources:
                self.progress_updated.emit(80, f"Mapping {len(gs_sources)} greyscale conversion texture(s)...")
                palette = np.array(sorted_colors, dtype=np.int16)  # use int16 to avoid uint8 overflow during diff

                def map_image_to_greyscale(img_arr_rgb):
                    h, w = img_arr_rgb.shape[:2]
                    flat = img_arr_rgb.reshape(-1, 3).astype(np.int16)
                    # chunked nearest palette index computation to control memory
                    idx_all = np.empty((flat.shape[0],), dtype=np.int32)
                    chunk = 200000
                    for start in range(0, flat.shape[0], chunk):
                        end = min(start + chunk, flat.shape[0])
                        block = flat[start:end][:, None, :]  # (n,1,3)
                        diffs = block - palette[None, :, :]  # (n,lut,3)
                        d2 = np.sum(diffs * diffs, axis=2)  # (n,lut)
                        idx_all[start:end] = np.argmin(d2, axis=1)
                    idx_img = idx_all.reshape(h, w).astype(np.uint16)
                    # map to 0..palette_size-1 greys (already indices)
                    gs_indices = idx_img.astype(np.uint8)
                    disp = (gs_indices * (255 / (palette_size - 1))).astype(np.uint8)
                    gs_img = Image.fromarray(disp, 'L').convert('RGB')
                    return gs_indices, gs_img

                for entry in gs_sources:
                    try:
                        proc_img = entry.get('processed_color_image')
                        proc_arr = None
                        if proc_img is not None:
                            proc_arr = np.array(proc_img)
                        else:
                            proc_arr = entry.get('processed_color_array')
                        if proc_arr is None:
                            # fall back to their quantized array if present
                            proc_arr = entry.get('quantized_array')
                        if proc_arr is None:
                            continue
                        gs_indices, gs_img = map_image_to_greyscale(proc_arr)
                        greyscale_textures_results.append({
                            'path': entry.get('path'),
                            'processed_color_image': proc_img if proc_img is not None else Image.fromarray(
                                proc_arr.astype('uint8'), 'RGB'),
                            'greyscale_image': gs_img,
                            'processed_dimensions': entry.get('processed_dimensions')
                        })
                    except Exception as ex:
                        logger.warning(f"Failed to map greyscale texture '{entry.get('path')}': {ex}")

            self.progress_updated.emit(100, "Greyscale conversion complete!")

            # Prepare results for greyscale step
            results = {
                'success': True,
                'step': 'greyscale',
                'greyscale_image': greyscale_rgb,
                'greyscale_array': greyscale_array,
                'color_to_grey': color_to_grey,
                'grey_to_color': grey_to_color,
                'sorted_colors': sorted_colors,
                'palette_size': palette_size,
                'greyscale_textures_results': greyscale_textures_results
            }

            logger.debug("Greyscale step completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error in greyscale step: {str(e)}", exc_info=True)
            raise

    def run_palette_step(self, greyscale_data, quantize_data):
        """Run only the Palette creation step using greyscale and quantized data"""
        try:
            logger.debug("Starting Palette step")
            self.progress_updated.emit(10, "Creating Palette...")

            # Use data from previous steps
            grey_to_color = greyscale_data['grey_to_color']
            greyscale_array = greyscale_data['greyscale_array']
            quantized_array = quantize_data['quantized_array']
            width, height = quantize_data['dimensions']
            palette_size = int(greyscale_data.get('palette_size') or quantize_data.get('palette_size') or len(greyscale_data.get('sorted_colors', [])) or len(quantize_data.get('unique_colors', [])) or 256)

            # Calculate Palette dimensions base (will expand when extras present)
            palette_width = palette_size

            # Base Palette from primary image mapping
            base_palette_array = np.zeros((palette_size, 3), dtype=np.uint8)
            for grey_value in range(palette_size):
                base_palette_array[grey_value] = grey_to_color[grey_value]

            # Build additional Palette arrays for each extra image using dominant color per grey index
            extra_palette_arrays = []
            extra_sources = quantize_data.get('extra_images', [])
            if extra_sources:
                # Build masks once for each grey_value
                # greyscale_array contains indices 0..palette_size-1
                height_g, width_g = greyscale_array.shape
                # Precompute positions for each grey index to speed up
                positions_by_grey = {}
                for g in range(palette_size):
                    positions = np.argwhere(greyscale_array == g)
                    positions_by_grey[g] = positions

                for extra in extra_sources:
                    extra_arr = extra.get('quantized_array')
                    if extra_arr is None:
                        continue
                    # Ensure dimensions match; if not, resize nearest
                    if extra_arr.shape[:2] != (height_g, width_g):
                        try:
                            img = Image.fromarray(extra_arr.astype('uint8'), 'RGB')
                            img = img.resize((width_g, height_g), Image.Resampling.NEAREST)
                            extra_arr = np.array(img)
                        except Exception:
                            pass
                    palette_for_extra = np.zeros((palette_size, 3), dtype=np.uint8)
                    for g in range(palette_size):
                        positions = positions_by_grey.get(g)
                        if positions is None or positions.size == 0:
                            # Fallback to base color when no pixels for this grey
                            palette_for_extra[g] = base_palette_array[g]
                            continue
                        # Gather colors from the extra image at these positions
                        colors = extra_arr[positions[:, 0], positions[:, 1]]
                        if colors.size == 0:
                            palette_for_extra[g] = base_palette_array[g]
                            continue
                        # Compute dominant color (mode)
                        tuples = [tuple(c) for c in colors]
                        most_common = Counter(tuples).most_common(1)
                        if most_common:
                            palette_for_extra[g] = np.array(most_common[0][0], dtype=np.uint8)
                        else:
                            palette_for_extra[g] = base_palette_array[g]
                    extra_palette_arrays.append(palette_for_extra)

            # Compose all Palette blocks: base + extras
            all_blocks = [base_palette_array] + extra_palette_arrays if extra_palette_arrays else [base_palette_array]
            num_blocks = len(all_blocks)
            # Calculate height based on settings
            greyscale_header_rows = 0
            required_height = greyscale_header_rows + self.palette_row_height * num_blocks
            palette_height = next_power_of_2(required_height)  # Ensure height is power of 2

            # Create Palette image
            palette_image_array = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)

            # Color Palette blocks: specified rows per block (base + extras)
            for block_idx, block_array in enumerate(all_blocks):
                start_row = greyscale_header_rows + block_idx * self.palette_row_height
                end_row = start_row + self.palette_row_height
                for row in range(start_row, end_row):
                    for col in range(palette_width):
                        if col < palette_size:
                            palette_image_array[row, col] = block_array[col]
                        else:
                            palette_image_array[row, col] = [0, 0, 0]

            # Fill any remaining rows (padding to reach power of 2 height) with greyscale gradient pattern
            filled_rows = greyscale_header_rows + self.palette_row_height * num_blocks
            if filled_rows < palette_height:
                for row in range(filled_rows, palette_height):
                    for col in range(palette_width):
                        # Use greyscale gradient pattern for padding
                        grey_value = int(col * (255 / (palette_width - 1)))  # Scale to 0-255
                        palette_image_array[row, col] = [grey_value, grey_value, grey_value]

            palette_image = Image.fromarray(palette_image_array.astype('uint8'), 'RGB')

            self.progress_updated.emit(50, "Applying Palette to greyscale for preview...")

            # Create preview: Apply Palette to greyscale image
            preview_array = self.apply_palette_to_greyscale(greyscale_array, base_palette_array, palette_size)
            preview_image = Image.fromarray(preview_array.astype('uint8'), 'RGB')

            if self.produce_color_report:
                self.progress_updated.emit(80, "Creating color report...")
                color_distribution = analyze_color_distribution(quantized_array)
                color_report_data = []
                for grey_value in range(palette_size):
                    color = tuple(base_palette_array[grey_value])  # Convert to tuple
                    frequency = color_distribution.get(color, 0)
                    color_report_data.append({
                        'grey_value': int(grey_value),
                        'color_rgb': [int(color[0]), int(color[1]), int(color[2])],
                        'frequency': int(frequency),
                        'frequency_percent': float((frequency / (height * width)) * 100)
                    })
            else:
                color_distribution = {}
                color_report_data = []

            self.progress_updated.emit(100, "Palette creation complete!")

            # Prepare results for Palette step
            results = {
                'success': True,
                'step': 'palette',
                'palette_image': palette_image,
                'preview_image': preview_image,
                'palette_array': base_palette_array,
                'color_report_data': color_report_data,
                'color_distribution': color_distribution,
                'palette_dimensions': (palette_width, palette_height),
                'palette_size': palette_size,
                'num_blocks': num_blocks,
                'block_sources': ['base'] + [os.path.basename(x.get('path', 'extra')) for x in
                                             quantize_data.get('extra_images', [])],
                'working_resolution': quantize_data.get('working_resolution'),
                'base_processed_dimensions': quantize_data.get('base_processed_dimensions'),
                'debug_info': {
                    'palette_structure': f'Top {greyscale_header_rows} rows: No header, then {num_blocks} block(s) of {self.palette_row_height} rows each ({palette_size} colors per block)'
                }
            }

            logger.debug("Palette step completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error in Palette step: {str(e)}", exc_info=True)
            raise

    def apply_palette_to_greyscale(self, greyscale_array, palette_array, palette_size):
        """Apply Palette to greyscale image to create preview (no sRGB compensation)"""
        logger.debug("Applying Palette to greyscale image")
        height, width = greyscale_array.shape
        result = np.zeros((height, width, 3), dtype=np.uint8)
        mask = greyscale_array < palette_size
        result[mask] = palette_array[greyscale_array[mask]]
        logger.debug("Palette application completed")
        return result

    def save_all_results(self):
        """Save all results after completing all steps"""
        try:
            logger.debug("Saving all results")
            self.progress_updated.emit(90, "Saving files...")

            # Verify we have all required data
            required_steps = ['quantize', 'greyscale', 'palette']
            missing_steps = [step for step in required_steps if step not in self.previous_data]

            if missing_steps:
                error_msg = f"Missing required data for steps: {missing_steps}. Cannot save results."
                logger.error(error_msg)
                self.error_occurred.emit(error_msg)
                return

            # Get data from all steps
            quantize_data = self.previous_data['quantize']
            greyscale_data = self.previous_data['greyscale']
            palette_data = self.previous_data['palette']

            # Extract data with safety checks
            if not all(key in quantize_data for key in ['quantized_image', 'unique_colors', 'dimensions']):
                error_msg = "Quantize data is incomplete"
                logger.error(error_msg)
                self.error_occurred.emit(error_msg)
                return

            if not all(key in greyscale_data for key in ['greyscale_image', 'color_to_grey', 'grey_to_color']):
                error_msg = "Greyscale data is incomplete"
                logger.error(error_msg)
                self.error_occurred.emit(error_msg)
                return

            if not all(key in palette_data for key in ['palette_image', 'preview_image', 'color_report_data']):
                error_msg = "Palette data is incomplete"
                logger.error(error_msg)
                self.error_occurred.emit(error_msg)
                return

            quantized_rgb = quantize_data['quantized_image']
            greyscale_rgb = greyscale_data['greyscale_image']
            palette_image = palette_data['palette_image']
            preview_image = palette_data['preview_image']
            color_to_grey = greyscale_data['color_to_grey']
            grey_to_color = greyscale_data['grey_to_color']
            unique_colors = quantize_data['unique_colors']
            color_distribution = palette_data.get('color_distribution', {})
            color_report_data = palette_data['color_report_data']
            width, height = quantize_data['dimensions']
            palette_size = int(palette_data.get('palette_size') or greyscale_data.get('palette_size') or quantize_data.get('palette_size') or len(quantize_data.get('unique_colors', [])) or 256)
            palette_width, palette_height = palette_data['palette_dimensions']

            # Simplified base name (no method/size suffixes)
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]

            # Determine output format based on input format
            source_is_dds = self.image_path.lower().endswith('.dds')
            output_extension = ".dds" if source_is_dds else ".png"

            output_files = {}

            # Save core outputs in the appropriate format
            greyscale_path = os.path.join(self.output_dir, f"{base_name}_greyscale{output_extension}")
            if source_is_dds:
                # For DDS output, use texconv when available; keep temps inside output_dir and always clean them up
                if cfg.get(cfg.texconv_file) and os.path.exists(cfg.get(cfg.texconv_file)):
                    temp_greyscale_path = os.path.join(self.output_dir, f"{base_name}_greyscale_temp.png")
                    try:
                        greyscale_rgb.save(temp_greyscale_path)
                        convert_to_dds(temp_greyscale_path, greyscale_path, cfg.get(cfg.texconv_file))
                    finally:
                        try:
                            if os.path.exists(temp_greyscale_path):
                                os.remove(temp_greyscale_path)
                        except Exception as _cleanup_ex:
                            logger.warning(f"Failed to remove temp file {temp_greyscale_path}: {_cleanup_ex}")
                else:
                    logger.warning("texconv.exe not found, saving as PNG instead of DDS")
                    greyscale_path = os.path.join(self.output_dir, f"{base_name}_greyscale.png")
                    greyscale_rgb.save(greyscale_path)
            else:
                greyscale_rgb.save(greyscale_path)
            output_files['greyscale'] = greyscale_path
            logger.debug(f"Saved greyscale image: {greyscale_path}")

            palette_path = os.path.join(self.output_dir, f"{base_name}_palette{output_extension}")
            if source_is_dds:
                # For DDS output, use texconv when available; keep temps inside output_dir and always clean them up
                if cfg.get(cfg.texconv_file) and os.path.exists(cfg.get(cfg.texconv_file)):
                    temp_palette_path = os.path.join(self.output_dir, f"{base_name}_palette_temp.png")
                    try:
                        palette_image.save(temp_palette_path)
                        convert_to_dds(temp_palette_path, palette_path, cfg.get(cfg.texconv_file), is_palette=True,
                                            palette_width=palette_width, palette_height=palette_height)
                    finally:
                        try:
                            if os.path.exists(temp_palette_path):
                                os.remove(temp_palette_path)
                        except Exception as _cleanup_ex:
                            logger.warning(f"Failed to remove temp file {temp_palette_path}: {_cleanup_ex}")
                else:
                    logger.warning("texconv.exe not found, saving as PNG instead of DDS")
                    palette_path = os.path.join(self.output_dir, f"{base_name}_palette.png")
                    palette_image.save(palette_path)
            else:
                palette_image.save(palette_path)
            output_files['palette'] = palette_path
            logger.debug(f"Saved Palette: {palette_path}")

            # Save greyscale-conversion textures in the appropriate format
            saved_gs_textures = []
            saved_color_textures = []
            gs_tex_list = greyscale_data.get('greyscale_textures_results', [])
            for idx, entry in enumerate(gs_tex_list, start=1):
                try:
                    color_img = entry.get('processed_color_image')
                    grey_img = entry.get('greyscale_image')
                    if color_img is None or grey_img is None:
                        continue

                    color_out = os.path.join(self.output_dir, f"{base_name}_texture{idx}{output_extension}")
                    grey_out = os.path.join(self.output_dir, f"{base_name}_greyscaletexture_{idx}{output_extension}")

                    if source_is_dds:
                        # For DDS output, save temp PNG and convert. Always clean up temps.
                        temp_color_path = os.path.join(self.output_dir, f"{base_name}_texture{idx}_temp.png")
                        temp_grey_path = os.path.join(self.output_dir, f"{base_name}_greyscaletexture_{idx}_temp.png")
                        try:
                            color_img.save(temp_color_path)
                            grey_img.save(temp_grey_path)
                            if cfg.get(cfg.texconv_file) and os.path.exists(cfg.get(cfg.texconv_file)):
                                convert_to_dds(temp_color_path, color_out, cfg.get(cfg.texconv_file))
                                convert_to_dds(temp_grey_path, grey_out, cfg.get(cfg.texconv_file))
                            else:
                                logger.warning("texconv.exe not found, saving as PNG instead of DDS")
                                color_out = os.path.join(self.output_dir, f"{base_name}_texture{idx}.png")
                                grey_out = os.path.join(self.output_dir, f"{base_name}_greyscaletexture_{idx}.png")
                                color_img.save(color_out)
                                grey_img.save(grey_out)
                        finally:
                            for _p in (temp_color_path, temp_grey_path):
                                try:
                                    if os.path.exists(_p):
                                        os.remove(_p)
                                except Exception as _cleanup_ex:
                                    logger.warning(f"Failed to remove temp file {_p}: {_cleanup_ex}")
                    else:
                        color_img.save(color_out)
                        grey_img.save(grey_out)

                    saved_color_textures.append(color_out)
                    saved_gs_textures.append(grey_out)
                except Exception as ex:
                    logger.warning(f"Failed saving greyscale conversion texture {idx}: {ex}")
            if saved_color_textures:
                output_files['textures'] = saved_color_textures
            if saved_gs_textures:
                output_files['greyscale_textures'] = saved_gs_textures

            # Optional: Save color report only when enabled
            color_report_path = None
            if self.produce_color_report and color_report_data:
                color_report_path = os.path.join(self.output_dir, f"{base_name}_color_report.json")
                self.save_color_report(color_report_data, color_report_path)
                output_files['color_report'] = color_report_path
                logger.debug(f"Saved color report: {color_report_path}")

            # Generate DDS if source was DDS (mirror simplified names) - initialize dds_files
            dds_files = {}
            source_is_dds = self.image_path.lower().endswith('.dds')
            texconv_path = cfg.get(cfg.texconv_file)

            if source_is_dds and texconv_path and os.path.exists(texconv_path):
                try:
                    # Convert greyscale to DDS
                    temp_greyscale_path = os.path.join(self.output_dir, f"{base_name}_greyscale_temp.png")
                    try:
                        greyscale_rgb.save(temp_greyscale_path)
                        greyscale_dds_path = os.path.join(self.output_dir, f"{base_name}_greyscale.dds")
                        convert_to_dds(temp_greyscale_path, greyscale_dds_path, texconv_path)
                        dds_files['greyscale'] = greyscale_dds_path
                    finally:
                        try:
                            if os.path.exists(temp_greyscale_path):
                                os.remove(temp_greyscale_path)
                        except Exception as _cleanup_ex:
                            logger.warning(f"Failed to remove temp file {temp_greyscale_path}: {_cleanup_ex}")

                    # Convert Palette to DDS
                    temp_palette_path = os.path.join(self.output_dir, f"{base_name}_palette_temp.png")
                    try:
                        palette_image.save(temp_palette_path)
                        palette_dds_path = os.path.join(self.output_dir, f"{base_name}_palette.dds")
                        convert_to_dds(temp_palette_path, palette_dds_path, texconv_path, is_palette=True, palette_width=palette_width,
                                            palette_height=palette_height)
                        dds_files['palette'] = palette_dds_path
                    finally:
                        try:
                            if os.path.exists(temp_palette_path):
                                os.remove(temp_palette_path)
                        except Exception as _cleanup_ex:
                            logger.warning(f"Failed to remove temp file {temp_palette_path}: {_cleanup_ex}")

                    # Convert greyscale conversion textures to DDS
                    for idx, (cpath, gpath) in enumerate(zip(saved_color_textures, saved_gs_textures), start=1):
                        try:
                            # Create temp files for conversion
                            temp_c_path = os.path.join(self.output_dir, f"temp_texture{idx}.png")
                            temp_g_path = os.path.join(self.output_dir, f"temp_greyscaletexture_{idx}.png")

                            # Load and save temp files
                            Image.open(cpath).save(temp_c_path)
                            Image.open(gpath).save(temp_g_path)

                            c_dds = os.path.join(self.output_dir, f"{base_name}_texture{idx}.dds")
                            g_dds = os.path.join(self.output_dir, f"{base_name}_greyscaletexture_{idx}.dds")
                            convert_to_dds(temp_c_path, c_dds, texconv_path)
                            convert_to_dds(temp_g_path, g_dds, texconv_path)
                            dds_files.setdefault('textures_dds', []).append(c_dds)
                            dds_files.setdefault('greyscale_textures_dds', []).append(g_dds)
                        except Exception as ex:
                            logger.warning(f"DDS mirror failed for greyscale texture {idx}: {ex}")
                        finally:
                            for _p in (temp_c_path, temp_g_path):
                                try:
                                    if os.path.exists(_p):
                                        os.remove(_p)
                                except Exception as _cleanup_ex:
                                    logger.warning(f"Failed to remove temp file {_p}: {_cleanup_ex}")
                except Exception as e:
                    logger.error(f"DDS conversion during save failed: {e}")

            # Optional: Save metadata only when enabled
            metadata_path = None
            if self.produce_metadata_json:
                metadata = {
                    'timestamp': datetime.now().isoformat(),
                    'original_image': self.image_path,
                    'quantization_method': cfg.get(cfg.ci_default_quant_method),
                    'palette_size': palette_size,
                    'working_resolution': quantize_data.get('working_resolution', None),
                    'base_original_dimensions': f"{quantize_data.get('base_original_dimensions', (width, height))[0]}x{quantize_data.get('base_original_dimensions', (width, height))[1]}",
                    'base_processed_dimensions': f"{width}x{height}",
                    'quantization_info': quantize_data['quantization_info'],
                    'dimensions': {
                        'original': f"{width}x{height}",
                        'greyscale': f"{width}x{height}",
                        'palette': f"{palette_width}x{palette_height}"
                    },
                    'color_mapping': {str([int(k[0]), int(k[1]), int(k[2])]): int(v) for k, v in color_to_grey.items()},
                    'unique_colors': len(unique_colors),
                    'sorting_method': 'perceptual_luminance_hue',
                    'color_distribution_summary': {
                        'most_common_color': [int(c) for c in max(color_distribution,
                                                                  key=color_distribution.get)] if color_distribution else None,
                        'most_common_frequency': max(color_distribution.values()) if color_distribution else 0,
                        'total_pixels': height * width
                    },
                    'dds_generated': source_is_dds and bool(dds_files),
                    'palette_texture_dimensions': f'{palette_width}x{palette_height} (Power of Two)',
                    'num_blocks': palette_data.get('num_blocks', 1),
                    'block_sources': palette_data.get('block_sources', ['base']),
                    'extra_images': [x.get('path') for x in quantize_data.get('extra_images', [])],
                    'extra_images_details': [
                        {
                            'path': x.get('path'),
                            'original_dimensions': f"{x.get('original_dimensions', x.get('dimensions', (0, 0)))[0]}x{x.get('original_dimensions', x.get('dimensions', (0, 0)))[1]}",
                            'processed_dimensions': f"{x.get('processed_dimensions', x.get('dimensions', (0, 0)))[0]}x{x.get('processed_dimensions', x.get('dimensions', (0, 0)))[1]}"
                        }
                        for x in quantize_data.get('extra_images', [])
                    ],
                    'greyscale_conversion_textures': [x.get('path') for x in
                                                      quantize_data.get('greyscale_textures', [])]
                }

                metadata_path = os.path.join(self.output_dir, f"{base_name}_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.debug(f"Saved metadata: {metadata_path}")

            # Merge DDS file paths into output files
            output_files.update(dds_files)

            results = {
                'success': True,
                'original_image': quantize_data['original_image'],
                'quantized_image': quantized_rgb,
                'greyscale_image': greyscale_rgb,
                'palette_image': palette_image,
                'preview_image': preview_image,
                'color_report_data': color_report_data,
                'color_distribution': color_distribution,
                'quantization_method': cfg.get(cfg.ci_default_quant_method),
                'palette_size': palette_size,
                'srgb_compensation_applied': self.srgb_compensation,
                'output_files': output_files,
                'mapping_info': {
                    'color_to_grey': color_to_grey,
                    'grey_to_color': grey_to_color,
                    'unique_colors': len(unique_colors),
                    'palette_array': palette_data.get('palette_array', np.zeros((palette_size, 3), dtype=np.uint8)).tolist()
                }
            }

            self.progress_updated.emit(100, "Complete!")
            self.result_ready.emit(results)
            logger.debug("All results saved successfully")

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}", exc_info=True)
            raise

    def save_color_report(self, color_report_data, file_path):
        """Save color report data as JSON file"""
        try:
            # Convert color_report_data to JSON-serializable format
            serializable_data = []
            for color_data in color_report_data:
                serializable_data.append({
                    'grey_value': int(color_data['grey_value']),
                    'color_rgb': [
                        int(color_data['color_rgb'][0]),
                        int(color_data['color_rgb'][1]),
                        int(color_data['color_rgb'][2])
                    ],
                    'frequency': int(color_data['frequency']),
                    'frequency_percent': float(color_data['frequency_percent'])
                })

            # Create a structured report
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_colors': len(serializable_data),
                    'most_used_colors': sorted(serializable_data, key=lambda x: x['frequency'], reverse=True)[:10],
                    'least_used_colors': sorted(serializable_data, key=lambda x: x['frequency'])[:10]
                },
                'color_mapping': serializable_data
            }

            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)

            logger.debug(f"Color report saved to: {file_path}")
        except Exception as e:
            logger.error(f"Error saving color report: {str(e)}")
            raise


class ColorReportWidget(QWidget):
    def __init__(self, color_report_data):
        super().__init__()
        self.color_report_data = color_report_data
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title = QLabel(f"Color to Greyscale Mapping Report - {cfg.get(cfg.ci_default_quant_method)} ({len(self.color_report_data)} colors)")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)

        # Description
        desc = QLabel(
            f"All {len(self.color_report_data)} colors with their assigned greyscale values. Colors are sorted perceptually by luminance and hue to keep similar colors together. Frequency shows how often each color appears in the image.")
        desc.setWordWrap(True)
        desc.setStyleSheet("margin: 5px; color: #666;")
        layout.addWidget(desc)

        # Create scroll area for the color grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QGridLayout(scroll_content)

        # Headers
        headers = ["Grey Value", "Color", "RGB Values", "Frequency", "Percent"]
        for col, header in enumerate(headers):
            label = QLabel(header)
            label.setStyleSheet("font-weight: bold; background-color: #e0e0e0; padding: 5px; border: 1px solid #ccc;")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            scroll_layout.addWidget(label, 0, col)

        # Add color rows
        for row, color_data in enumerate(self.color_report_data, 1):
            grey_value = color_data['grey_value']
            color_rgb = color_data['color_rgb']
            frequency = color_data['frequency']
            percent = color_data['frequency_percent']

            # Grey value
            grey_label = QLabel(str(grey_value))
            grey_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grey_label.setStyleSheet("padding: 5px; border: 1px solid #ccc;")
            scroll_layout.addWidget(grey_label, row, 0)

            # Color swatch
            color_widget = QLabel()
            color_widget.setMinimumSize(60, 30)
            color_widget.setStyleSheet(
                f"background-color: rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}); border: 1px solid #ccc;")
            color_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            scroll_layout.addWidget(color_widget, row, 1)

            # RGB values
            rgb_label = QLabel(f"R: {color_rgb[0]:3d} G: {color_rgb[1]:3d} B: {color_rgb[2]:3d}")
            rgb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            rgb_label.setStyleSheet("padding: 5px; border: 1px solid #ccc; font-family: monospace;")
            scroll_layout.addWidget(rgb_label, row, 2)

            # Frequency
            freq_label = QLabel(f"{frequency:,}")
            freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            freq_label.setStyleSheet("padding: 5px; border: 1px solid #ccc; font-family: monospace;")
            scroll_layout.addWidget(freq_label, row, 3)

            # Percentage
            percent_label = QLabel(f"{percent:.2f}%")
            percent_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # Color code based on frequency
            if percent > 5:
                percent_label.setStyleSheet(
                    "padding: 5px; border: 1px solid #ccc; background-color: #ffcccc; font-family: monospace;")
            elif percent > 1:
                percent_label.setStyleSheet(
                    "padding: 5px; border: 1px solid #ccc; background-color: #ffffcc; font-family: monospace;")
            else:
                percent_label.setStyleSheet(
                    "padding: 5px; border: 1px solid #ccc; background-color: #ccffcc; font-family: monospace;")
            scroll_layout.addWidget(percent_label, row, 4)

        scroll_layout.setColumnStretch(2, 1)
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        # Summary
        total_colors = len(self.color_report_data)
        dominant_colors = sorted(self.color_report_data, key=lambda x: x['frequency'], reverse=True)[:5]
        dominant_summary = " | ".join(
            [f"Grey {c['grey_value']}: {c['frequency_percent']:.1f}%" for c in dominant_colors])

        summary = QLabel(f"Total colors: {total_colors} | Most used: {dominant_summary}")
        summary.setStyleSheet("margin: 10px; padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc;")
        layout.addWidget(summary)


from PySide6.QtCore import Signal


class PaletteGenerator(BaseWidget):
    log_signal = Signal(str)

    def __init__(self, parent, text):
        super().__init__(parent=parent, text=text, vertical=True)
        self.main_widget = QWidget()
        # Persistent settings via qfluentwidgets ConfigItem
        self.ci_last_image_dir = ConfigItem("palette", "last_image_dir", "")
        self.ci_last_output_dir = ConfigItem("palette", "last_output_dir", "")
        self.current_image_path = None
        self.output_dir = None
        self.current_results = None
        self.quantized_data = None
        self.greyscale_data = None
        self.palette_data = None
        self.original_preview_label = None
        self.greyscale_preview_label = None
        self.quantized_preview_label = None
        self.preview_label = None
        self.previous_data = {}  # Store data from previous steps
        self.extra_image_paths = []  # Additional textures to include as Palette blocks
        self.greyscale_texture_paths = []  # Textures to convert to greyscale using base mapping
        self.pivot = SegmentedWidget(self)
        self.stackedWidget = QStackedWidget(self)
        self.init_ui(parent)
        self.pivot.setCurrentItem(self.main_tab.objectName())
        self.boxLayout.addWidget(self.pivot, 0, Qt.AlignmentFlag.AlignLeft)
        self.boxLayout.addWidget(self.stackedWidget)
        self.buttons_layout.addWidget(self.reset_greyscale_button, stretch=0)
        self.buttons_layout.addWidget(self.reset_extra_button, stretch=0)
        self.addButtonBarToBottom(self.generate_all_button)
        self.settings_widget = PaletteSettings(self)
        self.settings_drawer.addWidget(self.settings_widget)
        self.help_widget = PaletteHelp(self)
        self.help_drawer.addWidget(self.help_widget)


    def addSubInterface(self, widget: QWidget, objectName, text):
        widget.setObjectName(objectName)
        self.stackedWidget.addWidget(widget)
        self.pivot.addItem(
            routeKey=objectName,
            text=text,
            onClick=lambda: self.stackedWidget.setCurrentWidget(widget)
        )

    def onCurrentIndexChanged(self, index):
        widget = self.stackedWidget.widget(index)
        self.pivot.setCurrentItem(widget.objectName())

    def init_ui(self, parent):
        self.main_tab = QWidget()
        main_layout = QVBoxLayout(self.main_tab)
        main_layout.setContentsMargins(0,0,0,0)

        self.setup_main_tab(main_layout)

        self.report_tab = QWidget()
        self.report_layout = QVBoxLayout(self.report_tab)
        self.report_placeholder = QLabel("Generate a Palette first to see the color report")
        self.report_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.report_placeholder.setStyleSheet("font-size: 14px; color: #666; margin: 50px;")
        self.report_layout.addWidget(self.report_placeholder)

        parent.setStatusTip("Ready to select an image")
        logger.debug("UI initialization complete")

        self.addSubInterface(self.main_tab, 'mainTab', 'Generator')
        self.addSubInterface(self.report_tab, 'reportTab', 'Color Report')

        self.stackedWidget.currentChanged.connect(self.onCurrentIndexChanged)
        self.stackedWidget.setCurrentWidget(self.main_tab)


    def setup_main_tab(self, layout):

        # --- Image Selection Cards ---
        self.base_image_card = PushSettingCard(
            self.tr("Base Image"),
            CustomIcons.IMAGE.icon(stroke=True),
            self.tr("Source Image for Greyscale Template / Palette"),
            "No image selected"
        )
        self.base_image_card.clicked.connect(self.on_base_image_card)
        layout.addWidget(self.base_image_card)

        self.extra_images_card = PushSettingCard(
            self.tr("Additional Textures"),
            CustomIcons.IMAGEADD.icon(stroke=True),
            self.tr("Texture to create more color rows (Optional, multi-select)"),
            self.tr("None selected")
        )
        self.extra_images_card.clicked.connect(self.on_extra_images_card)
        layout.addWidget(self.extra_images_card)

        self.greyscale_images_card = PushSettingCard(
            self.tr("Greyscale Conversion Textures"),
            CustomIcons.GREYSCALE.icon(),
            self.tr("Textures to convert to greyscale using the palette (Optional, multi-select)"),
            self.tr("None selected")
        )
        self.greyscale_images_card.clicked.connect(self.on_greyscale_images_card)
        layout.addWidget(self.greyscale_images_card)

        self.output_dir_card = PushSettingCard(
            self.tr("Output Directory"),
            CustomIcons.FOLDERRIGHT.icon(),
            self.tr("Where generated files will be written. Defaults to the image folder."),
            self.tr("Will use image directory")
        )
        self.output_dir_card.clicked.connect(self.on_output_dir_card)
        layout.addWidget(self.output_dir_card)


        # Generate button
        self.generate_all_button = PrimaryPushButton(icon=FIF.RIGHT_ARROW, text="Generate")
        self.generate_all_button.clicked.connect(lambda: self.generate_step('all'))
        self.generate_all_button.setEnabled(False)

        # Reset buttons (next to Generate)
        self.reset_extra_button = PushButton(text="Reset Additional Textures")
        self.reset_extra_button.clicked.connect(self.clear_extra_images)
        self.reset_greyscale_button = PushButton(text="Reset Greyscales")
        self.reset_greyscale_button.clicked.connect(self.clear_greyscale_textures)

        # Preview area (4 previews horizontally)
        preview_splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)

        def make_preview_group(title, attr_name):
            group = QtWidgets.QGroupBox(title)
            group_layout = QtWidgets.QVBoxLayout(group)
            label = QtWidgets.QLabel(f"{title}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(300, 300)
            group_layout.addWidget(label)
            setattr(self, attr_name, label)
            return group

        preview_splitter.addWidget(make_preview_group("Original Image", "original_preview_label"))
        preview_splitter.addWidget(make_preview_group("Quantized", "quantized_preview_label"))
        preview_splitter.addWidget(make_preview_group("Greyscale (Color → Grey Mapping)", "greyscale_preview_label"))
        preview_splitter.addWidget(make_preview_group("Preview (Palette Applied to Greyscale)", "preview_label"))

        preview_splitter.setSizes([400, 400, 400, 400])
        layout.addWidget(preview_splitter)

        # Progress masked by parent window; no local ProgressBar



    def on_base_image_card(self):
        last_dir = str(self.ci_last_image_dir.value or "")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Image for Palette Generation"),
            last_dir,
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.dds *.tga);;All Files (*)"
        )
        if file_path:
            logger.debug(f"Image selected: {file_path}")
            self.current_image_path = file_path
            try:
                self.ci_last_image_dir.value = os.path.dirname(file_path)
            except Exception:
                pass
            # Default output to image directory if not explicitly set or previously unset
            self.output_dir = os.path.dirname(file_path)
            try:
                self.base_image_card.setContent(os.path.basename(file_path))
            except Exception:
                pass
            try:
                # If user hasn't chosen a custom output yet, show default
                self.output_dir_card.setContent(f"{self.tr('Using')}: {self.output_dir}")
            except Exception:
                pass

            self.load_and_display_image(file_path, self.original_preview_label, "Original Image")

            # Reset step data
            self.quantized_data = None
            self.greyscale_data = None
            self.palette_data = None
            self.previous_data = {}

            # Update button states
            self.update_button_states()

    def on_extra_images_card(self):
        last_dir = str(self.ci_last_image_dir.value or "")
        files, _ = QFileDialog.getOpenFileNames(
            self,
            self.tr("Select Additional Textures (will add 4-row blocks to Palette)"),
            last_dir,
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.dds *.tga);;All Files (*)"
        )
        if files:
            # Ensure uniqueness and avoid adding the base image itself
            new_paths = []
            for p in files:
                if p != self.current_image_path and p not in self.extra_image_paths:
                    new_paths.append(p)
            if new_paths:
                self.extra_image_paths.extend(new_paths)
                names = [os.path.basename(p) for p in self.extra_image_paths]
                if len(names) > 3:
                    display = ", ".join(names[:3]) + f" (+{len(names) - 3} more)"
                else:
                    display = ", ".join(names) if names else self.tr("None selected")
                try:
                    self.extra_images_card.setContent(display)
                except Exception:
                    pass

    def clear_extra_images(self):
        self.extra_image_paths = []
        try:
            self.extra_images_card.setContent(self.tr("None selected"))
        except Exception:
            pass

    def on_greyscale_images_card(self):
        last_dir = str(self.ci_last_image_dir.value or "")
        files, _ = QFileDialog.getOpenFileNames(
            self,
            self.tr("Select Textures to Convert to Greyscale (using base mapping)"),
            last_dir,
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.dds *.tga);;All Files (*)"
        )
        if files:
            new_paths = []
            for p in files:
                if p != self.current_image_path and p not in self.greyscale_texture_paths:
                    new_paths.append(p)
            if new_paths:
                self.greyscale_texture_paths.extend(new_paths)
                names = [os.path.basename(p) for p in self.greyscale_texture_paths]
                if len(names) > 3:
                    display = ", ".join(names[:3]) + f" (+{len(names) - 3} more)"
                else:
                    display = ", ".join(names) if names else self.tr("None selected")
                try:
                    self.greyscale_images_card.setContent(display)
                except Exception:
                    pass

    def clear_greyscale_textures(self):
        self.greyscale_texture_paths = []
        try:
            self.greyscale_images_card.setContent(self.tr("None selected"))
        except Exception:
            pass

    def on_output_dir_card(self):
        last_dir = str(self.ci_last_output_dir.value or (self.output_dir or ""))
        directory = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Output Directory"),
            last_dir,
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.output_dir = directory
            try:
                self.ci_last_output_dir.value = directory
            except Exception:
                pass
            try:
                self.output_dir_card.setContent(directory)
            except Exception:
                pass

    def load_and_display_image(self, file_path, label, description):
        try:
            logger.debug(f"Loading image for display: {file_path}")
            img = load_image(file_path, cfg.get(cfg.texconv_file))
            pixmap = self.pil_to_pixmap(img)

            scaled_pixmap = pixmap.scaled(
                label.width() - 20,
                label.height() - 20,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
            label.setText("")
            logger.debug(f"Image displayed successfully: {description}")

        except Exception as e:
            logger.error(f"Error loading {description}: {str(e)}")
            label.setText(f"Error loading {description}: {e}")

    def pil_to_pixmap(self, pil_image):
        """Convert PIL Image to QPixmap"""
        try:
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            data = pil_image.tobytes("raw", "RGB")
            qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(qimage)
        except Exception as e:
            logger.error(f"Error converting PIL to QPixmap: {str(e)}")
            # Return a blank pixmap on error
            return QPixmap(100, 100)

    def generate_step(self, step):
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "Please select an image first.")
            return

        if not self.output_dir:
            self.output_dir = os.path.dirname(self.current_image_path)
            try:
                self.output_dir_card.setContent(self.output_dir)
            except Exception:
                pass

        # Get selected Palette size from ConfigItem
        try:
            palette_size = int(cfg.get(cfg.ci_default_palette_size) or 256)
        except Exception:
            palette_size = 256

        # Get working resolution from ConfigItem
        wr_val = cfg.get(cfg.ci_default_working_res).value
        if isinstance(wr_val, str) and str(wr_val).lower().startswith('original'):
            working_resolution = None
        else:
            try:
                working_resolution = int(wr_val)
            except Exception:
                working_resolution = None

        # Check if source is DDS and if texconv is available
        source_is_dds = self.current_image_path.lower().endswith('.dds')
        texconv_path = str(cfg.get(cfg.texconv_file) or "")
        generate_dds = source_is_dds  # Only generate DDS if source is DDS

        if step in ['all', 'palette'] and generate_dds and (not texconv_path or not os.path.exists(texconv_path)):
            reply = QMessageBox.question(
                self,
                "texconv.exe not found",
                "Source image is DDS format but texconv.exe was not found or is invalid. "
                "DDS files cannot be generated. Do you want to continue without DDS generation?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
            generate_dds = False

        # Disable buttons during processing and use parent mask progress
        self.set_buttons_enabled(False)
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'show_progress'):
            try:
                p.show_progress()
            except Exception:
                pass

        logger.debug(f"Starting generation step: {step}, Palette size: {palette_size}")

        self.worker = SinglePaletteGenerationWorker(
            self.current_image_path,
            self.output_dir,
            generate_dds,
            step,
            self.previous_data,
            extra_image_paths=self.extra_image_paths,
            working_resolution=working_resolution,
            produce_color_report=bool(cfg.get(cfg.ci_produce_color_report)),
            produce_metadata_json=bool(cfg.get(cfg.ci_produce_metadata_json)),
            greyscale_texture_paths=self.greyscale_texture_paths,
            palette_row_height=int(cfg.get(cfg.ci_palette_row_height) or 4)
        )

        self.worker.progress_updated.connect(self.update_progress)
        self.worker.result_ready.connect(self.show_results)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.step_complete.connect(self.handle_step_complete)
        self.worker.start()

    def update_progress(self, value, message):
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'update_progress'):
            try:
                p.update_progress(int(value))
            except Exception:
                pass
        logger.info(f"Progress: {value}% - {message}")

    def show_results(self, results):
        self.set_buttons_enabled(True)
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'complete_loader'):
            try:
                p.complete_loader()
            except Exception:
                pass

        if results.get('success'):
            self.current_results = results

            # Update previews for all-in-one generation
            if 'quantized_image' in results:
                pixmap = self.pil_to_pixmap(results['quantized_image'])
                self.update_preview(pixmap, self.quantized_preview_label)

            if 'greyscale_image' in results:
                pixmap = self.pil_to_pixmap(results['greyscale_image'])
                self.update_preview(pixmap, self.greyscale_preview_label)

            if 'preview_image' in results:
                pixmap = self.pil_to_pixmap(results['preview_image'])
                self.update_preview(pixmap, self.preview_label)

            # Update report tab
            if 'color_report_data' in results:
                quantization_method = results.get('quantization_method', 'unknown')
                palette_size = results['palette_size']  # No fallback - should always be present
                self.update_report_tab(results['color_report_data'])

            # Show results
            output_files = results.get('output_files', {})
            quantization_method = results.get('quantization_method', 'unknown')
            palette_size = results['palette_size']  # No fallback - should always be present
            source_is_dds = self.current_image_path.lower().endswith('.dds')

            wr = results.get('working_resolution', None)
            base_dims = results.get('base_processed_dimensions') or (
                self.quantized_data.get('dimensions') if self.quantized_data else None)
            wr_str = 'Original' if not wr else f"{wr} px (long side)"
            base_dims_str = f"{base_dims[0]}x{base_dims[1]}" if base_dims else "unknown"

            result_text = f"""
=== Palette GENERATION COMPLETE ===

Method: {quantization_method}
Palette Size: {palette_size} colors
Working Resolution: {wr_str}
Processed Base Size: {base_dims_str}

Generated Files:
• Greyscale: {os.path.basename(output_files.get('greyscale', ''))}
• Palette: {os.path.basename(output_files.get('palette', ''))}
"""

            # Optional files
            if output_files.get('color_report'):
                result_text += f"• Color Report: {os.path.basename(output_files.get('color_report', ''))}\n"

            palette_width, palette_height = results.get('palette_dimensions', (palette_size, 8))
            # Determine how many header rows were included based on settings
            palette_row_height = int(cfg.get(cfg.ci_palette_row_height) or 4)
            num_blocks = results.get('num_blocks', 1)
            header_rows = 0

            result_text += f"""
Note: Palette texture is {palette_width}×{palette_height} pixels (power of two dimensions for game engines).
• Top {header_rows} rows: No header
• Bottom rows: {num_blocks} block(s) of {palette_row_height} rows each ({palette_size}-color Palette blocks)
• Colors are sorted perceptually to keep similar colors together
"""

            if bool(cfg.get(cfg.ci_produce_color_report)):
                result_text += "See the \"Color Report\" tab for detailed analysis of color distribution."

            logger.info(result_text)
            logger.info("All-in-one generation completed successfully")
        else:
            error_msg = results.get('error', 'Unknown error occurred')
            logger.error(f"Generation failed: {error_msg}")
            QMessageBox.critical(self, "Error", f"Palette generation failed: {error_msg}")

    def handle_error(self, error_message):
        self.set_buttons_enabled(True)
        logger.error(f"Worker error: {error_message}")
        QMessageBox.critical(self, "Error", f"Palette generation failed: {error_message}")
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'complete_loader'):
            try:
                p.complete_loader()
            except Exception:
                pass

    def handle_step_complete(self, step, results):
        """Handle completion of individual steps"""
        logger.debug(f"Step complete: {step}")

        if results.get('success'):
            if step == 'quantize':
                self.quantized_data = results
                self.previous_data['quantize'] = results  # Store for next steps

                # Check if color padding was needed
                actual_count = results.get('actual_color_count', results['palette_size'])
                palette_size = results['palette_size']
                if actual_count < palette_size:
                    padding_info = results.get('quantization_info', {}).get('color_padding', '')
                    warning_msg = f"⚠ Quantization produced only {actual_count} colors, padding was applied"
                    logger.warning(warning_msg)
                    if padding_info:
                        logger.info(f"   {padding_info}")
                else:
                    logger.info(f"✓ Quantization complete - Image reduced to {palette_size} colors")

                # Update quantized preview
                if 'quantized_image' in results:
                    pixmap = self.pil_to_pixmap(results['quantized_image'])
                    self.update_preview(pixmap, self.quantized_preview_label)

            elif step == 'greyscale':
                self.greyscale_data = results
                self.previous_data['greyscale'] = results  # Store for next steps
                palette_size = results['palette_size']
                logger.info(
                    f"✓ Greyscale conversion complete - {palette_size} colors mapped to greyscale values (perceptually sorted)")

                # Update greyscale preview
                if 'greyscale_image' in results:
                    pixmap = self.pil_to_pixmap(results['greyscale_image'])
                    self.update_preview(pixmap, self.greyscale_preview_label)

            elif step == 'palette':
                self.palette_data = results
                self.previous_data['palette'] = results  # Store for completeness
                palette_size = results['palette_size']
                srgb_compensation = results.get('srgb_compensation_applied', False)
                compensation_status = "with sRGB compensation" if srgb_compensation else ""

                # Update Palette and preview
                if 'preview_image' in results:
                    pixmap = self.pil_to_pixmap(results['preview_image'])
                    self.update_preview(pixmap, self.preview_label)
                if 'color_report_data' in results:
                    self.update_report_tab(results['color_report_data'])
                logger.info(
                    f"✓ Palette generation complete - {palette_size}-color lookup table created {compensation_status}")

        self.update_button_states()

    def update_preview(self, pixmap, label):
        scaled_pixmap = pixmap.scaled(
            label.width() - 20,
            label.height() - 20,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)
        label.setText("")

    def update_report_tab(self, color_report_data):
        """Update the report tab with color mapping data"""
        # Clear existing content
        for i in reversed(range(self.report_layout.count())):
            widget = self.report_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Create new report widget
        report_widget = ColorReportWidget(color_report_data)
        self.report_layout.addWidget(report_widget)

    def set_buttons_enabled(self, enabled):
        """Enable or disable interactive controls during processing"""
        # Main action
        self.generate_all_button.setEnabled(enabled and self.current_image_path is not None)
        # Setting cards and combos
        try:
            self.base_image_card.setEnabled(enabled)
            self.extra_images_card.setEnabled(enabled)
            self.greyscale_images_card.setEnabled(enabled)
            self.output_dir_card.setEnabled(enabled)
            # New reset buttons
            self.reset_extra_button.setEnabled(enabled)
            self.reset_greyscale_button.setEnabled(enabled)
        except Exception:
            pass

    def update_button_states(self):
        """Update button states based on available data (single-button workflow)"""
        self.generate_all_button.setEnabled(self.current_image_path is not None)

    def save_all_files(self):
        """Save all generated files"""
        if not self.quantized_data or not self.greyscale_data or not self.palette_data:
            QMessageBox.warning(self, "Warning", "Please complete all generation steps first.")
            return

        try:
            logger.debug("Saving all files")

            # Get base name and paths
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            method = cfg.get(cfg.ci_default_quant_method).split(" - ")[0]
            palette_size = self.quantized_data['palette_size']  # No fallback

            # Determine output format based on input format
            source_is_dds = self.current_image_path.lower().endswith('.dds')
            output_extension = ".dds" if source_is_dds else ".png"

            # Resolve texconv path from ConfigItem for this save operation
            texconv_path = str(cfg.get(cfg.texconv_file) or "")

            # Save quantized image in appropriate format
            quantized_path = os.path.join(self.output_dir,
                                          f"{base_name}_{method}_{palette_size}quantized{output_extension}")
            if source_is_dds:
                # Save as temp PNG and convert to DDS (temps in output_dir; always cleaned up)
                temp_quantized_path = os.path.join(self.output_dir,
                                                   f"{base_name}_{method}_{palette_size}quantized_temp.png")
                try:
                    self.quantized_data['quantized_image'].save(temp_quantized_path)
                    if texconv_path and os.path.exists(texconv_path):
                        self.convert_to_dds(temp_quantized_path, quantized_path, texconv_path)
                    else:
                        logger.warning("texconv.exe not found, saving as PNG instead of DDS")
                        quantized_path = os.path.join(self.output_dir,
                                                      f"{base_name}_{method}_{palette_size}quantized.png")
                        self.quantized_data['quantized_image'].save(quantized_path)
                finally:
                    try:
                        if os.path.exists(temp_quantized_path):
                            os.remove(temp_quantized_path)
                    except Exception as _cleanup_ex:
                        logger.warning(f"Failed to remove temp file {temp_quantized_path}: {_cleanup_ex}")
            else:
                self.quantized_data['quantized_image'].save(quantized_path)

            # Save greyscale image in appropriate format
            greyscale_path = os.path.join(self.output_dir,
                                          f"{base_name}_{method}_{palette_size}greyscale{output_extension}")
            if source_is_dds:
                # Save as temp PNG and convert to DDS (temps in output_dir; always cleaned up)
                temp_greyscale_path = os.path.join(self.output_dir,
                                                   f"{base_name}_{method}_{palette_size}greyscale_temp.png")
                try:
                    self.greyscale_data['greyscale_image'].save(temp_greyscale_path)
                    if texconv_path and os.path.exists(texconv_path):
                        self.convert_to_dds(temp_greyscale_path, greyscale_path, texconv_path)
                    else:
                        logger.warning("texconv.exe not found, saving as PNG instead of DDS")
                        greyscale_path = os.path.join(self.output_dir,
                                                      f"{base_name}_{method}_{palette_size}greyscale.png")
                        self.greyscale_data['greyscale_image'].save(greyscale_path)
                finally:
                    try:
                        if os.path.exists(temp_greyscale_path):
                            os.remove(temp_greyscale_path)
                    except Exception as _cleanup_ex:
                        logger.warning(f"Failed to remove temp file {temp_greyscale_path}: {_cleanup_ex}")
            else:
                self.greyscale_data['greyscale_image'].save(greyscale_path)

            # Save Palette in appropriate format
            palette_path = os.path.join(self.output_dir, f"{base_name}_{method}_{palette_size}Palette{output_extension}")
            if source_is_dds:
                # Save as temp PNG and convert to DDS (temps in output_dir; always cleaned up)
                temp_palette_path = os.path.join(self.output_dir,
                                             f"{base_name}_{method}_{palette_size}Palette_temp.png")
                try:
                    self.palette_data['palette_image'].save(temp_palette_path)
                    if texconv_path and os.path.exists(texconv_path):
                        palette_width, palette_height = self.palette_data.get('palette_dimensions', (palette_size, 8))
                        self.convert_to_dds(temp_palette_path, palette_path, texconv_path, is_palette=True, palette_width=palette_width,
                                            palette_height=palette_height)
                    else:
                        logger.warning("texconv.exe not found, saving as PNG instead of DDS")
                        palette_path = os.path.join(self.output_dir, f"{base_name}_{method}_{palette_size}Palette.png")
                        self.palette_data['palette_image'].save(palette_path)
                finally:
                    try:
                        if os.path.exists(temp_palette_path):
                            os.remove(temp_palette_path)
                    except Exception as _cleanup_ex:
                        logger.warning(f"Failed to remove temp file {temp_palette_path}: {_cleanup_ex}")
            else:
                self.palette_data['palette_image'].save(palette_path)

            # Generate DDS if source was DDS
            dds_files = {}
            source_is_dds_check = self.current_image_path.lower().endswith('.dds')
            texconv_path = str(cfg.get(cfg.texconv_file) or "")

            if source_is_dds_check and texconv_path and os.path.exists(texconv_path):
                try:
                    palette_width, palette_height = self.palette_data.get('palette_dimensions', (palette_size, 8))

                    # Convert quantized to DDS
                    quantized_dds_path = os.path.join(self.output_dir,
                                                      f"{base_name}_{method}_{palette_size}quantized.dds")
                    self.convert_to_dds(quantized_path, quantized_dds_path, texconv_path)
                    dds_files['quantized'] = quantized_dds_path

                    # Convert greyscale to DDS
                    greyscale_dds_path = os.path.join(self.output_dir,
                                                      f"{base_name}_{method}_{palette_size}greyscale.dds")
                    self.convert_to_dds(greyscale_path, greyscale_dds_path, texconv_path)
                    dds_files['greyscale'] = greyscale_dds_path

                    # Convert Palette to DDS
                    palette_dds_path = os.path.join(self.output_dir,
                                                f"{base_name}_{method}_{palette_size}Palette.dds")
                    self.convert_to_dds(palette_path, palette_dds_path, texconv_path, is_palette=True, palette_width=palette_width,
                                        palette_height=palette_height)
                    dds_files['palette'] = palette_dds_path

                except Exception as e:
                    logger.error(f"DDS conversion during save failed: {e}")
                    QMessageBox.warning(self, "DDS Conversion Failed", f"Could not generate DDS files: {e}")

            # Save color report
            color_report_path = os.path.join(self.output_dir,
                                             f"{base_name}_{method}_{palette_size}color_report.json")
            self.save_color_report(self.palette_data['color_report_data'], color_report_path)

            # Generate DDS if source was DDS
            dds_files = {}
            source_is_dds = self.current_image_path.lower().endswith('.dds')
            texconv_path = cfg.get(cfg.texconv_file) or ""

            if source_is_dds and texconv_path and os.path.exists(texconv_path):
                try:
                    palette_width, palette_height = self.palette_data.get('palette_dimensions', (palette_size, 8))

                    # Convert quantized to DDS
                    quantized_dds_path = os.path.join(self.output_dir,
                                                      f"{base_name}_{method}_{palette_size}quantized.dds")
                    self.convert_to_dds(quantized_path, quantized_dds_path, texconv_path)
                    dds_files['quantized'] = quantized_dds_path

                    # Convert greyscale to DDS
                    greyscale_dds_path = os.path.join(self.output_dir,
                                                      f"{base_name}_{method}_{palette_size}greyscale.dds")
                    self.convert_to_dds(greyscale_path, greyscale_dds_path, texconv_path)
                    dds_files['greyscale'] = greyscale_dds_path

                    # Convert Palette to DDS
                    palette_dds_path = os.path.join(self.output_dir,
                                                f"{base_name}_{method}_{palette_size}Palette.dds")
                    self.convert_to_dds(palette_path, palette_dds_path, texconv_path, is_palette=True, palette_width=palette_width,
                                        palette_height=palette_height)
                    dds_files['palette'] = palette_dds_path

                except Exception as e:
                    logger.error(f"DDS conversion during save failed: {e}")
                    QMessageBox.warning(self, "DDS Conversion Failed", f"Could not generate DDS files: {e}")

            # Save metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'original_image': self.current_image_path,
                'quantization_method': method,
                'palette_size': palette_size,
                'working_resolution': self.quantized_data.get('working_resolution'),
                'base_original_dimensions': f"{self.quantized_data.get('base_original_dimensions', self.quantized_data.get('dimensions', (0, 0)))[0]}x{self.quantized_data.get('base_original_dimensions', self.quantized_data.get('dimensions', (0, 0)))[1]}",
                'base_processed_dimensions': f"{self.quantized_data.get('dimensions', (0, 0))[0]}x{self.quantized_data.get('dimensions', (0, 0))[1]}",
                'num_blocks': self.palette_data.get('num_blocks', 1),
                'block_sources': self.palette_data.get('block_sources', ['base']),
                'output_files': {
                    'quantized': quantized_path,
                    'greyscale': greyscale_path,
                    'palette': palette_path,
                    'color_report': color_report_path,
                    **dds_files
                }
            }

            metadata_path = os.path.join(self.output_dir,
                                         f"{base_name}_{method}_{palette_size}metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Show success message
            file_list = "\n".join([f"• {os.path.basename(path)}" for path in [
                quantized_path, greyscale_path, palette_path, color_report_path, metadata_path
            ] + list(dds_files.values())])

            QMessageBox.information(self, "Save Complete",
                                    f"All files saved successfully!\n\n{file_list}")

            logger.info("✓ All files saved successfully!")

        except Exception as e:
            logger.error(f"Error saving files: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Save Error", f"Error saving files: {e}")

    def save_color_report(self, color_report_data, file_path):
        """Save color report data as JSON file"""
        try:
            # Convert color_report_data to JSON-serializable format
            serializable_data = []
            for color_data in color_report_data:
                serializable_data.append({
                    'grey_value': int(color_data['grey_value']),
                    'color_rgb': [
                        int(color_data['color_rgb'][0]),
                        int(color_data['color_rgb'][1]),
                        int(color_data['color_rgb'][2])
                    ],
                    'frequency': int(color_data['frequency']),
                    'frequency_percent': float(color_data['frequency_percent'])
                })

            # Create a structured report
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_colors': len(serializable_data),
                    'most_used_colors': sorted(serializable_data, key=lambda x: x['frequency'], reverse=True)[:10],
                    'least_used_colors': sorted(serializable_data, key=lambda x: x['frequency'])[:10]
                },
                'color_mapping': serializable_data
            }

            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)

            logger.debug(f"Color report saved to: {file_path}")
        except Exception as e:
            logger.error(f"Error saving color report: {str(e)}")
            raise

    def convert_to_dds(self, input_path, output_path, texconv_path, is_palette=False, palette_width=256, palette_height=8):
        """Convert image to DDS format using texconv.exe"""
        try:
            if is_palette:
                cmd = [
                    texconv_path,
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
                    texconv_path,
                    '-f', 'BC7_UNORM',
                    '-y',
                    '-m', '1',
                    '-srgb',
                    input_path,
                    '-o', os.path.dirname(output_path)
                ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise Exception(f"texconv failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise Exception("texconv timed out")
        except Exception as e:
            raise Exception(f"DDS conversion error: {str(e)}")
