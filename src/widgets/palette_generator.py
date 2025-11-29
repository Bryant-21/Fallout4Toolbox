import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
from PIL import Image, ImageFilter
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
from src.palette.palette_engine import perceptual_color_sort, adjacency_aware_color_sort, analyze_color_distribution, \
    map_rgb_array_to_palette_indices, \
    build_row_from_arrays, compose_palette_image, upscale_and_smooth_lut, \
    adjacency_from_p_mode, adjacency_aware_color_sort_pmode, map_rgb_array_to_palette_indices_coarse
from src.settings.palette_settings import PaletteSettings
from src.utils.appconfig import cfg
from src.utils.dds_utils import save_image, load_image
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger
from src.utils.filesystem_utils import get_app_root
from src.utils.palette_utils import quantize_image, apply_palette_to_greyscale, _get_palette_array, apply_smooth_dither


class SinglePaletteGenerationWorker(QThread):
    """Worker thread for generating a perfect Palette for a single image"""
    progress_updated = Signal(int, str)
    error_occurred = Signal(str)
    step_complete = Signal(str, object)  # step_name, result_data

    def __init__(self, parent, image_path, output_dir,
                 extra_image_paths=None,
                 working_resolution=None,
                 produce_color_report=False,
                 greyscale_texture_paths=None,
                 palette_row_height=4):
        super().__init__()
        self.parent = parent
        self.image_path = image_path
        self.output_dir = output_dir
        # previous_data is deprecated; all intermediates are local to run()
        self.extra_image_paths = extra_image_paths or []
        self.working_resolution = working_resolution  # None for Original, else max side target (e.g., 4096)
        self.produce_color_report = produce_color_report
        self.greyscale_texture_paths = greyscale_texture_paths or []
        self.palette_row_height = palette_row_height
        # sRGB compensation removed from UI/logic; keep attribute for backward-compatibility in save paths
        self.srgb_compensation = False


    def run(self):
        """Run the full Palette generation pipeline in one go, in a single method.
        This fully inlines the previous step-based logic to satisfy the requirement that
        all steps are executed inside one method without delegating to sub-steps.
        """
        try:
            logger.debug("Starting Palette generation (single-pass pipeline, fully inlined)")

            # =====================
            # 1) QUANTIZATION STEP
            # =====================
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

            palette_size = cfg.get(cfg.ci_default_palette_size)
            img = load_image(self.image_path)
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
            # quantize_image now returns a P-mode image; use its palette directly
            quantized = quantize_image(original_img, cfg.get(cfg.ci_default_quant_method))
            # Work with palette indices and palette colors (avoid early RGB convert)
            quantized_indices = np.array(quantized, dtype=np.uint8)
            palette_colors = _get_palette_array(quantized)
            # For algorithms that expect RGB arrays, materialize an RGB view via palette lookup
            quantized_array = palette_colors[quantized_indices]
            logger.debug(f"Quantization complete: {palette_colors} unique colors")

            # Prepare additional data: extra images and greyscale textures pre-processing
            extra_images_data = []
            if self.extra_image_paths:
                self.progress_updated.emit(80, f"Preparing {len(self.extra_image_paths)} additional texture(s)...")

                def _process_extra_image(p):
                    try:
                        ex_img = load_image(p)
                        ex_rgb = ex_img
                        ex_orig_w, ex_orig_h = ex_rgb.size
                        ex_proc = downscale_keep_aspect(ex_rgb, self.working_resolution) if self.working_resolution else ex_rgb

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
                        ex_quantized = quantize_image(ex_proc2, cfg.get(cfg.ci_default_quant_method))
                        ex_idx = np.array(ex_quantized, dtype=np.uint8)
                        ex_pal_flat = np.array(ex_quantized.getpalette(), dtype=np.uint8)
                        ex_pal = ex_pal_flat.reshape(-1, 3) if ex_pal_flat.size > 0 else np.zeros((0, 3), dtype=np.uint8)
                        ex_arr = ex_pal[ex_idx]
                        return {
                            'path': p,
                            'quantized_image': None,
                            'quantized_array': ex_arr,
                            'original_dimensions': (ex_orig_w, ex_orig_h),
                            'processed_dimensions': (ex_w2, ex_h2),
                            'dimensions': (ex_w2, ex_h2)
                        }
                    except Exception as ex:
                        logger.warning(f"Failed to process extra image '{p}': {ex}")
                        return None

                max_workers = max(1, int(cfg.get(cfg.threads_cfg)))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_process_extra_image, p): idx for idx, p in enumerate(self.extra_image_paths)}
                    ordered_results = [None] * len(self.extra_image_paths)
                    for fut in as_completed(futures):
                        idx = futures[fut]
                        res = fut.result()
                        ordered_results[idx] = res
                    extra_images_data.extend([r for r in ordered_results if r is not None])

            # Quantize greyscale-conversion textures (to later map using base mapping)
            greyscale_textures_data = []
            if self.greyscale_texture_paths:
                self.progress_updated.emit(88, f"Preparing {len(self.greyscale_texture_paths)} greyscale conversion texture(s)...")

                def _process_greyscale_texture(p):
                    try:
                        gs_img = load_image(p)
                        gs_rgb = gs_img
                        gs_orig_w, gs_orig_h = gs_rgb.size
                        gs_proc = downscale_keep_aspect(gs_rgb, self.working_resolution) if self.working_resolution else gs_rgb

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
                        gs_quantized = quantize_image(gs_proc2, cfg.get(cfg.ci_default_quant_method))
                        gs_idx = np.array(gs_quantized, dtype=np.uint8)
                        gs_pal_flat = np.array(gs_quantized.getpalette(), dtype=np.uint8)
                        gs_pal = gs_pal_flat.reshape(-1, 3) if gs_pal_flat.size > 0 else np.zeros((0, 3), dtype=np.uint8)
                        gs_quant_arr = gs_pal[gs_idx]
                        return {
                            'path': p,
                            'processed_color_image': gs_proc2.convert('RGB'),
                            'processed_color_array': np.array(gs_proc2.convert('RGB')),
                            'quantized_array': gs_quant_arr,
                            'original_dimensions': (gs_orig_w, gs_orig_h),
                            'processed_dimensions': (gs_w2, gs_h2)
                        }
                    except Exception as ex:
                        logger.warning(f"Failed to process greyscale texture '{p}': {ex}")
                        return None

                max_workers = max(1, int(cfg.get(cfg.threads_cfg)))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_process_greyscale_texture, p): idx for idx, p in enumerate(self.greyscale_texture_paths)}
                    ordered_results = [None] * len(self.greyscale_texture_paths)
                    for fut in as_completed(futures):
                        idx = futures[fut]
                        res = fut.result()
                        ordered_results[idx] = res
                    greyscale_textures_data.extend([r for r in ordered_results if r is not None])

            # =====================
            # 2) GREYSCALE STEP
            # =====================
            self.progress_updated.emit(10, "Preparing greyscale conversion...")


            self.progress_updated.emit(20, "Sorting colors and assigning greyscale values...")

            color_tuples = [tuple(color) for color in palette_colors]
            try:
                sorted_colors = perceptual_color_sort(color_tuples)
            except Exception as _:
                sorted_colors = adjacency_aware_color_sort_pmode(quantized)

            color_to_grey = {}
            grey_to_color = {}
            for grey_value, color in enumerate(sorted_colors):
                color_to_grey[color] = grey_value
                grey_to_color[grey_value] = np.array(color, dtype=np.uint8)

            logger.debug("Color to greyscale mapping created")
            self.progress_updated.emit(60, "Creating greyscale image...")

            lut_exact = {tuple(map(int, c)): int(i) for i, c in enumerate(sorted_colors)}
            pal_int16 = np.array(sorted_colors, dtype=np.uint8).astype(np.int16)
            greyscale_array = map_rgb_array_to_palette_indices(quantized_array, lut_exact, pal_int16).astype(np.uint8)
            # Mask out fully transparent pixels from greyscale mapping (use index 0 under mask)
            try:
                if original_array.shape[-1] == 4:
                    alpha_mask = original_array[:, :, 3]
                    mask_zero = (alpha_mask == 0)
                    if mask_zero.any():
                        greyscale_array[mask_zero] = 0
            except Exception:
                pass

            try:
                do_post = bool(cfg.get(cfg.ci_greyscale_post_enable)) if hasattr(cfg, 'ci_greyscale_post_enable') else False
            except Exception:
                do_post = False
            if do_post:
                greyscale_array = apply_smooth_dither(greyscale_array, palette_size)
                # Re-apply transparency mask to ensure excluded areas remain zero
                try:
                    if original_array.shape[-1] == 4:
                        alpha_mask = original_array[:, :, 3]
                        greyscale_array[alpha_mask == 0] = 0
                except Exception:
                    pass

            if palette_size > 1:
                scale = 255.0 / float(palette_size - 1)
            else:
                scale = 0.0
            disp = (greyscale_array.astype(np.float32) * scale).astype(np.uint8)
            # Preserve original alpha in the greyscale output if available
            try:
                if original_array.shape[-1] == 4:
                    alpha_mask = original_array[:, :, 3].astype(np.uint8)
                    la = np.stack([disp, alpha_mask], axis=2)
                    greyscale_rgb = Image.fromarray(la, 'LA')
                else:
                    greyscale_rgb = Image.fromarray(disp, 'L')
            except Exception:
                greyscale_rgb = Image.fromarray(disp, 'L')

            greyscale_textures_results = []
            gs_sources = greyscale_textures_data
            if gs_sources:
                self.progress_updated.emit(80, f"Mapping {len(gs_sources)} greyscale conversion texture(s)...")
                for entry in gs_sources:
                    try:
                        proc_img = entry.get('processed_color_image')
                        proc_arr = None
                        if proc_img is not None:
                            proc_arr = np.array(proc_img)
                        else:
                            proc_arr = entry.get('processed_color_array')
                        if proc_arr is None:
                            proc_arr = entry.get('quantized_array')
                        if proc_arr is None:
                            continue
                        gs_indices = map_rgb_array_to_palette_indices(proc_arr, lut_exact, pal_int16).astype(np.uint8)
                        try:
                            do_post_t = bool(cfg.get(cfg.ci_greyscale_post_enable)) if hasattr(cfg, 'ci_greyscale_post_enable') else False
                            apply_textures = bool(cfg.get(cfg.ci_greyscale_post_apply_to_textures)) if hasattr(cfg, 'ci_greyscale_post_apply_to_textures') else True
                        except Exception:
                            do_post_t = False
                            apply_textures = True
                        if do_post_t and apply_textures:
                            gs_indices = apply_smooth_dither(gs_indices, palette_size)
                        disp_t = (gs_indices * (255 / (palette_size - 1))).astype(np.uint8) if palette_size > 1 else np.zeros_like(gs_indices, dtype=np.uint8)
                        # Apply the texture's own alpha as a mask if present
                        try:
                            src_arr = np.array(proc_img)
                            if src_arr.shape[-1] == 4:
                                alpha_t = src_arr[:, :, 3].astype(np.uint8)
                                # Ensure masked areas are zero
                                gs_indices[alpha_t == 0] = 0
                                disp_t = (gs_indices * (255 / (palette_size - 1))).astype(np.uint8) if palette_size > 1 else np.zeros_like(gs_indices, dtype=np.uint8)
                                la_t = np.stack([disp_t, alpha_t], axis=2)
                                gs_img = Image.fromarray(la_t, 'LA')
                            else:
                                gs_img = Image.fromarray(disp_t, 'L')
                        except Exception:
                            gs_img = Image.fromarray(disp_t, 'L')
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

            # =====================
            # 3) PALETTE STEP
            # =====================
            self.progress_updated.emit(10, "Creating Palette...")

            # Base palette row
            base_palette_array = np.zeros((palette_size, 3), dtype=np.uint8)
            for grey_value in range(palette_size):
                base_palette_array[grey_value] = grey_to_color[grey_value]

            # Extra rows from additional textures
            extra_palette_arrays = []
            extra_sources = extra_images_data
            if extra_sources:
                for extra in extra_sources:
                    extra_arr = extra.get('quantized_array')
                    if extra_arr is None:
                        continue
                    label = os.path.basename(extra.get('path', 'extra'))
                    palette_for_extra = build_row_from_arrays(
                        greyscale_indices=greyscale_array,
                        rgb_array=extra_arr,
                        base_row=base_palette_array,
                        palette_size=palette_size,
                        log_top_k=3,
                        context_label=label
                    )
                    extra_palette_arrays.append(palette_for_extra)

            # Optional upscale to 256
            palette_size_out = palette_size
            try:
                do_upscale = bool(cfg.get(cfg.ci_palette_upscale_to_256)) if hasattr(cfg, 'ci_palette_upscale_to_256') else False
                sigma_cfg = float(cfg.get(cfg.ci_palette_upscale_sigma)) if hasattr(cfg, 'ci_palette_upscale_sigma') else 10.0
            except Exception:
                do_upscale = False
                sigma_cfg = 10.0
            upscale_sigma = float(sigma_cfg) / 10.0

            if do_upscale and palette_size < 256:
                base_palette_array = upscale_and_smooth_lut(base_palette_array, target_size=256, sigma=upscale_sigma)
                if extra_palette_arrays:
                    extra_palette_arrays = [upscale_and_smooth_lut(r, target_size=256, sigma=upscale_sigma) for r in extra_palette_arrays]
                # Remap greyscale indices to 0..255
                if palette_size > 1:
                    scale_u = 255.0 / float(palette_size - 1)
                    greyscale_array = np.clip(np.round(greyscale_array.astype(np.float32) * scale_u), 0, 255).astype(np.uint8)
                else:
                    greyscale_array = np.zeros_like(greyscale_array, dtype=np.uint8)
                palette_size_out = 256

            # Compose palette image
            all_blocks = [base_palette_array] + extra_palette_arrays if extra_palette_arrays else [base_palette_array]
            palette_image = compose_palette_image(
                rows=all_blocks,
                row_height=self.palette_row_height,
                palette_size=palette_size_out,
                pad_mode='gradient'
            )

            self.progress_updated.emit(50, "Applying Palette to greyscale for preview...")
            preview_image = apply_palette_to_greyscale(palette_image, greyscale_rgb)

            if self.produce_color_report:
                self.progress_updated.emit(80, "Creating color report...")
                color_distribution = analyze_color_distribution(quantized_array)
                color_report_data = []
                for grey_value in range(palette_size_out):
                    color = tuple(base_palette_array[grey_value])
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

            # =====================
            # 4) SAVE RESULTS
            # =====================
            # Inline save_all_results logic using local variables
            try:
                logger.debug("Saving all results")
                self.progress_updated.emit(90, "Saving files...")

                # Simplified base name (no method/size suffixes)
                base_name = os.path.splitext(os.path.basename(self.image_path))[0]

                # Determine output format based on input format
                source_is_dds = self.image_path.lower().endswith('.dds')
                output_extension = ".dds" if source_is_dds else ".png"

                output_files = {}

                # Save core outputs

                quantized_path = os.path.join(self.output_dir, f"{base_name}_quantized{output_extension}")
                save_image(quantized.convert("RGB"), quantized_path)

                greyscale_path = os.path.join(self.output_dir, f"{base_name}_greyscale{output_extension}")
                save_image(greyscale_rgb, greyscale_path)
                output_files['greyscale'] = greyscale_path
                logger.debug(f"Saved greyscale image: {greyscale_path}")

                palette_path = os.path.join(self.output_dir, f"{base_name}_palette{output_extension}")
                save_image(palette_image, palette_path)
                output_files['palette'] = palette_path
                logger.debug(f"Saved Palette: {palette_path}")

                # Save greyscale-conversion textures
                saved_gs_textures = []
                saved_color_textures = []
                gs_tex_list = greyscale_textures_results
                for idx, entry in enumerate(gs_tex_list, start=1):
                    try:
                        color_img = entry.get('processed_color_image')
                        grey_img = entry.get('greyscale_image')
                        if color_img is None or grey_img is None:
                            continue

                        color_out = os.path.join(self.output_dir, f"{base_name}_texture{idx}{output_extension}")
                        grey_out = os.path.join(self.output_dir, f"{base_name}_greyscaletexture_{idx}{output_extension}")

                        save_image(color_img, color_out)
                        save_image(grey_img, grey_out)

                        saved_color_textures.append(color_out)
                        saved_gs_textures.append(grey_out)
                    except Exception as ex:
                        logger.warning(f"Failed saving greyscale conversion texture {idx}: {ex}")
                if saved_color_textures:
                    output_files['textures'] = saved_color_textures
                if saved_gs_textures:
                    output_files['greyscale_textures'] = saved_gs_textures

                # Optional: Save color report
                color_report_path = None
                if self.produce_color_report and color_report_data:
                    color_report_path = os.path.join(self.output_dir, f"{base_name}_color_report.json")
                    self.save_color_report(color_report_data, color_report_path)
                    output_files['color_report'] = color_report_path
                    logger.debug(f"Saved color report: {color_report_path}")

                pixmap = self.parent.pil_to_pixmap(quantized.convert("RGB"))
                self.parent.update_preview(pixmap, self.parent.quantized_preview_label)

                pixmap = self.parent.pil_to_pixmap(greyscale_rgb)
                self.parent.update_preview(pixmap, self.parent.greyscale_preview_label)

                pixmap = self.parent.pil_to_pixmap(preview_image)
                self.parent.update_preview(pixmap, self.parent.preview_label)

                default_grey_path = os.path.normpath(os.path.join(get_app_root(), 'resource', 'grayscale_4k_cutout.png'))
                pixmap = self.parent.pil_to_pixmap(apply_palette_to_greyscale(palette_image, load_image(default_grey_path, f='L')))
                self.parent.update_preview(pixmap, self.parent.debug_preview_label)

                self.progress_updated.emit(100, "Complete!")
                logger.debug("All results saved successfully")
            except Exception as e:
                logger.error(f"Error saving results: {str(e)}", exc_info=True)
                raise

        except Exception as e:
            logger.error(f"Error in Palette generation: {str(e)}", exc_info=True)
            self.error_occurred.emit(str(e))


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
        # Per-step cached data removed: pipeline runs and completes in one worker call
        self.original_preview_label = None
        self.greyscale_preview_label = None
        self.debug_preview_label = None
        self.quantized_preview_label = None
        self.preview_label = None
        # previous_data removed; no step-by-step orchestration
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
        self.generate_all_button.clicked.connect(self.generate_step)
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

        # Updated previews: Quantized, Greyscale, Preview on Greyscale, Preview on Debug Texture
        preview_splitter.addWidget(make_preview_group("Quantized", "quantized_preview_label"))
        preview_splitter.addWidget(make_preview_group("Greyscale (Color → Grey Mapping)", "greyscale_preview_label"))
        preview_splitter.addWidget(make_preview_group("Preview (Palette Applied to Greyscale)", "preview_label"))
        preview_splitter.addWidget(make_preview_group("Preview (Palette Applied to Debug Texture)", "debug_preview_label"))

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

            # No longer showing Original Image preview in the UI

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
            img = load_image(file_path)
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

    def generate_step(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "Please select an image first.")
            return

        if not self.output_dir:
            self.output_dir = os.path.dirname(self.current_image_path)
            try:
                self.output_dir_card.setContent(self.output_dir)
            except Exception:
                pass


        # Get working resolution from ConfigItem
        wr_val = cfg.get(cfg.ci_default_working_res).value
        if isinstance(wr_val, str) and str(wr_val).lower().startswith('original'):
            working_resolution = None
        else:
            try:
                working_resolution = int(wr_val)
            except Exception:
                working_resolution = None

        # Disable buttons during processing and use parent mask progress
        self.set_buttons_enabled(False)
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'show_progress'):
            try:
                p.show_progress()
            except Exception:
                pass

        self.worker = SinglePaletteGenerationWorker(
            self,
            self.current_image_path,
            self.output_dir,
            extra_image_paths=self.extra_image_paths,
            working_resolution=working_resolution,
            produce_color_report=bool(cfg.get(cfg.ci_produce_color_report)),
            greyscale_texture_paths=self.greyscale_texture_paths,
            palette_row_height=int(cfg.get(cfg.ci_palette_row_height) or 4)
        )

        self.worker.progress_updated.connect(self.update_progress)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()

    def update_progress(self, value, message):
        p = getattr(self, 'parent', None)
        if value == 100:
            self.set_buttons_enabled(True)
            if p and hasattr(p, 'complete_loader'):
                try:
                    p.complete_loader()
                except Exception:
                    pass
        else:
            if p and hasattr(p, 'update_progress'):
                try:
                    p.update_progress(int(value))
                except Exception:
                    pass
        logger.info(f"Progress: {value}% - {message}")

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
