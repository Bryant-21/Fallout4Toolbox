import os
import re
from collections import Counter, defaultdict
from typing import List, Tuple

import numpy as np
from PIL import Image
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget, QLabel, QFileDialog, QMessageBox
from qfluentwidgets import (
    FluentIcon as FIF,
    PrimaryPushButton,
    PushSettingCard,
)

from help.palette_combiner_help import PaletteCombineHelp
from settings.palette_settings import PaletteSettings
from src.palette.palette_engine import (
    next_power_of_2,
    perceptual_color_sort,
)
from src.utils.dds_utils import save_image, load_image
from src.utils.appconfig import cfg, ConfigItem
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger
from src.utils.cards import TextSettingCard

# Reuse supported extensions from bulk palette if available, otherwise define here
try:
    from src.widgets.bulk_palette_generator import SUPPORTED_EXTS, BulkPaletteWorker
except Exception:
    SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tif', '.tiff', '.dds'}
    BulkPaletteWorker = None

# Add a config item to persist merge group names if not already defined
if not hasattr(cfg, 'ci_merge_groups'):
    # Fallback dynamic attribute (won't be persisted across restarts unless config is updated separately)
    cfg.ci_merge_groups = ConfigItem("palette", "merge_groups", "")


class CombinePalettesWorker(QThread):
    progress = Signal(int, str)
    error = Signal(str)
    completed = Signal(dict)

    def __init__(self, directory: str, groups_csv: str, output_dir: str | None, target_palette_size: int):
        super().__init__()
        self.directory = directory
        self.groups = [self._sanitize_for_filename(g) for g in (groups_csv or '').split(',') if g.strip()]
        self.output_dir = output_dir or directory
        self.target_palette_size = int(target_palette_size or 256)

    def run(self):
        try:
            if not self.groups or len(self.groups) < 2:
                raise Exception("Please provide two or more group names (CSV).")

            self.progress.emit(1, "Scanning for group palettes and greyscales...")
            include_subdirs_raw = cfg.get(cfg.ci_include_subdirs) if hasattr(cfg, 'ci_include_subdirs') else False
            include_subdirs = include_subdirs_raw.value if hasattr(include_subdirs_raw, 'value') else include_subdirs_raw
            include_subdirs = bool(include_subdirs)

            # Find group palettes and greyscales
            group_palettes = {}  # group -> (palette_image_array RGB [H,W,3], width)
            group_greys = defaultdict(list)  # group -> list of dicts {path, base_name, ext}

            # Collect files
            all_files = []
            if include_subdirs:
                for root, _, files in os.walk(self.directory):
                    for name in files:
                        all_files.append(os.path.join(root, name))
            else:
                for name in os.listdir(self.directory):
                    all_files.append(os.path.join(self.directory, name))

            files_lc = [f for f in all_files if os.path.isfile(f)]

            # Helper to check ext
            def has_supported_ext(path):
                return os.path.splitext(path)[1].lower() in SUPPORTED_EXTS

            # Locate palettes per group
            for g in self.groups:
                pal_paths = [p for p in files_lc if has_supported_ext(p) and os.path.splitext(os.path.basename(p))[0].lower() == f"{g.lower()}_palette"]
                if not pal_paths:
                    # Also try pattern *\\{g}_palette.ext (already handled by exact base name check), nothing else
                    pass
                if not pal_paths:
                    raise Exception(f"Palette image for group '{g}' not found (expected '{g}_palette.*').")
                # Prefer DDS if multiple, else first
                pal_path = None
                dds = [p for p in pal_paths if os.path.splitext(p)[1].lower() == '.dds']
                if dds:
                    pal_path = dds[0]
                else:
                    pal_path = pal_paths[0]

                pal_img = load_image(pal_path)
                pal_img = pal_img.convert('RGB')
                pal_arr = np.array(pal_img)
                # The palette image is rows of repeated palette colors; take first row width as palette size
                width = pal_arr.shape[1]
                first_row = pal_arr[0, :width, :]
                group_palettes[g] = (first_row.copy(), width)

                # Find greyscales for this group: pattern '*_{group}_greyscale.ext'
                for p in files_lc:
                    if not has_supported_ext(p):
                        continue
                    base, ext = os.path.splitext(os.path.basename(p))
                    if base.lower().endswith(f"_{g.lower()}_greyscale"):
                        # derive base_name without trailing _{group}_greyscale
                        base_name = base[:-(len(g) + len("_greyscale") + 1)]
                        group_greys[g].append({
                            'path': p,
                            'base_name': base_name,
                            'ext': ext.lower(),
                        })

            # Verify we have greys for each group
            for g in self.groups:
                if not group_greys[g]:
                    logger.warning(f"No greyscale images found for group '{g}'. We'll still merge palettes.")

            self.progress.emit(10, "Building global color histogram from group palettes and greyscales...")

            # Build global color counter and coverage using greyscales if available
            global_counter = Counter()
            color_image_coverage = Counter()  # color -> in how many distinct images it appears
            # Ensure all palette colors are included at least once
            all_palette_colors = []

            # Mapping for later remap: per group index -> RGB color
            group_index_to_color = {}
            for g, (pal_row, width) in group_palettes.items():
                colors = [tuple(int(x) for x in c) for c in pal_row[:width].tolist()]
                group_index_to_color[g] = colors
                all_palette_colors.extend(colors)

            # Count from greyscales
            total_greys = sum(len(v) for v in group_greys.values())
            processed = 0
            for g in self.groups:
                pal_colors = group_index_to_color.get(g, [])
                for item in group_greys[g]:
                    img = load_image(item['path'])
                    imgL = img.convert('L')
                    arr = np.array(imgL)
                    # Validate indices within palette width
                    h, w = arr.shape
                    max_idx = int(np.max(arr)) if arr.size > 0 else 0
                    if max_idx >= len(pal_colors):
                        logger.warning(f"Greyscale indices exceed palette length for {item['path']}. Indices will be clipped.")
                    # Flatten and count
                    values, counts = np.unique(arr.reshape(-1), return_counts=True)
                    used_colors_in_image = set()
                    for v, c in zip(values.tolist(), counts.tolist()):
                        idx = int(v)
                        if idx < 0 or idx >= len(pal_colors):
                            continue
                        col = pal_colors[idx]
                        global_counter[col] += int(c)
                        used_colors_in_image.add(col)
                    for col in used_colors_in_image:
                        color_image_coverage[col] += 1

                    processed += 1
                    pct = 10 + int((processed / max(1, total_greys)) * 30)
                    self.progress.emit(pct, f"Counted colors from {os.path.basename(item['path'])}")

            # Ensure every palette color appears at least once so it can be mapped even if greyscales missing
            for col in all_palette_colors:
                if col not in global_counter:
                    global_counter[col] += 1

            if not global_counter:
                raise Exception("No colors found from inputs. Ensure groups have palettes and/or greyscales.")

            self.progress.emit(45, f"Reducing {len(global_counter)} colors to ≤ {self.target_palette_size}...")

            # Build unified palette using the same algorithm as bulk palette
            if BulkPaletteWorker is None:
                raise Exception("BulkPaletteWorker not available to reuse palette reduction. Import failed.")
            temp_worker = BulkPaletteWorker(self.directory, '', self.output_dir, palette_size=self.target_palette_size)
            palette_colors, color_map, pad_candidates = temp_worker.build_global_palette(global_counter, target_size=self.target_palette_size, color_coverage=color_image_coverage)

            # Ensure exact target size with padding and perceptual sort like BulkPaletteWorker
            colors_np = np.array(palette_colors, dtype=np.uint8)
            sample_img_for_padding = None  # not needed here
            if len(colors_np) < self.target_palette_size:
                try:
                    if pad_candidates:
                        seen = {tuple(map(int, c)) for c in colors_np.tolist()}
                        to_add = []
                        for c in pad_candidates:
                            t = tuple(int(x) for x in c)
                            if t not in seen:
                                to_add.append(t)
                                seen.add(t)
                                if len(colors_np) + len(to_add) >= self.target_palette_size:
                                    break
                        if to_add:
                            colors_np = np.vstack([colors_np, np.array(to_add, dtype=np.uint8)])
                    if len(colors_np) < self.target_palette_size:
                        # Repeat last colors if still short
                        while len(colors_np) < self.target_palette_size:
                            colors_np = np.vstack([colors_np, colors_np[:max(1, self.target_palette_size - len(colors_np))]])
                        colors_np = colors_np[:self.target_palette_size]
                except Exception:
                    while len(colors_np) < self.target_palette_size:
                        colors_np = np.vstack([colors_np, colors_np[:max(1, self.target_palette_size - len(colors_np))]])
                    colors_np = colors_np[:self.target_palette_size]

            sorted_colors = perceptual_color_sort([tuple(c) for c in colors_np])
            unified_palette = np.array(sorted_colors, dtype=np.uint8)

            self.progress.emit(65, "Remapping greyscales to the unified palette and saving outputs...")

            # LUT: representative color -> unified index
            palette_index_lut = {tuple(color): i for i, color in enumerate(unified_palette.tolist())}

            results = []
            # For each greyscale, build index LUT via color_map and group's palette colors
            for g in self.groups:
                pal_colors = group_index_to_color.get(g, [])
                # Precompute mapping from group index to unified palette index
                idx_to_unified = np.zeros((len(pal_colors),), dtype=np.uint16)
                for i, col in enumerate(pal_colors):
                    rep = color_map.get(tuple(col))
                    if rep is None:
                        # Map unused colors to nearest by RGB distance in unified palette
                        pal_int16 = unified_palette.astype(np.int16)
                        c = np.array(col, dtype=np.int16)
                        d2 = np.sum((pal_int16 - c) ** 2, axis=1)
                        idx_to_unified[i] = int(np.argmin(d2))
                    else:
                        idx_to_unified[i] = int(palette_index_lut[tuple(rep)])

                for item in group_greys[g]:
                    img = load_image(item['path'])
                    arrL = np.array(img.convert('L'))
                    # Map indices
                    flat = arrL.reshape(-1)
                    # Clip indices just in case
                    flat = np.clip(flat, 0, max(0, len(idx_to_unified) - 1))
                    mapped = idx_to_unified[flat]
                    grey_indices = mapped.reshape(arrL.shape).astype(np.uint8)

                    # Prepare filenames: base_merged_*
                    base_with_merged = f"{item['base_name']}_merged"
                    src_ext = item['ext']

                    # Save greyscale and quant
                    grey_base = f"{base_with_merged}_greyscale"
                    color_base = f"{base_with_merged}_quant"
                    grey_out = os.path.join(self.output_dir, grey_base, src_ext)
                    color_out = os.path.join(self.output_dir, color_base,src_ext)
                    color_arr = unified_palette[grey_indices]
                    save_image(Image.fromarray(grey_indices, mode='L'), grey_out)
                    save_image(Image.fromarray(color_arr.astype(np.uint8), mode='RGB'), color_out)

                    results.append({
                        'source_greyscale': item['path'],
                        'greyscale_out': grey_out,
                        'quant_out': color_out,
                        'group': g,
                    })

            # Save unified palette
            self.progress.emit(90, "Saving unified palette image...")
            palette_width = self.target_palette_size
            palette_row_height = cfg.get(cfg.ci_palette_row_height) if hasattr(cfg, 'ci_palette_row_height') else 8
            palette_height = next_power_of_2(palette_row_height)
            pal_img_arr = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
            for row in range(palette_height):
                pal_img_arr[row, :palette_width] = unified_palette
            palette_img = Image.fromarray(pal_img_arr, 'RGB')

            # Decide extension: DDS only if all input palettes were DDS
            palette_ext = '.png'
            if all(os.path.splitext(self._find_group_palette_path(g, files_lc))[1].lower() == '.dds' for g in self.groups):
                palette_ext = '.dds'

            palette_out = os.path.join(self.output_dir, "merged_palette", palette_ext)
            save_image(palette_img, palette_out)

            self.progress.emit(100, "Palette group merge complete")
            self.completed.emit({
                'palette_path': palette_out,
                'palette_colors': unified_palette.tolist(),
                'images': results,
            })

        except Exception as e:
            logger.error(f"CombinePalettesWorker error: {e}", exc_info=True)
            self.error.emit(str(e))

    @staticmethod
    def _sanitize_for_filename(name: str) -> str:
        if not name:
            return ""
        cleaned = re.sub(r'[<>:"/\\|?*]+', '', name)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip().strip('.')
        return cleaned

    @staticmethod
    def _find_group_palette_path(group: str, files: List[str]) -> str:
        candidates = []
        for p in files:
            base = os.path.splitext(os.path.basename(p))[0].lower()
            if base == f"{group.lower()}_palette" and os.path.splitext(p)[1].lower() in SUPPORTED_EXTS:
                candidates.append(p)
        if not candidates:
            return ''
        dds = [p for p in candidates if os.path.splitext(p)[1].lower() == '.dds']
        return dds[0] if dds else candidates[0]


class CombinePaletteGroupsWidget(BaseWidget):
    def __init__(self, parent=None, text=None):
        super().__init__(parent=parent, text=text, vertical=True)
        self.main_widget = QWidget()
        self.dir_card = None
        self.output_dir_card = None
        self.groups_card = None
        self.generate_button = None
        self.progress_bar = None
        self.info_label = None

        self.directory = None
        self.output_dir = None

        # Directory selection
        self.dir_card = PushSettingCard(
            self.tr("Texture Directory"),
            FIF.FOLDER,
            self.tr("Directory containing group outputs (palettes, greyscales)"),
            self.tr("No directory selected")
        )
        self.dir_card.clicked.connect(self._choose_directory)
        self.addToFrame(self.dir_card)

        # Include subdirs switch (reuse existing global setting card like BulkPaletteWidget)
        self.subdirs_card = None
        try:
            from qfluentwidgets import SwitchSettingCard
            self.subdirs_card = SwitchSettingCard(
                icon=CustomIcons.SUB.icon(),
                title=self.tr("Include Subdirectories"),
                configItem=cfg.ci_include_subdirs
            )
            self.addToFrame(self.subdirs_card)
        except Exception:
            pass

        # Group CSV card
        self.groups_card = TextSettingCard(
            cfg.ci_merge_groups,
            FIF.TAG,
            self.tr("Group names (CSV)"),
            self.tr("Enter two or more group names to merge, e.g., GroupA, GroupB, GroupC"),
        )
        self.addToFrame(self.groups_card)

        # Output directory
        self.output_dir_card = PushSettingCard(
            self.tr("Output Directory"),
            CustomIcons.FOLDERRIGHT.icon(),
            self.tr("Where merged greyscale, quant, and palette images will be written"),
            self.tr("Will use input directory if not set")
        )
        self.output_dir_card.clicked.connect(self._choose_output_dir)
        self.addToFrame(self.output_dir_card)

        # Progress masked by parent window; no local ProgressBar

        # Info label
        self.info_label = QLabel("")
        self.addToFrame(self.info_label)

        # Run button
        self.generate_button = PrimaryPushButton(icon=FIF.RIGHT_ARROW, text=self.tr("Run"))
        self.generate_button.clicked.connect(self._run)
        self.generate_button.setEnabled(True)
        self.addButtonBarToBottom(self.generate_button)

        self.settings_widget = PaletteSettings(self)
        self.settings_drawer.addWidget(self.settings_widget)

        self.help_widget = PaletteCombineHelp(self)
        self.help_drawer.addWidget(self.help_widget)

    def _choose_directory(self):
        directory = QFileDialog.getExistingDirectory(self, self.tr("Select Directory"), "")
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
            QMessageBox.warning(self, self.tr("Missing Directory"), self.tr("Please choose a directory."))
            return

        groups_csv = cfg.get(cfg.ci_merge_groups)
        if not groups_csv or not str(groups_csv).strip():
            QMessageBox.warning(self, self.tr("Missing Groups"), self.tr("Please enter group names (CSV)."))
            return

        # Determine target palette size from config
        target_size = int(cfg.get(cfg.ci_default_palette_size))

        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'show_progress'):
            try:
                p.show_progress()
            except Exception:
                pass
        self.info_label.setText(self.tr("Starting..."))
        self.worker = CombinePalettesWorker(
            directory=self.directory,
            groups_csv=str(groups_csv),
            output_dir=self.output_dir,
            target_palette_size=target_size,
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
        self.info_label.setText(self.tr(f"Done. Saved merged palette to {pal_path}. Remapped {count} greyscale(s)."))
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'complete_loader'):
            try:
                p.complete_loader()
            except Exception:
                pass

    def _on_error(self, message):
        QMessageBox.critical(self, self.tr("Combine Palettes Error"), message)
        self.info_label.setText(message)
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'complete_loader'):
            try:
                p.complete_loader()
            except Exception:
                pass
