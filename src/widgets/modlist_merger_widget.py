import os
from pathlib import Path
from typing import List, Dict

from PySide6.QtWidgets import QFileDialog
from qfluentwidgets import PrimaryPushButton, PushSettingCard

from src.utils.appconfig import cfg
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger


class MO2Merger:
    """
    Inlined from legacy modlistmerge.py. Merges modlist.txt, loadorder.txt, and archives.txt
    files from multiple MO2 profiles.
    """

    def __init__(self):
        pass

    def normalize_name(self, line: str) -> str:
        """Remove +/- prefix and strip whitespace for comparison."""
        line = line.strip()
        if line and line[0] in ['+', '-']:
            return line[1:].strip()
        return line

    def get_state(self, line: str) -> str:
        """Get the +/- state from a line."""
        line = line.strip()
        if line and line[0] in ['+', '-']:
            return line[0]
        return '-'

    def is_separator(self, line: str) -> bool:
        """Check if a line is a separator."""
        normalized = self.normalize_name(line).lower()
        return normalized.endswith('_separator')

    def parse_modlist_structure(self, filepath: str) -> Dict[str, List[str]]:
        """
        Parse modlist into a dictionary of separator -> list of mods.
        Returns a dict where keys are separator names (normalized, lowercase)
        and values are lists of mod lines (with +/- prefix).
        Also includes order of separators.
        """
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
            return {'_order': [], '_orphans': []}

        structure = {'_order': [], '_orphans': []}
        current_separator = None

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if self.is_separator(line):
                    sep_name = self.normalize_name(line).lower()
                    current_separator = sep_name
                    if sep_name not in structure:
                        structure[sep_name] = {'state': self.get_state(line), 'mods': []}
                        structure['_order'].append(sep_name)
                else:
                    if current_separator:
                        structure[current_separator]['mods'].append(line)
                    else:
                        structure['_orphans'].append(line)

        return structure

    def merge_modlists(self, master_file: str, secondary_file: str, output_file: str):
        """Merge two modlist.txt files with master taking precedence."""
        print(f"\nMerging modlists:")
        print(f"  Master: {master_file}")
        print(f"  Secondary: {secondary_file}")
        print(f"  Output: {output_file}")

        master_structure = self.parse_modlist_structure(master_file)
        secondary_structure = self.parse_modlist_structure(secondary_file)

        existing_mods = set()
        for sep_name in master_structure.get('_order', []):
            for mod in master_structure[sep_name]['mods']:
                existing_mods.add(self.normalize_name(mod).lower())
        for mod in master_structure.get('_orphans', []):
            existing_mods.add(self.normalize_name(mod).lower())

        for sep_name in secondary_structure.get('_order', []):
            sep_data = secondary_structure[sep_name]
            new_mods = []
            for mod in sep_data['mods']:
                mod_normalized = self.normalize_name(mod).lower()
                if mod_normalized not in existing_mods:
                    new_mods.append(mod)
                    existing_mods.add(mod_normalized)
            if new_mods:
                if sep_name in master_structure:
                    master_structure[sep_name]['mods'].extend(new_mods)
                    print(f"  Added {len(new_mods)} mod(s) to existing separator '{sep_name}'")
                else:
                    master_structure[sep_name] = {'state': sep_data['state'], 'mods': new_mods}
                    master_structure['_order'].append(sep_name)
                    print(f"  Added new separator '{sep_name}' with {len(new_mods)} mod(s)")

        orphan_count = 0
        for mod in secondary_structure.get('_orphans', []):
            mod_normalized = self.normalize_name(mod).lower()
            if mod_normalized not in existing_mods:
                master_structure['_orphans'].append(mod)
                existing_mods.add(mod_normalized)
                orphan_count += 1
        if orphan_count > 0:
            print(f"  Added {orphan_count} orphan mod(s) at the beginning")

        # Ensure the 'Base' separator is always the very last separator
        # Combine Base mods from both lists is already handled by the general merge logic above,
        # but we must enforce its position as the final separator.
        base_key = 'base_separator'
        order_list = master_structure.get('_order', [])
        if base_key in order_list:
            # Move Base separator to the end of the separators order
            master_structure['_order'] = [k for k in order_list if k != base_key] + [base_key]
            print("  Ensured 'Base' separator is last")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# This file was automatically generated by Mod Organizer.\n")
            for mod in master_structure.get('_orphans', []):
                f.write(f"{mod}\n")
            for sep_name in master_structure.get('_order', []):
                sep_data = master_structure[sep_name]
                f.write(f"{sep_data['state']}{sep_name}\n")
                for mod in sep_data['mods']:
                    f.write(f"{mod}\n")
        print(f"✓ Merged modlist saved to {output_file}")

    def merge_simple_list(self, master_file: str, secondary_file: str, output_file: str, file_type: str):
        """Merge loadorder.txt or archives.txt (simple line-by-line lists)."""
        print(f"\nMerging {file_type}:")
        print(f"  Master: {master_file}")
        print(f"  Secondary: {secondary_file}")
        print(f"  Output: {output_file}")

        master_items = set()
        result = []

        if os.path.exists(master_file):
            with open(master_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        master_items.add(line.lower())
                        result.append(line)
        else:
            print(f"  Warning: Master file not found")

        added = 0
        if os.path.exists(secondary_file):
            with open(secondary_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if line.lower() not in master_items:
                            result.append(line)
                            master_items.add(line.lower())
                            added += 1
        else:
            print(f"  Warning: Secondary file not found")

        with open(output_file, 'w', encoding='utf-8') as f:
            if file_type == "loadorder":
                f.write("# This file was automatically generated by Mod Organizer.\n")
            for item in result:
                f.write(f"{item}\n")
        print(f"✓ Merged {file_type} saved ({added} new items added)")

    def merge_profiles(self, master_dir: str, secondary_dir: str, output_dir: str):
        """
        Merge all MO2 files from two profile directories.
        """
        master_path = Path(master_dir)
        secondary_path = Path(secondary_dir)
        output_path = Path(output_dir)

        output_path.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("MO2 Profile Merger")
        print("=" * 60)

        self.merge_modlists(
            str(master_path / "modlist.txt"),
            str(secondary_path / "modlist.txt"),
            str(output_path / "modlist.txt")
        )

        self.merge_simple_list(
            str(master_path / "loadorder.txt"),
            str(secondary_path / "loadorder.txt"),
            str(output_path / "loadorder.txt"),
            "loadorder"
        )

        self.merge_simple_list(
            str(master_path / "archives.txt"),
            str(secondary_path / "archives.txt"),
            str(output_path / "archives.txt"),
            "archives"
        )

        print("\n" + "=" * 60)
        print("✓ All files merged successfully!")
        print("=" * 60)


class ModlistMergerWidget(BaseWidget):
    """
    GUI wrapper for modlistmerge.py to merge two MO2 profiles
    (modlist.txt, loadorder.txt, archives.txt) into an output profile directory.
    Settings persist via appconfig.
    """

    def __init__(self, parent=None, text=None):
        super().__init__(parent=parent, text=text, vertical=True)

        self.master_card = PushSettingCard(
            self.tr("Master Profile Folder"),
            CustomIcons.FOLDERS_LINE.icon(stroke=True),
            self.tr("Profile with priority (modlist takes precedence)"),
            cfg.get(cfg.mm_master_profile) or "",
        )
        self.secondary_card = PushSettingCard(
            self.tr("Secondary Profile Folder"),
            CustomIcons.FOLDER_IMAGE.icon(stroke=True),
            self.tr("Profile to merge in (only missing mods are added)"),
            cfg.get(cfg.mm_secondary_profile) or "",
        )
        self.output_card = PushSettingCard(
            self.tr("Output Folder"),
            CustomIcons.FILE_ADD.icon(stroke=True),
            self.tr("Where to write merged modlist/loadorder/archives"),
            cfg.get(cfg.mm_output_folder) or "",
        )

        self.run_btn = PrimaryPushButton(text=self.tr("Merge Profiles"))

        self.addToFrame(self.master_card)
        self.addToFrame(self.secondary_card)
        self.addToFrame(self.output_card)
        self.boxLayout.addStretch(1)
        self.addButtonBarToBottom(self.run_btn)

        self.master_card.clicked.connect(lambda: self._choose_folder(cfg.mm_master_profile, self.master_card, self.tr('Select Master Profile Folder')))
        self.secondary_card.clicked.connect(lambda: self._choose_folder(cfg.mm_secondary_profile, self.secondary_card, self.tr('Select Secondary Profile Folder')))
        self.output_card.clicked.connect(lambda: self._choose_folder(cfg.mm_output_folder, self.output_card, self.tr('Select Output Folder'), select_existing=True))
        self.run_btn.clicked.connect(self._run)

    def _choose_folder(self, config_item, card, title, select_existing=True):
        start_dir = cfg.get(config_item) or ''
        path = QFileDialog.getExistingDirectory(self, title, start_dir)
        if path:
            cfg.set(config_item, path)
            card.setContent(path)

    def _run(self):
        master = (cfg.get(cfg.mm_master_profile) or '').strip()
        secondary = (cfg.get(cfg.mm_secondary_profile) or '').strip()
        output = (cfg.get(cfg.mm_output_folder) or '').strip()

        if not master or not os.path.isdir(master):
            logger.debug('Please select a valid master profile folder.')
            return
        if not secondary or not os.path.isdir(secondary):
            logger.debug('Please select a valid secondary profile folder.')
            return
        if not output:
            logger.debug('Please select an output folder.')
            return

        # Ensure output exists
        os.makedirs(output, exist_ok=True)

        try:
            merger = MO2Merger()
            merger.merge_profiles(master, secondary, output)
            logger.info(f"Merged profiles written to: {output}")
        except Exception as e:
            logger.exception(f"Modlist merge failed: {e}")