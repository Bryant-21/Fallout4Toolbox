import os
import shutil

from PySide6.QtWidgets import QFileDialog
from qfluentwidgets import PrimaryPushButton, PushSettingCard

from src.utils.appconfig import cfg
from src.utils.cards import TextSettingCard
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger


class FolderRenamerWidget(BaseWidget):
    """
    Folder Renamer tool (logic inlined).

    Allows selecting a source folder to duplicate, entering the new folder name,
    and providing CSV replacement strings that will be used both for folder/file
    names and file contents (text files only). All parameters persist via appconfig.
    """

    def __init__(self, parent=None, text=None):
        super().__init__(parent=parent, text=text, vertical=True)

        self.source_card = PushSettingCard(
            self.tr("Source Folder"),
            CustomIcons.FOLDERRIGHT.icon(stroke=True),
            self.tr("Select the folder to copy and rename"),
            cfg.get(cfg.fr_source_folder) or "",
        )

        self.new_name_card = TextSettingCard(
            cfg.fr_new_folder_name,
            CustomIcons.FILE_EDIT.icon(stroke=True),
            self.tr("New Folder Name"),
            cfg.get(cfg.fr_new_folder_name) or "",
        )

        self.replacements_card = TextSettingCard(
            cfg.fr_replacements,
            CustomIcons.REPLACE.icon(),
            self.tr("Strings to replace (CSV). Use 'old:new' for custom replacements"),
            cfg.get(cfg.fr_replacements) or "",
        )

        self.run_btn = PrimaryPushButton(text=self.tr("Copy && Rename Folder"))

        self.addToFrame(self.source_card)
        self.addToFrame(self.new_name_card)
        self.addToFrame(self.replacements_card)
        self.boxLayout.addStretch(1)
        self.addButtonBarToBottom(self.run_btn)

        self.source_card.clicked.connect(self._on_source_clicked)
        self.run_btn.clicked.connect(self._run)

    def _on_source_clicked(self):
        start_dir = cfg.get(cfg.fr_source_folder) or ''
        path = QFileDialog.getExistingDirectory(self, self.tr('Select Source Folder'), start_dir)
        if path:
            cfg.set(cfg.fr_source_folder, path)
            self.source_card.setContent(path)

    # --- Inlined logic from former folderrenamer.py ---
    def _replace_strings_in_file(self, filepath: str, replacements: dict) -> bool:
        """Replace all occurrences of strings in replacements dict in a text file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            new_content = content
            for old_string, new_string in replacements.items():
                if old_string in new_content:
                    new_content = new_content.replace(old_string, new_string)
            if new_content != content:
                with open(filepath, 'w', encoding='utf-8') as file:
                    file.write(new_content)
                return True
            return False
        except (UnicodeDecodeError, IOError):
            # Skip binary files or unreadable files
            return False

    def _copy_and_rename_folder(self, source_folder: str, new_folder_name: str, strings_to_replace: dict):
        """
        Copy a folder and rename all matching subdirectories and text file contents.
        """
        source_parent = os.path.dirname(source_folder)
        new_folder_path = os.path.join(source_parent, new_folder_name)

        logger.info(f"Copying '{source_folder}' to '{new_folder_path}'...")
        shutil.copytree(source_folder, new_folder_path)

        text_extensions = ('.txt', '.py', '.js', '.html', '.css', '.json',
                           '.xml', '.md', '.yml', '.yaml', '.ini', '.cfg',
                           '.conf', '.java', '.cpp', '.c', '.h', '.hpp',
                           '.php', '.rb', '.go', '.rs', '.ts', '.jsx', '.tsx',
                           '.vue', '.svelte', '.sql', '.sh', '.bat', '.ps1',
                           '.csv', '.toml', '.env', '.properties', '.gradle',
                           '.kt', '.swift', '.dart')

        for root, dirs, files in os.walk(new_folder_path, topdown=False):
            # Rename directories
            for dir_name in list(dirs):
                original_dir_name = dir_name
                new_dir_name = dir_name
                for old_string, new_string in strings_to_replace.items():
                    if old_string in new_dir_name:
                        new_dir_name = new_dir_name.replace(old_string, new_string)
                if new_dir_name != original_dir_name:
                    old_path = os.path.join(root, original_dir_name)
                    new_path = os.path.join(root, new_dir_name)
                    try:
                        os.rename(old_path, new_path)
                        logger.debug(f"Renamed directory: {old_path} -> {new_path}")
                        if original_dir_name in dirs:
                            dirs.remove(original_dir_name)
                            dirs.append(new_dir_name)
                    except OSError as e:
                        logger.exception(f"Error renaming directory {old_path}: {e}")

            # Process files
            for file_name in files:
                original_file_name = file_name
                new_file_name = file_name
                file_path = os.path.join(root, file_name)

                for old_string, new_string in strings_to_replace.items():
                    if old_string in new_file_name:
                        new_file_name = new_file_name.replace(old_string, new_string)

                if new_file_name != original_file_name:
                    new_file_path = os.path.join(root, new_file_name)
                    try:
                        os.rename(file_path, new_file_path)
                        logger.debug(f"Renamed file: {file_path} -> {new_file_path}")
                        file_path = new_file_path
                    except OSError as e:
                        logger.exception(f"Error renaming file {file_path}: {e}")

                if file_path.lower().endswith(text_extensions):
                    if self._replace_strings_in_file(file_path, strings_to_replace):
                        logger.debug(f"Updated content in: {file_path}")

    def _parse_csv_replacements(self, csv_string: str, new_folder_name: str) -> dict:
        """
        Parse CSV string into replacement dictionary.
        Format: "old1,old2" or "old1:new1,old2:new2"
        """
        replacements: dict[str, str] = {}
        if not csv_string:
            return replacements
        if ':' in csv_string:
            items = csv_string.split(',')
            for item in items:
                if ':' in item:
                    old, new = item.split(':', 1)
                    replacements[old.strip()] = new.strip()
                else:
                    replacements[item.strip()] = new_folder_name
        else:
            items = csv_string.split(',')
            for item in items:
                replacements[item.strip()] = new_folder_name
        return replacements

    def _run(self):
        source_folder = (cfg.get(cfg.fr_source_folder) or '').strip()
        new_folder_name = (cfg.get(cfg.fr_new_folder_name) or '').strip()
        replacements_csv = (cfg.get(cfg.fr_replacements) or '').strip()

        if not source_folder:
            logger.debug('Please choose a source folder.')
            return
        if not os.path.isdir(source_folder):
            logger.debug('Source folder does not exist.')
            return
        if not new_folder_name:
            logger.debug('Please enter the new folder name.')
            return
        if not replacements_csv:
            logger.debug('Please enter at least one string to replace.')
            return

        # Build destination path and guard against overwriting existing data
        dest_parent = os.path.dirname(source_folder)
        dest_path = os.path.join(dest_parent, new_folder_name)
        if os.path.exists(dest_path):
            logger.debug(f"Destination already exists: {dest_path}. Remove it or choose a different name.")
            return

        try:
            replacements = self._parse_csv_replacements(replacements_csv, new_folder_name)
            self._copy_and_rename_folder(source_folder, new_folder_name, replacements)
            logger.info(f"Folder copied to {dest_path}")
        except Exception as e:
            # Clean up partially copied folder if something failed mid-way
            if os.path.exists(dest_path) and os.path.isdir(dest_path) and not os.listdir(dest_path):
                try:
                    shutil.rmtree(dest_path)
                except Exception:
                    pass
            logger.exception(f"Folder copy/rename failed: {e}")