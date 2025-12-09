import os

from PySide6 import QtWidgets
from PySide6.QtWidgets import QWidget, QFileDialog
from qfluentwidgets import FluentIcon as FIF, InfoBar
from qfluentwidgets import PrimaryPushButton, PushSettingCard, ConfigItem, FolderValidator

from src.help.Archlist_help import ArchlistHelp
from src.settings.archlist_settings import ArchlistSettings
from src.utils.appconfig import FileValidator
from src.utils.helpers import BaseWidget


def create_archlist(directory_path, output_file):
    """
    Recursively scan a directory and create an archlist file with all files found.
    The archlist file follows the format:
    [
        "Data\\path\\to\\file1.ext",
        "Data\\path\\to\\file2.ext",
        ...
    ]

    Args:
        directory_path: Path to the directory to scan
        output_file: Path to the output archlist file

    Returns:
        bool: True if successful, False otherwise
    """
    # Ensure the directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return False

    # Get all files in the directory and subdirectories
    all_files = []

    # Check if the directory is named "Data" or contains a "Data" subdirectory
    data_dir = directory_path
    if os.path.basename(directory_path) != "Data":
        potential_data_dir = os.path.join(directory_path, "Data")
        if os.path.isdir(potential_data_dir):
            data_dir = potential_data_dir

    for root, _, files in os.walk(directory_path):
        for file in files:
            # Get the full path of the file
            full_path = os.path.join(root, file)

            # Create path in the format "Data\\path\\to\\file"
            if os.path.basename(data_dir) == "Data":
                # If we're in a Data directory, make paths relative to it
                if full_path.startswith(data_dir):
                    rel_path = os.path.relpath(full_path, os.path.dirname(data_dir))
                else:
                    # If file is outside Data directory, skip it
                    continue
            else:
                # If no Data directory found, use the base directory name as the start
                rel_path = os.path.relpath(full_path, os.path.dirname(directory_path))
                # Prepend "Data\\" to match the format
                rel_path = os.path.join("Data", rel_path)

            # Convert forward slashes to backslashes for Windows style
            rel_path = rel_path.replace('/', '\\')
            all_files.append(rel_path)

    # Sort files for consistent output
    all_files.sort()

    # Check if any files were found
    if not all_files:
        print(f"Warning: No files found in directory '{directory_path}'.")
        # Still create an empty archlist file
        with open(output_file, 'w') as f:
            f.write('[\n]\n')
        return True

    # Write to the archlist file
    with open(output_file, 'w') as f:
        f.write('[\n')
        for i, file_path in enumerate(all_files):
            # Escape backslashes for proper output format
            escaped_path = file_path.replace('\\', '\\\\')
            # Add comma after each entry except the last one
            if i < len(all_files) - 1:
                f.write(f'\t"{escaped_path}",\n')
            else:
                f.write(f'\t"{escaped_path}"\n')
        f.write(']\n')

    print(f"Added {len(all_files)} files to the archlist.")
    return True

class ArchlistWidget(BaseWidget):
    def __init__(self, parent, text):

        super().__init__(parent=parent, text=text, vertical=True)
        self.main_widget = QWidget()

        self.input_dir = ConfigItem("arch", "input_dir", "", FolderValidator())

        self.input_dir_card = PushSettingCard(
            self.tr('Source Directory'),
            FIF.FOLDER,
            self.tr("All Files and Sub Directories will be added to the archlist."),
            self.input_dir.value
        )
        self.save_as = ConfigItem("arch", "save_as", "", FileValidator('archlist'))

        self.save_as_card = PushSettingCard(
            self.tr('Save As'),
            FIF.SAVE,
            self.tr("Save As..."),
            self.save_as.value
        )

        self.create_btn = PrimaryPushButton(icon=FIF.RIGHT_ARROW, text='Create Archlist')

        main_layout = QtWidgets.QVBoxLayout(self.main_widget)
        main_layout.addWidget(self.input_dir_card)
        main_layout.addWidget(self.save_as_card)
        main_layout.addStretch(1)

        self.input_dir_card.clicked.connect(self.__onOutputFolderCardClicked)
        self.save_as_card.clicked.connect(self.__onSaveCardClicked)
        self.create_btn.clicked.connect(self.run_create)

        self.addToFrame(self.main_widget)
        self.addButtonBarToBottom(self.create_btn)

        self.settings_widget = ArchlistSettings(self)
        self.settings_drawer.addWidget(self.settings_widget)

        self.help_widget = ArchlistHelp(self)
        self.help_drawer.addWidget(self.help_widget)


    def __onOutputFolderCardClicked(self):
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Choose Input Directory"), "../")
        if not folder:
            return

        self.input_dir.value = folder
        self.input_dir_card.setContent(folder)

    def __onSaveCardClicked(self):
        start_dir = os.path.dirname(self.save_as.value.strip() or '')
        start_name = os.path.basename(self.save_as.value.strip() or 'archive.achlist')
        initial = os.path.join(start_dir, start_name) if start_dir or start_name else ''
        folder, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            'Select Output File',
            str(initial),
            'Archlist (*.achlist *.archlist);;All Files (*.*)'
        )
        if not folder:
            return

        self.save_as.value = folder
        self.save_as_card.setContent(folder)

    def run_create(self):
        directory_path = self.input_dir.value.strip()
        output_file = self.save_as.value.strip()

        if not directory_path:
            InfoBar.warning(
                title=self.tr('Validation'),
                content=self.tr('Please choose a source folder.'),
                duration=3000,
                parent=self,
            )
            return
        if not os.path.isdir(directory_path):
            InfoBar.error(
                title=self.tr('Error'),
                content=self.tr(f"Directory does not exist:\n{directory_path}"),
                duration=5000,
                parent=self,
            )
            return
        if not output_file:
            InfoBar.warning(
                title=self.tr('Validation'),
                content=self.tr('Please choose an output file path.'),
                duration=3000,
                parent=self,
            )
            return
        out_dir = os.path.dirname(output_file)
        if out_dir and not os.path.isdir(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                InfoBar.error(
                    title=self.tr('Error'),
                    content=self.tr(f"Cannot create output directory:\n{out_dir}\n\n{e}"),
                    duration=5000,
                    parent=self,
                )
                return

        try:
            ok = create_archlist(directory_path, output_file)
        except Exception as e:
            InfoBar.error(
                title=self.tr('Error'),
                content=self.tr(f"Failed to create archlist file:\n{e}"),
                duration=5000,
                parent=self,
            )
            return

        if ok:
            InfoBar.success(
                title=self.tr('Success'),
                content=self.tr(f"Archlist file created successfully:\n{output_file}"),
                duration=3000,
                parent=self,
            )
        else:
            InfoBar.warning(
                title=self.tr('Result'),
                content=self.tr('No files found or operation failed.'),
                duration=3000,
                parent=self,
            )
