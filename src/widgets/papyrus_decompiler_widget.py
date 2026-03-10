import os
import subprocess
from typing import Optional

from PySide6 import QtWidgets
from PySide6.QtCore import QThread, Signal
from qfluentwidgets import (
    PushSettingCard,
    PrimaryPushButton,
    SwitchSettingCard,
    InfoBar,
    FluentIcon as FIF,
)

from src.utils.appconfig import cfg
from src.utils.filesystem_utils import get_app_root
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger


class PapyrusDecompilerWorker(QThread):
    """Worker that decompiles .pex files using Champollion."""

    progress = Signal(int, int)  # current, total
    info = Signal(str)
    error = Signal(str)
    finished = Signal(int, int)  # processed, failed

    def __init__(
        self,
        input_path: str,
        output_dir: Optional[str],
        include_subdirs: bool,
        recreate_subdirs: bool,
        assemble: bool,
        comment: bool,
        threaded: bool,
    ):
        super().__init__()
        self._abort = False
        self.input_path = input_path
        self.output_dir = output_dir
        self.include_subdirs = include_subdirs
        self.recreate_subdirs = recreate_subdirs
        self.assemble = assemble
        self.comment = comment
        self.threaded = threaded
        self._process = None

    def abort(self):
        self._abort = True
        if self._process:
            try:
                self._process.terminate()
            except:
                pass

    def run(self):
        try:
            champollion_path = os.path.join(get_app_root(), "resource", "apps", "champollion", "Champollion.exe")
            if not os.path.exists(champollion_path):
                self.error.emit(f"Champollion.exe not found at {champollion_path}")
                self.finished.emit(0, 1)
                return

            command = [champollion_path, self.input_path]

            if self.output_dir:
                command.extend(["-p", self.output_dir])
            
            if self.assemble:
                # Champollion uses -a [assembly directory]. If we don't have a separate dir, 
                # we might just use the output_dir or same as input.
                # The docs say: Champollion will write an assembly version of the PEX file in the given directory
                asm_dir = self.output_dir if self.output_dir else os.path.dirname(self.input_path) if os.path.isfile(self.input_path) else self.input_path
                command.extend(["-a", asm_dir])

            if self.comment:
                command.append("-c")
            
            if self.threaded:
                command.append("-t")
            
            if self.include_subdirs:
                command.append("-r")
            
            if self.recreate_subdirs:
                command.append("-s")

            logger.info("PapyrusDecompilerWorker running: %s", " ".join(command))
            
            # Champollion handles multiple files/directories itself, so we just invoke it once.
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )

            # We don't have a granular progress from Champollion easily without parsing stdout.
            # For now, we'll just wait for it to finish.
            stdout, _ = self._process.communicate()
            
            if self._process.returncode == 0:
                logger.info("Champollion finished successfully")
                self.finished.emit(1, 0) # Just mark as 1 "task" done
            else:
                logger.error("Champollion failed with return code %d: %s", self._process.returncode, stdout)
                self.error.emit(f"Champollion failed: {stdout[:200]}...")
                self.finished.emit(0, 1)

        except Exception as e:
            logger.exception("PapyrusDecompilerWorker crashed")
            self.error.emit(f"Worker crashed: {e}")
            self.finished.emit(0, 1)


class PapyrusDecompilerWidget(BaseWidget):
    """UI for decompiling PEX files using Champollion."""

    def __init__(self, parent: Optional[QtWidgets.QWidget], text: str = "Papyrus Decompiler"):
        super().__init__(parent=parent, text=text, vertical=True)

        self._worker: Optional[PapyrusDecompilerWorker] = None
        self._input_file: str = ""
        self._input_dir: str = ""
        self._output_dir: str = ""

        self.input_file_card = PushSettingCard(
            self.tr("Input File"),
            CustomIcons.FILE.icon(stroke=False),
            self.tr("Select a .pex file to decompile"),
            self.tr("Browse"),
        )
        self.input_dir_card = PushSettingCard(
            self.tr("Input Folder"),
            CustomIcons.FOLDER_IMAGE.icon(stroke=True),
            self.tr("Select a folder containing .pex files"),
            self.tr("Browse"),
        )
        self.output_card = PushSettingCard(
            self.tr("Output Folder (Optional)"),
            CustomIcons.FOLDERRIGHT.icon(stroke=True),
            self.tr("Default is same as input"),
            self.tr("Browse"),
        )

        self.subdirs_switch = SwitchSettingCard(
            icon=CustomIcons.SUB.icon(stroke=False),
            title=self.tr("Include subfolders"),
            content=self.tr("Recursively scan specified directory(s) for pex files"),
            configItem=cfg.papyrus_include_subdirs,
        )

        self.recreate_subdirs_switch = SwitchSettingCard(
            icon=FIF.FOLDER,
            title=self.tr("Recreate directory structure"),
            content=self.tr("Recreates directory structure for script in root of output directory (Fallout 4 only)"),
            configItem=cfg.papyrus_recreate_subdirs,
        )

        self.assemble_switch = SwitchSettingCard(
            icon=FIF.EDIT,
            title=self.tr("Write assembly"),
            content=self.tr("Write an assembly version of the PEX file"),
            configItem=cfg.papyrus_assemble,
        )

        self.comment_switch = SwitchSettingCard(
            icon=FIF.SCROLL,
            title=self.tr("Annotate with assembly"),
            content=self.tr("Decompiled file will be annotated with assembly instructions"),
            configItem=cfg.papyrus_comment,
        )

        self.threaded_switch = SwitchSettingCard(
            icon=FIF.DEVELOPER_TOOLS,
            title=self.tr("Threaded decompilation"),
            content=self.tr("Parallelize the decompilation"),
            configItem=cfg.papyrus_threaded,
        )

        self.addToFrame(self.input_file_card)
        self.addToFrame(self.input_dir_card)
        self.addToFrame(self.output_card)
        self.addToFrame(self.subdirs_switch)
        self.addToFrame(self.recreate_subdirs_switch)
        self.addToFrame(self.assemble_switch)
        self.addToFrame(self.comment_switch)
        self.addToFrame(self.threaded_switch)
        self.boxLayout.addStretch(1)

        self.run_button = PrimaryPushButton(icon=FIF.PLAY, text=self.tr("Run"))
        self.stop_button = PrimaryPushButton(icon=FIF.CANCEL, text=self.tr("Stop"))
        self.stop_button.setEnabled(False)

        self.input_file_card.clicked.connect(self._on_pick_file)
        self.input_dir_card.clicked.connect(self._on_pick_dir)
        self.output_card.clicked.connect(self._on_pick_output)
        self.run_button.clicked.connect(self._on_run)
        self.stop_button.clicked.connect(self._on_stop)

        self.buttons_layout.addWidget(self.stop_button, stretch=1)
        self.addButtonBarToBottom(self.run_button)

    def _on_pick_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, self.tr("Select PEX File"), os.getcwd(), self.tr("PEX Files (*.pex);;All Files (*.*)")
        )
        if path:
            self._input_file = path
            self._input_dir = ""
            self.input_file_card.setContent(path)
            self.input_dir_card.setContent(self.tr("(unused; file selected)"))

    def _on_pick_dir(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr("Select Folder"), os.getcwd())
        if directory:
            self._input_dir = directory
            self._input_file = ""
            self.input_dir_card.setContent(directory)
            self.input_file_card.setContent(self.tr("(unused; folder selected)"))

    def _on_pick_output(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr("Select Output Folder"), os.getcwd())
        if directory:
            self._output_dir = directory
            self.output_card.setContent(directory)

    def _show_info(self, text: str):
        InfoBar.success(self.tr("Info"), text, parent=self)

    def _show_error(self, text: str):
        InfoBar.error(self.tr("Error"), text, parent=self)

    def _on_run(self):
        if self._worker:
            return

        input_path = self._input_file or self._input_dir
        if not input_path:
            self._show_error(self.tr("Please select a file or folder first."))
            return

        self._worker = PapyrusDecompilerWorker(
            input_path=input_path,
            output_dir=self._output_dir if self._output_dir else None,
            include_subdirs=self.subdirs_switch.isChecked(),
            recreate_subdirs=self.recreate_subdirs_switch.isChecked(),
            assemble=self.assemble_switch.isChecked(),
            comment=self.comment_switch.isChecked(),
            threaded=self.threaded_switch.isChecked(),
        )

        self._worker.info.connect(self._show_info)
        self._worker.error.connect(self._show_error)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)

        self.stop_button.setEnabled(True)
        self.run_button.setEnabled(False)
        self._worker.start()

    def _on_stop(self):
        if self._worker:
            self._worker.abort()
            self._show_info(self.tr("Stopping..."))

    def _on_progress(self, current: int, total: int):
        if total > 0:
            self.run_button.setText(f"{current}/{total}")

    def _on_finished(self, processed: int, failed: int):
        self.stop_button.setEnabled(False)
        self.run_button.setEnabled(True)
        self.run_button.setText(self.tr("Run"))
        self._worker = None
        if failed:
            self._show_error(self.tr(f"Decompilation failed"))
        else:
            self._show_info(self.tr(f"Decompilation finished"))
