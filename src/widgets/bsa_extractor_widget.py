import os
import subprocess
from typing import Optional, List, Tuple

from PySide6 import QtWidgets
from PySide6.QtCore import QThread, Signal
from qfluentwidgets import (
    PushSettingCard,
    PrimaryPushButton,
    SwitchSettingCard,
    ComboBoxSettingCard,
    InfoBar,
    FluentIcon as FIF,
)

from src.utils.appconfig import cfg
from src.utils.filesystem_utils import get_app_root
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger


class BSAExtractorWorker(QThread):
    """Worker that invokes BSArch.exe to pack/unpack archives."""

    progress = Signal(int, int)  # current, total
    info = Signal(str)
    error = Signal(str)
    finished = Signal(int, int)  # processed, failed

    def __init__(
        self,
        input_path: str,
        output_path: str,
        mode_pack: bool,
        game_format: str,
        compress: bool,
        include_subdirs: bool = False,
    ):
        super().__init__()
        self._abort = False
        self.input_path = input_path
        self.output_path = output_path
        self.mode_pack = mode_pack
        self.game_format = game_format  # e.g., "-sse", "-fo4"
        self.compress = compress
        self.include_subdirs = include_subdirs

        self.app_root = get_app_root()
        self.exe_path = os.path.join(self.app_root, "resource", "apps", "xedit", "BSArch.exe")

    def abort(self):
        self._abort = True

    def _is_archive(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in [".bsa", ".ba2"]

    def _collect_tasks(self) -> Tuple[List[Tuple[str, Optional[str]]], Optional[str]]:
        """Return list of (file_path, rel_root) to process and a base_dir for rel paths."""
        ip = os.path.abspath(self.input_path)
        tasks: List[Tuple[str, Optional[str]]] = []

        if self.mode_pack:
            # Packing always takes a single source (which can be a directory or a file)
            # and produces a single archive.
            return [(ip, None)], os.path.dirname(ip)

        if os.path.isfile(ip):
            if not self._is_archive(ip):
                return [], None
            return [(ip, None)], os.path.dirname(ip)

        # Directory mode for unpacking
        base_dir = ip
        if self.include_subdirs:
            walker = os.walk(base_dir)
        else:
            files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
            walker = [(base_dir, [], files)]

        for root, _dirs, files in walker:
            rel = os.path.relpath(root, base_dir)
            for f in files:
                path = os.path.join(root, f)
                if self._is_archive(path):
                    tasks.append((path, rel))

        return tasks, base_dir

    def run(self):
        try:
            if not os.path.exists(self.exe_path):
                err = f"BSArch.exe not found at {self.exe_path}"
                logger.error(err)
                self.error.emit(err)
                self.finished.emit(0, 1)
                return

            tasks, base_dir = self._collect_tasks()
            total = len(tasks)
            if total == 0:
                msg = "No matching archives to process."
                logger.info(msg)
                self.info.emit(msg)
                self.finished.emit(0, 0)
                return

            processed = 0
            failed = 0

            self.progress.emit(0, total)

            for i, (src, rel) in enumerate(tasks, start=1):
                if self._abort:
                    logger.info("BSAExtractorWorker aborted by user.")
                    break

                cmd = [self.exe_path]
                if self.mode_pack:
                    # BSArch.exe pack <source> <archive> [parameters]
                    cmd.append("pack")
                    cmd.append(src)
                    
                    # If output path is empty, default to input path but change extension/add .bsa
                    out = self.output_path.strip()
                    if not out:
                        if os.path.isdir(src):
                            out = src.rstrip("\\/") + ".bsa"
                        else:
                            out = os.path.splitext(src)[0] + ".bsa"
                    
                    cmd.append(out)
                    cmd.append(self.game_format)
                    if self.compress:
                        cmd.append("-z")
                else:
                    # BSArch.exe unpack <archive> [folder] [parameters]
                    cmd.append("unpack")
                    cmd.append(src)
                    
                    out_dir = self.output_path.strip()
                    if out_dir:
                        # If multiple tasks, mirror rel structure
                        if total > 1:
                            rel_dir = rel if (rel and rel != ".") else ""
                            # For unpacking a BSA, usually we want it in its own folder named after the BSA
                            bsa_name = os.path.splitext(os.path.basename(src))[0]
                            final_out = os.path.join(out_dir, rel_dir, bsa_name)
                        else:
                            final_out = out_dir
                        
                        os.makedirs(final_out, exist_ok=True)
                        cmd.append(final_out)

                logger.info("Running BSArch command: %s", " ".join(cmd))
                self.info.emit(f"Processing: {os.path.basename(src)}")

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    shell=False,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )

                stdout, _ = process.communicate()

                if process.returncode != 0:
                    logger.error("BSArch failed with return code %d for %s\nOutput: %s", process.returncode, src, stdout)
                    self.error.emit(f"Failed {os.path.basename(src)}: {stdout.strip()}")
                    failed += 1
                else:
                    processed += 1

                self.progress.emit(i, total)

            logger.info("BSAExtractorWorker finished: processed=%d failed=%d", processed, failed)
            self.finished.emit(processed, failed)

        except Exception as e:
            logger.exception("BSAExtractorWorker crashed")
            self.error.emit(f"Worker crashed: {e}")
            self.finished.emit(0, 1)


class BSAExtractorWidget(BaseWidget):
    """UI for packing/unpacking BSA files using BSArch.exe."""

    def __init__(self, parent: Optional[QtWidgets.QWidget], text: str = "BSA Extractor"):
        super().__init__(parent=parent, text=text, vertical=True)

        self._worker: Optional[BSAExtractorWorker] = None
        self._input_file: str = ""
        self._input_dir: str = ""
        self._output_path: str = ""

        self.input_file_card = PushSettingCard(
            self.tr("Input File"),
            CustomIcons.FILE.icon(stroke=False),
            self.tr("Select a .bsa/.ba2 file to unpack"),
            self.tr("Browse"),
        )
        self.input_dir_card = PushSettingCard(
            self.tr("Input Folder"),
            CustomIcons.FOLDER_IMAGE.icon(stroke=True),
            self.tr("Select a folder to pack, or a folder containing archives to unpack"),
            self.tr("Browse"),
        )
        self.output_card = PushSettingCard(
            self.tr("Output Path (Optional)"),
            CustomIcons.FOLDERRIGHT.icon(stroke=True),
            self.tr("Default is same as input"),
            self.tr("Browse"),
        )

        self.pack_switch = SwitchSettingCard(
            icon=CustomIcons.BULK_EDIT.icon(),
            title=self.tr("Pack mode"),
            content=self.tr("On: pack directory to BSA. Off: unpack BSA(s) to directory."),
            configItem=cfg.bsa_pack_mode,
        )

        self.subdirs_switch = SwitchSettingCard(
            icon=CustomIcons.SUB.icon(stroke=False),
            title=self.tr("Include subfolders"),
            content=self.tr("When a folder is selected for unpacking, also process subdirectories"),
        )

        self.format_card = ComboBoxSettingCard(
            configItem=cfg.bsa_format,
            icon=FIF.GAME,
            title=self.tr("Archive Format"),
            content=self.tr("Select the target game format for packing"),
            texts=[
                "Skyrim SE/AE (-sse)",
                "Skyrim LE/FO3/FNV (-tes5)",
                "Fallout 4 (-fo4)",
                "Fallout 4 DDS (-fo4dds)",
                "Starfield (-sf1)",
                "Starfield DDS (-sf1dds)",
                "Morrowind (-tes3)",
                "Oblivion (-tes4)",
            ]
        )
        self.format_mapping = {
            "Skyrim SE/AE (-sse)": "-sse",
            "Skyrim LE/FO3/FNV (-tes5)": "-tes5",
            "Fallout 4 (-fo4)": "-fo4",
            "Fallout 4 DDS (-fo4dds)": "-fo4dds",
            "Starfield (-sf1)": "-sf1",
            "Starfield DDS (-sf1dds)": "-sf1dds",
            "Morrowind (-tes3)": "-tes3",
            "Oblivion (-tes4)": "-tes4",
        }

        self.compress_switch = SwitchSettingCard(
            icon=FIF.ZIP_FOLDER,
            title=self.tr("Compress"),
            content=self.tr("Enable compression (-z)"),
            configItem=cfg.bsa_compress,
        )

        self.addToFrame(self.input_file_card)
        self.addToFrame(self.input_dir_card)
        self.addToFrame(self.output_card)
        self.addToFrame(self.pack_switch)
        self.addToFrame(self.subdirs_switch)
        self.addToFrame(self.format_card)
        self.addToFrame(self.compress_switch)
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
            self, self.tr("Select BSA/BA2 Archive"), self._input_file or self._input_dir or os.getcwd(),
            self.tr("Archives (*.bsa *.ba2);;All Files (*.*)")
        )
        if path:
            self._input_file = path
            self._input_dir = ""
            self.input_file_card.setContent(path)
            self.input_dir_card.setContent(self.tr("(unused; file selected)"))

    def _on_pick_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select Folder"), self._input_dir or self._input_file or os.getcwd()
        )
        if path:
            self._input_dir = path
            self._input_file = ""
            self.input_dir_card.setContent(path)
            self.input_file_card.setContent(self.tr("(unused; folder selected)"))

    def _on_pick_output(self):
        if self.pack_switch.isChecked():
            # Packing: output is a file
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, self.tr("Select Output Archive"), self._output_path or os.getcwd(),
                self.tr("Archives (*.bsa *.ba2);;All Files (*.*)")
            )
        else:
            # Unpacking: output is a folder
            path = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr("Select Output Folder"), self._output_path or os.getcwd())
        
        if path:
            self._output_path = path
            self.output_card.setContent(path)

    def _on_run(self):
        input_path = self._input_file or self._input_dir
        if not input_path:
            InfoBar.error(self.tr("Error"), self.tr("Please select an input path."), parent=self)
            return

        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        game_format_label = self.format_card.comboBox.currentText()
        game_format = self.format_mapping.get(game_format_label, "-sse")

        self._worker = BSAExtractorWorker(
            input_path=input_path,
            output_path=self._output_path,
            mode_pack=self.pack_switch.isChecked(),
            game_format=game_format,
            compress=self.compress_switch.isChecked(),
            include_subdirs=self.subdirs_switch.isChecked()
        )
        self._worker.info.connect(lambda msg: logger.info(f"[BSAExtractor] {msg}"))
        self._worker.error.connect(lambda msg: InfoBar.error(self.tr("Error"), msg, parent=self))
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_progress(self, current: int, total: int):
        if total == 0:
            return
        self.run_button.setText(self.tr(f"{current}/{total}"))

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.abort()
            self.stop_button.setEnabled(False)

    def _on_worker_finished(self, processed, failed):
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.run_button.setText(self.tr("Run"))
        if failed == 0 and processed > 0:
            InfoBar.success(self.tr("Success"), self.tr(f"Operation completed: {processed} files processed."), parent=self)
        elif failed > 0:
            InfoBar.error(self.tr("Done"), self.tr(f"Processed: {processed}, Failed: {failed}"), parent=self)
        self._worker = None
