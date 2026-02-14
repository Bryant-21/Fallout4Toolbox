import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Tuple

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
from src.utils.audio_utils import extract_fuz, create_xwm
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger


class AudioExtractorWorker(QThread):
    """Worker that extracts .wav from .fuz or .xwm files."""

    progress = Signal(int, int)  # current, total
    info = Signal(str)
    error = Signal(str)
    finished = Signal(int, int)  # processed, failed

    def __init__(
        self,
        input_path: str,
        output_dir: Optional[str],
        include_subdirs: bool,
        keep_intermediate: bool,
    ):
        super().__init__()
        self._abort = False
        self.input_path = input_path
        self.output_dir = output_dir
        self.include_subdirs = include_subdirs
        self.keep_intermediate = keep_intermediate

    def abort(self):
        self._abort = True

    def _is_audio(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in [".fuz", ".xwm"]

    def _collect_tasks(self) -> Tuple[List[Tuple[str, Optional[str]]], Optional[str]]:
        """Return list of (file_path, rel_root) to process and a base_dir for rel paths."""
        ip = os.path.abspath(self.input_path)
        tasks: List[Tuple[str, Optional[str]]] = []

        if os.path.isfile(ip):
            if not self._is_audio(ip):
                return [], None
            return [(ip, None)], os.path.dirname(ip)

        # Directory mode
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
                if self._is_audio(path):
                    tasks.append((path, rel))

        return tasks, base_dir

    def _process_single_file(self, src: str, rel: str, target_dir_base: Optional[str]) -> bool:
        """Process a single audio file. Returns True if successful."""
        if self._abort:
            return False

        try:
            ext = os.path.splitext(src)[1].lower()
            base_name = os.path.splitext(os.path.basename(src))[0]

            # Determine target output path
            if target_dir_base:
                rel_dir = rel if (rel and rel != ".") else ""
                target_dir = os.path.join(target_dir_base, rel_dir)
                os.makedirs(target_dir, exist_ok=True)
            else:
                target_dir = os.path.dirname(src)

            target_wav = os.path.join(target_dir, base_name + ".wav")

            if ext == ".fuz":
                # FUZ extraction logic
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_fuz = os.path.join(temp_dir, os.path.basename(src))
                    shutil.copy2(src, temp_fuz)

                    extract_fuz(temp_fuz)

                    temp_xwm = os.path.join(temp_dir, base_name + ".xwm")
                    temp_lip = os.path.join(temp_dir, base_name + ".lip")

                    if os.path.exists(temp_xwm):
                        # Convert XWM to WAV in target dir
                        create_xwm(temp_xwm, target_wav, encode=False)

                        if self.keep_intermediate:
                            shutil.copy2(temp_xwm, os.path.join(target_dir, base_name + ".xwm"))
                            if os.path.exists(temp_lip):
                                shutil.copy2(temp_lip, os.path.join(target_dir, base_name + ".lip"))
                            else:
                                logger.warning("No LIP file found for FUZ: %s", os.path.basename(src))
                        return True
                    else:
                        raise Exception(f"Failed to extract XWM from FUZ: {os.path.basename(src)}")

            elif ext == ".xwm":
                # XWM conversion logic
                create_xwm(src, target_wav, encode=False)
                return True

        except Exception as e:
            logger.exception("Failed to process audio file: %s", src)
            self.error.emit(f"Error {os.path.basename(src)}: {str(e)}")
            return False

        return False

    def run(self):
        try:
            tasks, base_dir = self._collect_tasks()
            total = len(tasks)
            if total == 0:
                msg = "No matching audio files to process."
                logger.info(msg)
                self.info.emit(msg)
                self.finished.emit(0, 0)
                return

            processed = 0
            failed = 0
            completed_count = 0
            
            # Lock for thread-safe counters
            lock = threading.Lock()

            self.progress.emit(0, total)

            # Use cfg.threads_cfg if available, else default to CPU count
            max_workers = cfg.threads_cfg.value if hasattr(cfg, 'threads_cfg') else (os.cpu_count() or 1)
            logger.info("AudioExtractorWorker starting with %d threads for %d tasks", max_workers, total)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for src, rel in tasks:
                    futures.append(executor.submit(self._process_single_file, src, rel, self.output_dir))

                for future in futures:
                    if self._abort:
                        break
                    
                    success = future.result()
                    
                    with lock:
                        completed_count += 1
                        if success:
                            processed += 1
                        else:
                            failed += 1
                        self.progress.emit(completed_count, total)

            if self._abort:
                logger.info("AudioExtractorWorker aborted by user.")

            logger.info("AudioExtractorWorker finished: processed=%d failed=%d", processed, failed)
            self.finished.emit(processed, failed)

        except Exception as e:
            logger.exception("AudioExtractorWorker crashed")
            self.error.emit(f"Worker crashed: {e}")
            self.finished.emit(0, 1)


class AudioExtractorWidget(BaseWidget):
    """UI for extracting WAV from FUZ/XWM files."""

    def __init__(self, parent: Optional[QtWidgets.QWidget], text: str = "Audio Extractor"):
        super().__init__(parent=parent, text=text, vertical=True)

        self._worker: Optional[AudioExtractorWorker] = None
        self._input_file: str = ""
        self._input_dir: str = ""
        self._output_dir: str = ""

        self.input_file_card = PushSettingCard(
            self.tr("Input File"),
            CustomIcons.FILE.icon(stroke=False),
            self.tr("Select a .fuz or .xwm file to extract"),
            self.tr("Browse"),
        )
        self.input_dir_card = PushSettingCard(
            self.tr("Input Folder"),
            CustomIcons.FOLDER_IMAGE.icon(stroke=True),
            self.tr("Select a folder containing audio files to extract"),
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
            content=self.tr("When a folder is selected, also process subdirectories"),
            configItem=cfg.audio_include_subdirs,
        )

        self.keep_intermediate_switch = SwitchSettingCard(
            icon=FIF.SAVE,
            title=self.tr("Keep intermediate files"),
            content=self.tr("Keep .xwm and .lip files extracted from .fuz"),
            configItem=cfg.audio_keep_intermediate,
        )

        self.addToFrame(self.input_file_card)
        self.addToFrame(self.input_dir_card)
        self.addToFrame(self.output_card)
        self.addToFrame(self.subdirs_switch)
        self.addToFrame(self.keep_intermediate_switch)
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
            self, self.tr("Select Audio File"), os.getcwd(), self.tr("Audio Files (*.fuz *.xwm);;All Files (*.*)")
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

        self._worker = AudioExtractorWorker(
            input_path=input_path,
            output_dir=self._output_dir if self._output_dir else None,
            include_subdirs=self.subdirs_switch.isChecked(),
            keep_intermediate=self.keep_intermediate_switch.isChecked(),
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
            self._show_error(self.tr(f"Done: {processed} OK, {failed} failed"))
        else:
            self._show_info(self.tr(f"Done: {processed} OK"))
