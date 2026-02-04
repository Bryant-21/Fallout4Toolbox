import os
import subprocess
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

from src.utils.filesystem_utils import get_app_root
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger


def _is_hkx(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".hkx"


def _is_xml(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".xml"


class HKXPackWorker(QThread):
    """Worker that invokes hkxpack-cli.jar to pack/unpack files.

    - If given a file, processes that single file.
    - If given a directory, walks it and (optionally) subdirectories,
      building a list of matching files based on the selected mode.
    - Outputs next to source by default, or mirrors directory layout
      under the provided output root using the CLI `-o` argument.
    """

    progress = Signal(int, int)  # current, total
    info = Signal(str)
    error = Signal(str)
    finished = Signal(int, int)  # processed, failed

    def __init__(
        self,
        input_path: str,
        mode_pack: bool,
        include_subdirs: bool,
        output_dir: Optional[str],
        verbosity: str,  # "normal" | "quiet" | "verbose"
    ):
        super().__init__()
        self._abort = False
        self.input_path = input_path
        self.mode_pack = mode_pack
        self.include_subdirs = include_subdirs
        self.output_dir = (output_dir or "").strip()
        self.verbosity = verbosity

        self.app_root = get_app_root()
        self.jar_path = os.path.join(self.app_root, "resource", "hkxpack-cli.jar")

    def abort(self):
        self._abort = True

    # ---------------------------- helpers ----------------------------
    def _collect_tasks(self) -> Tuple[List[Tuple[str, Optional[str]]], Optional[str]]:
        """Return list of (file_path, rel_root) to process and a base_dir for rel paths.

        rel_root is the relative directory path under base_dir for the file's directory.
        """
        ip = os.path.abspath(self.input_path)
        tasks: List[Tuple[str, Optional[str]]] = []

        if os.path.isfile(ip):
            # Single-file mode
            if self.mode_pack and not _is_xml(ip):
                return [], None
            if (not self.mode_pack) and not _is_hkx(ip):
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
                if self.mode_pack:
                    if _is_xml(path):
                        tasks.append((path, rel))
                else:
                    if _is_hkx(path):
                        tasks.append((path, rel))

        return tasks, base_dir

    def _java_cmd(self, op: str, src: str, out_dir: Optional[str]) -> List[str]:
        cmd = ["java", "-jar", self.jar_path, op, src]
        if self.verbosity == "quiet":
            cmd.insert(3, "-q")  # after jar path
        elif self.verbosity == "verbose":
            cmd.insert(3, "-v")
        if out_dir:
            cmd.extend(["-o", out_dir])
        return cmd

    # ------------------------------ run ------------------------------
    def run(self):  # pragma: no cover - UI thread
        try:
            if not os.path.exists(self.jar_path):
                err = "hkxpack-cli.jar not found in resource folder."
                logger.error(err + f" path={self.jar_path}")
                self.error.emit(err)
                self.finished.emit(0, 1)
                return

            tasks, base_dir = self._collect_tasks()
            total = len(tasks)
            logger.info(
                "HKXPackWorker starting: mode=%s, input=%s, total_tasks=%d, include_subdirs=%s, output_dir=%s",
                "pack" if self.mode_pack else "unpack",
                self.input_path,
                total,
                self.include_subdirs,
                self.output_dir or "<same as source>",
            )
            self.progress.emit(0, total)
            if total == 0:
                msg = "No matching files to process."
                logger.info(msg)
                self.info.emit(msg)
                self.finished.emit(0, 0)
                return

            processed = 0
            failed = 0

            for i, (src, rel) in enumerate(tasks, start=1):
                if self._abort:
                    logger.info("HKXPackWorker aborted by user after %d/%d tasks", i - 1, total)
                    break

                # Determine output directory for this file
                out_dir = None
                if self.output_dir:
                    # Mirror rel folder under output root
                    rel_dir = rel if (rel and rel != ".") else ""
                    out_dir = os.path.join(self.output_dir, rel_dir) if rel_dir else self.output_dir
                    os.makedirs(out_dir, exist_ok=True)

                op = "pack" if self.mode_pack else "unpack"
                cmd = self._java_cmd(op, src, out_dir)

                try:
                    # On Windows, avoid shell=True; capture output for feedback
                    logger.debug("Running command: %s", " ".join(cmd))
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        shell=False,
                    )
                    if result.returncode != 0:
                        failed += 1
                        msg = result.stdout.strip() or f"{op} failed for {os.path.basename(src)}"
                        logger.error(
                            "HKXPack %s failed (rc=%d) for file=%s out_dir=%s\nOutput:\n%s",
                            op,
                            result.returncode,
                            src,
                            out_dir or "<source dir>",
                            result.stdout,
                        )
                        self.error.emit(msg)
                    else:
                        processed += 1
                        # Be brief unless verbose selected
                        if self.verbosity != "quiet":
                            self.info.emit(f"{op.title()} OK: {os.path.basename(src)}")
                except Exception as e:
                    failed += 1
                    logger.exception("HKXPack %s crashed for file=%s out_dir=%s", op, src, out_dir or "<source dir>")
                    self.error.emit(f"{op} crashed for {os.path.basename(src)}: {e}")

                self.progress.emit(i, total)

            logger.info("HKXPackWorker finished: processed=%d failed=%d", processed, failed)
            self.finished.emit(processed, failed)

        except Exception as e:  # catastrophic
            logger.exception("HKXPackWorker crashed fatally")
            self.error.emit(str(e))
            self.finished.emit(0, 1)


class HKXPackWidget(BaseWidget):
    """Simple UI for packing/unpacking HKX/XML using hkxpack-cli.jar.

    Features:
    - Select either a single file (HKX or XML) or a folder
    - Optional output directory; if omitted, outputs next to source
    - Toggle between Pack and Unpack
    - Option to include subfolders when folder is selected
    - Verbosity: Normal, Quiet (-q), Verbose (-v)
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget], text: str = "HKX Pack / Unpack"):
        super().__init__(parent=parent, text=text, vertical=True)

        # Runtime state
        self._worker: Optional[HKXPackWorker] = None
        self._input_file: str = ""
        self._input_dir: str = ""
        self._output_dir: str = ""
        self._verbosity: str = "normal"  # normal | quiet | verbose

        # Cards
        self.input_file_card = PushSettingCard(
            self.tr("Input File (HKX or XML)"),
            CustomIcons.FILE.icon(stroke=False),
            self.tr("Process a single file"),
            self.tr("Please select an input file"),
        )
        self.input_dir_card = PushSettingCard(
            self.tr("Input Folder"),
            CustomIcons.FOLDER_IMAGE.icon(stroke=True),
            self.tr("Process all HKX or XML files in a folder"),
            self.tr("Please select a folder (optional if file chosen)"),
        )
        self.output_dir_card = PushSettingCard(
            self.tr("Output Folder (Optional)"),
            CustomIcons.FOLDERRIGHT.icon(stroke=True),
            self.tr("If empty, files are written next to sources"),
            self.tr("Same as input if empty"),
        )

        self.pack_switch = SwitchSettingCard(
            icon=CustomIcons.FILE_EDIT.icon(stroke=True) if hasattr(CustomIcons, 'FILE_EDIT') else FIF.EDIT,
            title=self.tr("Pack mode"),
            content=self.tr("On: pack XML to HKX. Off: unpack HKX to XML."),
        )
        self.subdirs_switch = SwitchSettingCard(
            icon=CustomIcons.SUB.icon(stroke=False),
            title=self.tr("Include subfolders"),
            content=self.tr("When a folder is selected, also process subdirectories"),
        )
        self.verbose_switch = SwitchSettingCard(
            icon=FIF.INFO,
            title=self.tr("Verbose output"),
            content=self.tr("Show detailed CLI output (-v). Overrides Quiet."),
        )
        self.quiet_switch = SwitchSettingCard(
            icon=FIF.MUTE,
            title=self.tr("Quiet output"),
            content=self.tr("Minimal CLI output (-q)."),
        )

        # Layout
        self.addToFrame(self.input_file_card)
        self.addToFrame(self.input_dir_card)
        self.addToFrame(self.output_dir_card)
        self.addToFrame(self.pack_switch)
        self.addToFrame(self.subdirs_switch)
        self.addToFrame(self.verbose_switch)
        self.addToFrame(self.quiet_switch)
        self.boxLayout.addStretch(1)

        # Controls
        self.run_button = PrimaryPushButton(icon=FIF.PLAY, text=self.tr("Run"))
        self.stop_button = PrimaryPushButton(icon=FIF.CANCEL, text=self.tr("Stop"))
        self.stop_button.setEnabled(False)

        self.input_file_card.clicked.connect(self._on_pick_file)
        self.input_dir_card.clicked.connect(self._on_pick_dir)
        self.output_dir_card.clicked.connect(self._on_pick_out)
        self.verbose_switch.checkedChanged.connect(self._on_verbosity_changed)
        self.quiet_switch.checkedChanged.connect(self._on_verbosity_changed)
        self.run_button.clicked.connect(self._on_run)
        self.stop_button.clicked.connect(self._on_stop)

        self.buttons_layout.addWidget(self.stop_button, stretch=1)
        self.addButtonBarToBottom(self.run_button)

    # ----------------------------- UI slots -----------------------------
    def _on_pick_file(self):
        start_dir = self._input_file or self._input_dir or os.getcwd()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select HKX or XML"),
            start_dir,
            self.tr("HKX/XML Files (*.hkx *.xml);;All Files (*.*)"),
        )
        if path:
            self._input_file = path
            self._input_dir = ""
            self.input_file_card.setContent(path)
            self.input_dir_card.setContent(self.tr("(unused; file selected)"))

    def _on_pick_dir(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select folder"), self._input_dir or os.getcwd()
        )
        if directory:
            self._input_dir = directory
            self._input_file = ""
            self.input_dir_card.setContent(directory)
            self.input_file_card.setContent(self.tr("(unused; folder selected)"))

    def _on_pick_out(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select output folder"), self._output_dir or os.getcwd()
        )
        if directory:
            self._output_dir = directory
            self.output_dir_card.setContent(directory)

    def _on_verbosity_changed(self):
        # Ensure mutual exclusivity: verbose overrides quiet, quiet overrides normal
        sender = self.sender()
        if sender == self.verbose_switch and self.verbose_switch.isChecked():
            if self.quiet_switch.isChecked():
                self.quiet_switch.setChecked(False)
            self._verbosity = "verbose"
        elif sender == self.quiet_switch and self.quiet_switch.isChecked():
            if self.verbose_switch.isChecked():
                self.verbose_switch.setChecked(False)
            self._verbosity = "quiet"
        else:
            # Neither checked
            self._verbosity = "normal"

    def _show_info(self, text: str):
        InfoBar.success(self.tr("Info"), text, parent=self)

    def _show_error(self, text: str):
        logger.error("HKXPack UI error: %s", text)
        InfoBar.error(self.tr("Error"), text, parent=self)

    def _on_run(self):
        if self._worker:
            return

        input_path = self._input_file or self._input_dir
        if not input_path:
            self._show_error(self.tr("Please select a file or a folder first."))
            return

        # Pre-flight checks
        jar_path = os.path.join(get_app_root(), "resource", "hkxpack-cli.jar")
        if not os.path.exists(jar_path):
            logger.error("Missing hkxpack-cli.jar in resource folder: %s", jar_path)
            self._show_error(self.tr("Missing hkxpack-cli.jar in resource/"))
            return

        
        mode_pack = self.pack_switch.isChecked()
        include_subdirs = self.subdirs_switch.isChecked()

        self._worker = HKXPackWorker(
            input_path=input_path,
            mode_pack=mode_pack,
            include_subdirs=include_subdirs,
            output_dir=(self._output_dir or None),
            verbosity=self._verbosity,
        )

        self._worker.info.connect(self._show_info)
        self._worker.error.connect(self._show_error)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)

        self.stop_button.setEnabled(True)
        self.run_button.setEnabled(False)
        op = self.tr("Pack") if mode_pack else self.tr("Unpack")
        logger.info("Starting HKX %s: input=%s, out_dir=%s, include_subdirs=%s, verbosity=%s",
                    op.lower(),
                    self._input_file or self._input_dir,
                    self._output_dir or "<source dir>",
                    include_subdirs,
                    self._verbosity)
        self._show_info(self.tr(f"Starting {op.lower()}..."))
        self._worker.start()

    def _on_stop(self):
        if self._worker:
            self._worker.abort()
            logger.info("HKX pack/unpack stop requested by user")
            self._show_info(self.tr("Stopping..."))

    def _on_progress(self, current: int, total: int):
        if total == 0:
            return
        # Lightweight feedback in title content
        self.run_button.setText(self.tr(f"{current}/{total}"))

    def _on_finished(self, processed: int, failed: int):
        self.stop_button.setEnabled(False)
        self.run_button.setEnabled(True)
        self.run_button.setText(self.tr("Run"))
        self._worker = None
        if failed:
            self._show_error(self.tr(f"Done: {processed} OK, {failed} failed"))
        else:
            self._show_info(self.tr(f"Done: {processed} OK"))
