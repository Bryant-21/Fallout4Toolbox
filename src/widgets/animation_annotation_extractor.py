import os
import xml.etree.ElementTree as ET
from typing import Optional, List, Set

from PySide6 import QtWidgets
from PySide6.QtCore import QThread, Signal
from qfluentwidgets import (
    PushSettingCard,
    PrimaryPushButton,
    SwitchSettingCard,
    InfoBar,
    FluentIcon as FIF,
)

from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger


def _is_xml(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".xml"

class AnnotationExtractorWorker(QThread):
    progress = Signal(int, int)
    info = Signal(str)
    error = Signal(str)
    finished = Signal(int, int)

    def __init__(
        self,
        input_path: str,
        output_dir: str,
        include_subdirs: bool,
    ):
        super().__init__()
        self._abort = False
        self.input_path = input_path
        self.output_dir = output_dir
        self.include_subdirs = include_subdirs
        self.global_annotations: Set[str] = set()
        self.global_event_names: Set[str] = set()

    def abort(self):
        self._abort = True

    def run(self):
        try:
            tasks = self._collect_tasks()
            total = len(tasks)
            if total == 0:
                self.info.emit("No XML files found.")
                self.finished.emit(0, 0)
                return

            self.progress.emit(0, total)
            processed = 0
            
            os.makedirs(self.output_dir, exist_ok=True)

            for i, (src, rel_dir) in enumerate(tasks, start=1):
                if self._abort:
                    break
                
                self._process_file(src, rel_dir)
                processed += 1
                self.progress.emit(i, total)

            self._write_logs()
            self.finished.emit(processed, 0)
        except Exception as e:
            logger.exception("AnnotationExtractorWorker failed")
            self.error.emit(str(e))
            self.finished.emit(0, 1)

    def _collect_tasks(self) -> List[tuple[str, str]]:
        tasks = []
        if os.path.isfile(self.input_path):
            if _is_xml(self.input_path):
                tasks.append((self.input_path, ""))
        else:
            for root, dirs, files in os.walk(self.input_path):
                if not self.include_subdirs and root != self.input_path:
                    continue
                rel_dir = os.path.relpath(root, self.input_path)
                if rel_dir == ".":
                    rel_dir = ""
                for f in files:
                    if _is_xml(f):
                        tasks.append((os.path.join(root, f), rel_dir))
        return tasks

    def _process_file(self, path: str, rel_dir: str):
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            
            # Find all hkobject with class="hkaSplineCompressedAnimation"
            animations = root.findall(".//hkobject[@class='hkaSplineCompressedAnimation']")
            
            file_annotations = set()
            for anim in animations:
                # Find annotationTracks hkparam
                tracks = anim.find("./hkparam[@name='annotationTracks']")
                if tracks is not None:
                    # Iterate over hkobject in tracks
                    for track_obj in tracks.findall("./hkobject"):
                        # Find annotations hkparam
                        annots_param = track_obj.find("./hkparam[@name='annotations']")
                        if annots_param is not None:
                            for annot_obj in annots_param.findall("./hkobject"):
                                text_param = annot_obj.find("./hkparam[@name='text']")
                                if text_param is not None and text_param.text:
                                    file_annotations.add(text_param.text.strip())

            if file_annotations:
                self.global_annotations.update(file_annotations)

            # Find all hkobject with class="hkbBehaviorGraphStringData"
            behavior_string_datas = root.findall(".//hkobject[@class='hkbBehaviorGraphStringData']")
            file_event_names = set()
            for bsd in behavior_string_datas:
                event_names_param = bsd.find("./hkparam[@name='eventNames']")
                if event_names_param is not None:
                    for cstring in event_names_param.findall("./hkcstring"):
                        if cstring.text:
                            file_event_names.add(cstring.text.strip())

            if file_event_names:
                self.global_event_names.update(file_event_names)
                
        except Exception as e:
            logger.warning(f"Failed to process {path}: {e}")

    def _write_logs(self):
        # Global logs
        os.makedirs(self.output_dir, exist_ok=True)
        
        global_annot_path = os.path.join(self.output_dir, "annotations.txt")
        with open(global_annot_path, "w", encoding="utf-8") as f:
            for annot in sorted(self.global_annotations):
                f.write(annot + "\n")

        global_event_path = os.path.join(self.output_dir, "eventNames.txt")
        with open(global_event_path, "w", encoding="utf-8") as f:
            for event in sorted(self.global_event_names):
                f.write(event + "\n")

class AnnotationExtractorWidget(BaseWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget], text: str = "Annotation Extractor"):
        super().__init__(parent=parent, text=text, vertical=True)
        self._input_path = ""
        self._output_dir = ""
        self._worker = None

        self.input_card = PushSettingCard(
            self.tr("Input Path"),
            CustomIcons.FOLDER_IMAGE.icon(stroke=True),
            self.tr("Select XML file or folder"),
            self.tr("No path selected"),
        )
        self.output_card = PushSettingCard(
            self.tr("Output Folder"),
            CustomIcons.FOLDERRIGHT.icon(stroke=True),
            self.tr("Where to save logs"),
            self.tr("No path selected"),
        )
        self.subdirs_switch = SwitchSettingCard(
            icon=CustomIcons.SUB.icon(stroke=False),
            title=self.tr("Include subfolders"),
            content=self.tr("Search for XMLs in subdirectories"),
        )
        self.subdirs_switch.setChecked(True)

        self.addToFrame(self.input_card)
        self.addToFrame(self.output_card)
        self.addToFrame(self.subdirs_switch)
        self.boxLayout.addStretch(1)

        self.run_button = PrimaryPushButton(icon=FIF.PLAY, text=self.tr("Run"))
        self.stop_button = PrimaryPushButton(icon=FIF.CANCEL, text=self.tr("Stop"))
        self.stop_button.setEnabled(False)

        self.input_card.clicked.connect(self._on_pick_input)
        self.output_card.clicked.connect(self._on_pick_output)
        self.run_button.clicked.connect(self._on_run)
        self.stop_button.clicked.connect(self._on_stop)

        self.buttons_layout.addWidget(self.stop_button, stretch=1)
        self.addButtonBarToBottom(self.run_button)

    def _on_pick_input(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr("Select Input Folder"))
        if not path:
             # Try file
             path, _ = QtWidgets.QFileDialog.getOpenFileName(self, self.tr("Select XML File"), filter="XML Files (*.xml)")
        
        if path:
            self._input_path = path
            self.input_card.setContent(path)

    def _on_pick_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr("Select Output Folder"))
        if path:
            self._output_dir = path
            self.output_card.setContent(path)

    def _on_run(self):
        if not self._input_path or not self._output_dir:
            InfoBar.error(self.tr("Error"), self.tr("Select input and output paths"), parent=self)
            return

        self._worker = AnnotationExtractorWorker(
            self._input_path,
            self._output_dir,
            self.subdirs_switch.isChecked()
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.info.connect(lambda msg: InfoBar.info(self.tr("Info"), msg, parent=self))
        self._worker.error.connect(lambda msg: InfoBar.error(self.tr("Error"), msg, parent=self))

        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._worker.start()

    def _on_stop(self):
        if self._worker:
            self._worker.abort()

    def _on_progress(self, current, total):
        self.run_button.setText(f"{current}/{total}")

    def _on_finished(self, processed, failed):
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.run_button.setText(self.tr("Run"))
        InfoBar.success(self.tr("Success"), self.tr(f"Processed {processed} files"), parent=self)
        self._worker = None
