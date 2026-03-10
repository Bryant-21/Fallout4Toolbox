import os
import subprocess
import time
from typing import Optional, Dict

from PySide6 import QtWidgets
from PySide6.QtCore import QThread, Signal, Qt
from qfluentwidgets import (
    PushSettingCard,
    InfoBar,
    FluentIcon as FIF,
    CardWidget,
    BodyLabel,
    CaptionLabel,
    TransparentToolButton,
    CheckBox,
    PlainTextEdit,
    SubtitleLabel,
)

from src.utils.appconfig import cfg
from src.utils.filesystem_utils import get_app_root
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger


def _is_xml(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".xml"


def _get_hkx_path(xml_path: str) -> str:
    return os.path.splitext(xml_path)[0] + ".hkx"


class HKXWatchWorker(QThread):
    """Worker that monitors a directory for XML changes and packs them if newer than HKX."""

    info = Signal(str)
    error = Signal(str)
    file_processed = Signal(str)
    file_removed = Signal(str)

    def __init__(self, watch_dir: str, include_subdirs: bool):
        super().__init__()
        self._abort = False
        self.watch_dir = watch_dir
        self.include_subdirs = include_subdirs
        self.app_root = get_app_root()
        self.jar_path = os.path.join(self.app_root, "resource", "hkxpack-cli.jar")
        self.last_mtimes: Dict[str, float] = {}
        self.known_xmls: set[str] = set()

    def abort(self):
        self._abort = True

    def run(self):
        if not os.path.exists(self.jar_path):
            self.error.emit("hkxpack-cli.jar not found in resource folder.")
            return

        self.info.emit(f"Started monitoring: {self.watch_dir}")
        
        while not self._abort:
            try:
                self._check_for_changes()
            except Exception as e:
                logger.exception("Error during HKX watch cycle")
                self.error.emit(f"Watch error: {e}")
            
            # Sleep for a bit to avoid hammering the disk
            for _ in range(20): # 2 seconds total sleep, checking abort frequently
                if self._abort:
                    break
                time.sleep(0.1)

        self.info.emit(f"Monitoring stopped: {self.watch_dir}")

    def _check_for_changes(self):
        if not os.path.isdir(self.watch_dir):
            return

        current_xmls = set()

        try:
            if self.include_subdirs:
                walker = os.walk(self.watch_dir)
            else:
                files = [f for f in os.listdir(self.watch_dir) if os.path.isfile(os.path.join(self.watch_dir, f))]
                walker = [(self.watch_dir, [], files)]

            for root, _dirs, files in walker:
                for f in files:
                    if self._abort:
                        return
                    if _is_xml(f):
                        xml_path = os.path.normpath(os.path.join(root, f))
                        current_xmls.add(xml_path)
                        hkx_path = _get_hkx_path(xml_path)
                        
                        try:
                            xml_mtime = os.path.getmtime(xml_path)
                            
                            # If we have seen this file before and it hasn't changed on disk, skip
                            if self.last_mtimes.get(xml_path) == xml_mtime:
                                continue
                            
                            self.last_mtimes[xml_path] = xml_mtime

                            if os.path.exists(hkx_path):
                                hkx_mtime = os.path.getmtime(hkx_path)
                                if xml_mtime > hkx_mtime:
                                    logger.debug("XML newer than HKX: %s", xml_path)
                                    self._pack_file(xml_path)
                                else:
                                    # Already up to date, but we record the mtime to avoid re-checking every 2s
                                    logger.debug("XML up to date: %s", xml_path)
                            else:
                                logger.debug("HKX missing, packing: %s", xml_path)
                                self._pack_file(xml_path)
                                
                        except OSError as e:
                            logger.warning("Could not check mtime for %s: %s", xml_path, e)
                            continue
        except Exception as e:
            logger.exception("Error scanning directory %s", self.watch_dir)
            self.error.emit(f"Scan error {self.watch_dir}: {e}")
        
        # Check for deleted XMLs
        deleted_xmls = self.known_xmls - current_xmls
        for deleted_xml in deleted_xmls:
            hkx_to_remove = _get_hkx_path(deleted_xml)
            if os.path.exists(hkx_to_remove):
                try:
                    os.remove(hkx_to_remove)
                    logger.info("Removed orphaned HKX: %s", hkx_to_remove)
                    self.file_removed.emit(os.path.basename(hkx_to_remove))
                except Exception as e:
                    logger.error("Failed to remove orphaned HKX %s: %s", hkx_to_remove, e)
            
            if deleted_xml in self.last_mtimes:
                del self.last_mtimes[deleted_xml]
        
        self.known_xmls = current_xmls

    def _pack_file(self, xml_path: str):
        cmd = ["java", "-jar", self.jar_path, "pack", xml_path]
        logger.info("Auto-packing: %s", xml_path)
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=False,
            )
            if result.returncode == 0:
                logger.info("Auto-packed successfully: %s", xml_path)
                self.file_processed.emit(os.path.basename(xml_path))
            else:
                logger.error("Auto-pack failed for %s: %s", xml_path, result.stdout)
                self.error.emit(f"Pack failed for {os.path.basename(xml_path)}")
        except Exception as e:
            logger.exception("Auto-pack crashed for %s", xml_path)
            self.error.emit(f"Auto-pack crash: {e}")


class WatchDirCard(CardWidget):
    """Card representing a single watched directory."""
    removed = Signal(str)
    subdirs_changed = Signal(str, bool)

    def __init__(self, path: str, include_subdirs: bool, parent=None):
        super().__init__(parent)
        self.path = path
        
        self.setFixedHeight(73)
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(20, 11, 11, 11)
        self.layout.setSpacing(15)

        self.iconWidget = QtWidgets.QLabel()
        self.iconWidget.setPixmap(CustomIcons.FOLDER_IMAGE.icon(stroke=True).pixmap(32, 32))
        self.layout.addWidget(self.iconWidget)

        self.textLayout = QtWidgets.QVBoxLayout()
        self.textLayout.setSpacing(0)
        self.textLayout.setContentsMargins(0, 0, 0, 0)
        
        self.pathLabel = BodyLabel(path, self)
        self.statusLabel = CaptionLabel(self.tr("Monitoring active"), self)
        self.statusLabel.setTextColor("#606060", "#d2d2d2")
        
        self.textLayout.addWidget(self.pathLabel, 0, Qt.AlignVCenter)
        self.textLayout.addWidget(self.statusLabel, 0, Qt.AlignVCenter)
        self.layout.addLayout(self.textLayout)

        self.layout.addStretch(1)

        self.subdirs_checkbox = CheckBox(self.tr("Subfolders"), self)
        self.subdirs_checkbox.setChecked(include_subdirs)
        self.subdirs_checkbox.stateChanged.connect(self._on_subdirs_changed)
        self.layout.addWidget(self.subdirs_checkbox)

        self.remove_button = TransparentToolButton(FIF.DELETE, self)
        self.remove_button.clicked.connect(lambda: self.removed.emit(self.path))
        self.layout.addWidget(self.remove_button)

    def _on_subdirs_changed(self, state):
        self.subdirs_changed.emit(self.path, self.subdirs_checkbox.isChecked())


class HKXWatchWidget(BaseWidget):
    """Widget to monitor multiple XML directories and auto-pack them when they change."""

    def __init__(self, parent: Optional[QtWidgets.QWidget], text: str = "HKX Auto-Packer"):
        super().__init__(parent=parent, text=text, vertical=True)

        # UI Components
        self.add_dir_card = PushSettingCard(
            self.tr("Add Folder to Watch"),
            FIF.ADD,
            self.tr("Select a new folder to monitor for XML changes"),
            self.tr("Click to select folder")
        )
        self.add_dir_card.clicked.connect(self._on_add_dir)
        self.addToFrame(self.add_dir_card)

        # Log Area
        self.log_group = QtWidgets.QFrame(self)
        self.log_layout = QtWidgets.QVBoxLayout(self.log_group)
        self.log_label = SubtitleLabel(self.tr("Recent Activity"), self.log_group)
        self.log_edit = PlainTextEdit(self.log_group)
        self.log_edit.setReadOnly(True)
        self.log_edit.setFixedHeight(150)
        self.log_layout.addWidget(self.log_label)
        self.log_layout.addWidget(self.log_edit)
        
        # Scroll area for the list of directories
        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scroll_area.setStyleSheet("background: transparent;")
        
        self.list_container = QtWidgets.QWidget()
        self.list_container.setStyleSheet("background: transparent;")
        self.list_layout = QtWidgets.QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(0, 10, 0, 0)
        self.list_layout.setSpacing(10)
        self.list_layout.addStretch(1)
        
        self.scroll_area.setWidget(self.list_container)
        self.boxLayout.addWidget(self.scroll_area)
        self.boxLayout.addWidget(self.log_group)

        # Initial load
        self._refresh_list()
        
        # Connect to existing workers if any
        self._sync_with_main()

    def _log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_edit.appendPlainText(f"[{timestamp}] {message}")
        # Scroll to bottom
        self.log_edit.verticalScrollBar().setValue(self.log_edit.verticalScrollBar().maximum())

    def _refresh_list(self):
        # Clear existing cards (except stretch)
        for i in reversed(range(self.list_layout.count())):
            item = self.list_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()

        # Add cards for each watched dir
        watch_dirs_raw = cfg.get(cfg.hkx_watch_dirs)
        for item in watch_dirs_raw:
            if "|" in item:
                path, sub = item.split("|", 1)
                include_sub = sub.lower() == "true"
                self._add_card_to_ui(path, include_sub)
        
        self.list_layout.addStretch(1)

    def _add_card_to_ui(self, path: str, include_sub: bool):
        card = WatchDirCard(path, include_sub, self)
        card.removed.connect(self._on_remove_dir)
        card.subdirs_changed.connect(self._on_subdirs_settings_changed)
        self.list_layout.insertWidget(self.list_layout.count() - 1, card)

    def _sync_with_main(self):
        from FalloutToolbox import FalloutToolboxMainWindow
        main_win = self.window()
        if isinstance(main_win, FalloutToolboxMainWindow):
            for path, worker in main_win.hkx_watchers.items():
                self._connect_worker(worker)

    def _connect_worker(self, worker: HKXWatchWorker):
        # Prevent multiple connections if possible, though Signal.connect is usually safe 
        # but might trigger multiple times if not careful. 
        # Better to connect once when starting.
        try:
            worker.info.disconnect(self._show_info)
            worker.error.disconnect(self._show_error)
            worker.file_processed.disconnect(self._on_file_processed)
            worker.file_removed.disconnect(self._on_file_removed)
        except:
            pass
            
        worker.info.connect(self._show_info)
        worker.error.connect(self._show_error)
        worker.file_processed.connect(self._on_file_processed)
        worker.file_removed.connect(self._on_file_removed)

    def _on_add_dir(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select folder to watch"), os.getcwd()
        )
        if not directory:
            return

        directory = os.path.normpath(directory)
        watch_dirs = cfg.get(cfg.hkx_watch_dirs)
        
        # Check if already added
        if any(d.split("|")[0] == directory for d in watch_dirs):
            InfoBar.warning(self.tr("Already Watching"), self.tr("This folder is already in the watch list."), parent=self)
            return

        # Update Config
        watch_dirs.append(f"{directory}|False")
        cfg.set(cfg.hkx_watch_dirs, watch_dirs)
        cfg.set(cfg.hkx_watch_enabled, True) # Ensure enabled if we add one

        # Add to UI
        self._add_card_to_ui(directory, False)

        # Start Worker
        from FalloutToolbox import FalloutToolboxMainWindow
        main_win = self.window()
        if isinstance(main_win, FalloutToolboxMainWindow):
            main_win.start_hkx_watcher(directory, False)
            worker = main_win.hkx_watchers.get(directory)
            if worker:
                self._connect_worker(worker)

    def _on_remove_dir(self, path: str):
        # Update Config
        watch_dirs = cfg.get(cfg.hkx_watch_dirs)
        watch_dirs = [d for d in watch_dirs if d.split("|")[0] != path]
        cfg.set(cfg.hkx_watch_dirs, watch_dirs)
        
        if not watch_dirs:
            cfg.set(cfg.hkx_watch_enabled, False)

        # Stop Worker
        from FalloutToolbox import FalloutToolboxMainWindow
        main_win = self.window()
        if isinstance(main_win, FalloutToolboxMainWindow):
            main_win.stop_hkx_watcher(path)

        # Update UI
        self._refresh_list()

    def _on_subdirs_settings_changed(self, path: str, include_sub: bool):
        # Update Config
        watch_dirs = cfg.get(cfg.hkx_watch_dirs)
        new_watch_dirs = []
        for d in watch_dirs:
            p, _ = d.split("|", 1)
            if p == path:
                new_watch_dirs.append(f"{path}|{include_sub}")
            else:
                new_watch_dirs.append(d)
        cfg.set(cfg.hkx_watch_dirs, new_watch_dirs)

        # Restart Worker with new settings
        from FalloutToolbox import FalloutToolboxMainWindow
        main_win = self.window()
        if isinstance(main_win, FalloutToolboxMainWindow):
            main_win.start_hkx_watcher(path, include_sub)
            worker = main_win.hkx_watchers.get(path)
            if worker:
                self._connect_worker(worker)

    def _show_info(self, text: str):
        self._log(f"INFO: {text}")
        InfoBar.info(self.tr("Watcher"), text, parent=self)

    def _show_error(self, text: str):
        self._log(f"ERROR: {text}")
        InfoBar.error(self.tr("Watcher Error"), text, parent=self)

    def _on_file_processed(self, filename: str):
        self._log(f"PACKED: {filename}")
        InfoBar.success(self.tr("Auto-Packed"), self.tr(f"Packed {filename} because it was modified."), parent=self, duration=2000)

    def _on_file_removed(self, filename: str):
        self._log(f"REMOVED: {filename}")
        InfoBar.warning(self.tr("Auto-Removed"), self.tr(f"Removed {filename} because source XML no longer exists."), parent=self, duration=2000)
