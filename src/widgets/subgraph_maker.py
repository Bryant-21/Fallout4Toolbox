import os
from typing import List, Tuple, Optional

from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QDialog, QVBoxLayout as QVBoxLayoutQt, QLabel, QDialogButtonBox
from qfluentwidgets import (
    FluentIcon as FIF,
    PrimaryPushButton,
    SwitchSettingCard,
    ConfigItem,
)

from help.subgraph_help import SubgraphHelp
from settings.basic_settings import BasicSettings
from src.utils.cards import TextSettingCard
from src.utils.helpers import BaseWidget
from src.utils.logging_utils import logger
from src.utils.appconfig import cfg
from utils.filesystem_utils import get_app_root
from utils.icons import CustomIcons

RESOURCE_FILES = {
    'Human': 'SubGraphData_HumanRaceSubGraphData.txt',
    'PowerArmor': 'SubGraphData_PowerArmorRace.txt',
    'SuperMutant': 'SubGraphData_SuperMutantRace.txt',
}


class SubGraphWorker(QtCore.QThread):
    progress = QtCore.Signal(int, int, str)  # processed, total, message
    finished = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(self,
                 target_anim: str,
                 new_anim: str,
                 target_folder: str,
                 new_folder: str,
                 prepend_mode: bool,
                 races: List[str],
                 parent=None):
        super().__init__(parent)
        self.target_anim = (target_anim or '').strip()
        self.new_anim = (new_anim or '').strip()
        self.target_folder = (target_folder or '').strip()
        self.new_folder = (new_folder or '').strip()
        self.prepend_mode = bool(prepend_mode)
        self.races = list(races or [])
        self._abort = False
        # Per-file results: list of tuples (input_basefile, out_name, count)
        self.results: List[Tuple[str, str, int]] = []

    def abort(self):
        self._abort = True

    # --- Helpers ---
    @staticmethod
    def _split_csv_preserve(items_csv: str) -> List[str]:
        # Split by comma, trimming whitespace around tokens, preserving inherent order
        if items_csv is None:
            return []
        parts = [p.strip() for p in items_csv.split(',')]
        return [p for p in parts if p != '']

    @staticmethod
    def _join_csv(items: List[str]) -> str:
        # Standardize with comma+space between items
        return ', '.join(items)

    def _process_line(self, line: str) -> Tuple[str, bool]:
        # Returns (possibly modified line, matched)
        # Expect TSV with the 4th field = animations CSV, 6th field = folders CSV
        try:
            if not line.strip():
                return line, False
            # Keep trailing newline if present to re-attach later outside
            newline = ''
            if line.endswith('\r\n'):
                newline = '\r\n'
                core = line[:-2]
            elif line.endswith('\n'):
                newline = '\n'
                core = line[:-1]
            else:
                core = line

            cols = core.split('\t')
            # Require at least 6 columns
            if len(cols) < 6:
                return line, False

            # Field indexes
            # 0-based: animations @ 3, folders @ 5
            anim_csv = cols[3]
            folders_csv = cols[5]

            # Check match conditions (support multiple targets via comma-separated values)
            anim_tokens = self._split_csv_preserve(anim_csv)
            anim_targets = self._split_csv_preserve(self.target_anim)
            # If no targets provided, nothing to do
            if not anim_targets:
                return line, False
            has_anim = any(tok in anim_targets for tok in anim_tokens)
            if not has_anim:
                return line, False

            folder_tokens = self._split_csv_preserve(folders_csv)
            folder_targets = self._split_csv_preserve(self.target_folder)
            match_indices: List[int] = []
            for i, entry in enumerate(folder_tokens):
                last_seg = entry.replace('/', '\\').split('\\')[-1]
                if last_seg in folder_targets:
                    match_indices.append(i)

            if not match_indices:
                return line, False

            # Replace animation token(s): any token that equals one of the targets becomes new_anim
            new_anim_tokens = [self.new_anim if tok in anim_targets else tok for tok in anim_tokens]
            cols[3] = self._join_csv(new_anim_tokens)

            # Depending on mode, either prepend new entries before matches or replace matches
            if self.prepend_mode:
                # For each matching folder entry, insert a new entry before it with the same parent path
                # but with last segment replaced by new_folder
                # We'll iterate indices in increasing order but account for growth by offset
                offset = 0
                for idx in match_indices:
                    real_idx = idx + offset
                    entry = folder_tokens[real_idx]
                    parts = entry.replace('/', '\\').split('\\')
                    parts[-1] = self.new_folder
                    new_entry = '\\'.join(parts)
                    # Avoid immediate duplicate at insertion point
                    if real_idx - 1 >= 0 and folder_tokens[real_idx - 1] == new_entry:
                        pass
                    else:
                        folder_tokens.insert(real_idx, new_entry)
                        offset += 1
            else:
                # Replace matching folder entries in-place with new_folder path under the same parent
                for idx in match_indices:
                    entry = folder_tokens[idx]
                    parts = entry.replace('/', '\\').split('\\')
                    parts[-1] = self.new_folder
                    new_entry = '\\'.join(parts)
                    folder_tokens[idx] = new_entry

            cols[5] = self._join_csv(folder_tokens)

            new_line = '\t'.join(cols) + newline
            return new_line, True
        except Exception as e:
            # On parsing error, do not treat as match; return original line
            logger.debug(f"SubGraphWorker line error: {e}")
            return line, False

    def run(self):
        try:
            selected = [r for r in self.races if r in RESOURCE_FILES]
            if not selected:
                self.finished.emit('No races selected.')
                return

            # Build file list
            in_files = [os.path.join(get_app_root(), 'resource', RESOURCE_FILES[r]) for r in selected]
            total = len(in_files)
            processed = 0

            # Determine output directory under project root: <root>/output/subgraph
            try:
                root_dir = os.path.abspath(os.path.join(self.resource_dir, os.pardir))
            except Exception:
                root_dir = os.getcwd()
            out_dir = os.path.join(root_dir, 'output', 'subgraph')
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception:
                # Fallback to resource directory if cannot create output dir
                out_dir = self.resource_dir
            # Expose for UI summary
            self.output_dir = out_dir

            for in_path in in_files:
                if self._abort:
                    break
                basefile = os.path.basename(in_path)
                name_no_ext, _ = os.path.splitext(basefile)
                # Build output filename using the NEW folder label per requirement
                nf_label = (self.new_folder or '').strip().replace(',', '+').replace(' ', '')
                if not nf_label:
                    nf_label = 'new'
                mode_label = 'additive' if self.prepend_mode else 'replace'
                out_name = f"{name_no_ext}_{nf_label}_{mode_label}.txt"
                out_path = os.path.join(out_dir, out_name)

                if not os.path.isfile(in_path):
                    self.progress.emit(processed, total, f"Missing: {basefile}")
                    # Record zero entries for missing file with expected output name
                    try:
                        self.results.append((basefile, out_name, 0))
                    except Exception:
                        pass
                    processed += 1
                    continue

                # Read, transform matching rows, and write only those rows
                matches: List[str] = []
                header_line: str | None = None
                try:
                    with open(in_path, 'r', encoding='utf-8') as f:
                        for idx, raw in enumerate(f):
                            if self._abort:
                                break
                            if idx == 0:
                                header_line = raw
                                continue  # don't process header through matcher
                            new_line, matched = self._process_line(raw)
                            if matched:
                                matches.append(new_line)
                except UnicodeDecodeError:
                    # Fallback to locale default if needed
                    with open(in_path, 'r', encoding='cp1252', errors='ignore') as f:
                        for idx, raw in enumerate(f):
                            if self._abort:
                                break
                            if idx == 0:
                                header_line = raw
                                continue
                            new_line, matched = self._process_line(raw)
                            if matched:
                                matches.append(new_line)

                if self._abort:
                    break

                try:
                    # Only write if we have matches; overwrite existing
                    if matches:
                        with open(out_path, 'w', encoding='utf-8', newline='') as out_f:
                            # Write original header first
                            if header_line is not None:
                                out_f.write(header_line if header_line.endswith('\n') or header_line.endswith('\r\n') else header_line + '\n')
                            # Then write all matched and transformed rows
                            for m in matches:
                                out_f.write(m if m.endswith('\n') or m.endswith('\r\n') else m + '\n')
                        self.progress.emit(processed + 1, total, f"Wrote: {os.path.basename(out_path)} ({len(matches)} rows)")
                        # Record results
                        try:
                            self.results.append((basefile, out_name, len(matches)))
                        except Exception:
                            pass
                    else:
                        self.progress.emit(processed + 1, total, f"No matching rows in {basefile}")
                        # Record zero results (no file written)
                        try:
                            self.results.append((basefile, out_name, 0))
                        except Exception:
                            pass
                except Exception as e:
                    self.progress.emit(processed + 1, total, f"ERROR writing {out_name}: {e}")
                    # Record error case with zero (unknown) to still list item
                    try:
                        self.results.append((basefile, out_name, 0))
                    except Exception:
                        pass

                processed += 1

            if self._abort:
                self.finished.emit('Aborted by user.')
            else:
                self.finished.emit('Done.')
        except Exception as e:
            self.error.emit(str(e))


class SubGraphMakerWindow(BaseWidget):
    def __init__(self, parent=None, text=None):
        super().__init__(parent=parent, text=text, vertical=True)

        # Config items to persist user inputs
        self.cfg_new_anim = ConfigItem("subgraph_maker", "new_anim", "")
        self.cfg_new_folder = ConfigItem("subgraph_maker", "new_folder", "")
        self.worker = None

        # Text inputs
        self.target_anim_card = TextSettingCard(
            cfg.cfg_target_anim,
            FIF.EDIT,
            self.tr("Target Animation Keyword"),
            self.tr("e.g., AnimsCryolator")
        )
        self.new_anim_card = TextSettingCard(
            self.cfg_new_anim,
            FIF.EDIT,
            self.tr("My Animation Keyword"),
            self.tr("e.g., AnimsTest")
        )
        self.target_folder_card = TextSettingCard(
            cfg.cfg_target_folder,
            FIF.FOLDER,
            self.tr("Target Animations Path(s) Command Seperated"),
            self.tr("The ones we want to replace e.g., Cryolater,Cryolater (Beth bad at spelling)")
        )

        self.new_folder_card = TextSettingCard(
            self.cfg_new_folder,
            FIF.FOLDER,
            self.tr("My Mod Animation Path"),
            self.tr("Where are your Anims located? e.g., ../B21_PepperShaker/test.hkx")
        )

        self.prepend_card = SwitchSettingCard(configItem=cfg.cfg_target_prepend, icon=CustomIcons.PADDING.icon(), title=self.tr("Add Animations Path (On) or Replace (Off)"), content=self.tr("Should we keep the existing Animation paths (On) or Replace them (Off)"))

        # Toggles
        self.human_card = SwitchSettingCard(configItem=cfg.cfg_human, icon=CustomIcons.PERSON_WALKING.icon(), title=self.tr("Human"), content=self.tr("Process SubGraphData_HumanRaceSubGraphData.txt"))
        self.power_card = SwitchSettingCard(configItem=cfg.cfg_power, icon=CustomIcons.SHIELD.icon(stroke=True), title=self.tr("PowerArmor"), content=self.tr("Process SubGraphData_PowerArmorRace.txt"))
        self.mutant_card = SwitchSettingCard(configItem=cfg.cfg_mutant, icon=CustomIcons.MUTANT.icon(), title=self.tr("SuperMutant"), content=self.tr("Process SubGraphData_SuperMutantRace.txt"))

        # Layout
        self.addToFrame(self.target_anim_card)
        self.addToFrame(self.new_anim_card)
        self.addToFrame(self.target_folder_card)
        self.addToFrame(self.new_folder_card)
        self.addToFrame(self.prepend_card)

        self.addToFrame(self.human_card)
        self.addToFrame(self.power_card)
        self.addToFrame(self.mutant_card)

        self.boxLayout.addStretch(1)

        # Bottom controls
        self.run_button = PrimaryPushButton(icon=FIF.RIGHT_ARROW, text=self.tr("Run"))

        self.run_button.clicked.connect(self.on_run)
        self.addButtonBarToBottom(self.run_button)

        self.settings_widget = BasicSettings(self)
        self.settings_drawer.addWidget(self.settings_widget)

        self.help_widget = SubgraphHelp(self)
        self.help_drawer.addWidget(self.help_widget)

        # Runtime context
        self._run_ctx: dict = {}

    # --- Dialog helper copied/adapted from dds_resizer ---
    def _show_info_dialog(self, title: str, lines: list[str]):
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        layout = QVBoxLayoutQt(dlg)
        lbl = QLabel("<br/>".join([QtWidgets.QApplication.translate("SubGraphMakerWindow", l) if isinstance(l, str) else str(l) for l in lines]))
        lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        layout.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        dlg.exec()

    @QtCore.Slot()
    def on_run(self):
        target_anim = (cfg.cfg_target_anim.value or '').strip()
        new_anim = (self.cfg_new_anim.value or '').strip()
        target_folder = (cfg.cfg_target_folder.value or '').strip()
        new_folder = (self.cfg_new_folder.value or '').strip()

        races: List[str] = []
        if bool(cfg.cfg_human.value):
            races.append('Human')
        if bool(cfg.cfg_power.value):
            races.append('PowerArmor')
        if bool(cfg.cfg_mutant.value):
            races.append('SuperMutant')

        # Validate
        if not races:
            QtWidgets.QMessageBox.warning(self, "Validation", "Please select at least one race to process.")
            return
        if not target_anim:
            QtWidgets.QMessageBox.warning(self, "Validation", "Please enter Target Animation Keyword.")
            return
        if not new_anim:
            QtWidgets.QMessageBox.warning(self, "Validation", "Please enter New Keyword Animation.")
            return
        if not target_folder:
            QtWidgets.QMessageBox.warning(self, "Validation", "Please enter Target Folder.")
            return
        if not new_folder:
            QtWidgets.QMessageBox.warning(self, "Validation", "Please enter New Folder.")
            return

        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'show_progress'):
            try:
                p.show_progress()
            except Exception:
                pass

        prepend_mode = bool(cfg.cfg_target_prepend.value)
        self.worker = SubGraphWorker(
            target_anim=target_anim,
            new_anim=new_anim,
            target_folder=target_folder,
            new_folder=new_folder,
            prepend_mode=prepend_mode,
            races=races,
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    @QtCore.Slot(int, int, str)
    def on_progress(self, processed: int, total: int, message: str):
        if total:
            p = getattr(self, 'parent', None)
            if p and hasattr(p, 'update_progress'):
                try:
                    percent = int(max(0, min(100, round((processed / total) * 100))))
                    p.update_progress(percent)
                except Exception:
                    pass
        logger.debug(message)

    @QtCore.Slot(str)
    def on_finished(self, message: str):
        lines = [
            f"<b>Status: {message}</b>",
        ]
        # Summarize results with counts per file
        try:
            results = getattr(self.worker, 'results', []) if self.worker else []
            if results:
                lines.append("<b>Per-file entries created:</b>")
                for basefile, out_name, count in results:
                    suffix = " (no file written)" if count == 0 else ""
                    lines.append(f"{out_name}: <b>{count}</b> entries{suffix}")
            else:
                lines.append("No files were processed.")
        except Exception as e:
            lines.append(f"Could not build results summary: {e}")
        try:
            out_dir = getattr(self.worker, 'output_dir', None)
        except Exception:
            out_dir = None
        if not out_dir:
            try:
                out_dir = self.worker.resource_dir
            except Exception:
                out_dir = 'output'
        lines.append(f"Outputs are saved in: {out_dir}")
        self._show_info_dialog(self.tr("SubGraph Processing Completed"), lines)
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'complete_loader'):
            try:
                p.complete_loader()
            except Exception:
                pass

    @QtCore.Slot(str)
    def on_error(self, message: str):
        lines = [
            f"<b>{self.tr('An error occurred')}</b>",
            f"Message: {message}",
        ]
        self._show_info_dialog(self.tr("SubGraph Processing Error"), lines)
        p = getattr(self, 'parent', None)
        if p and hasattr(p, 'complete_loader'):
            try:
                p.complete_loader()
            except Exception:
                pass
