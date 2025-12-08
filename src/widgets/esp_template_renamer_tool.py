import os
import struct
from typing import Tuple, List

from PySide6 import QtWidgets
from PySide6.QtWidgets import QWidget, QFileDialog
from qfluentwidgets import FluentIcon as FIF, PrimaryPushButton, PushSettingCard, ConfigItem

from src.help.template_help import TemplateHelp
from src.settings.template_settings import TemplateSettings
from src.utils.appconfig import cfg
from src.utils.cards import TextSettingCard
from src.utils.helpers import BaseWidget
from src.utils.logging_utils import logger
from src.utils.icons import CustomIcons

# Minimalistic ESP EDID replacer for Fallout 4/Skyrim-style ESPs based on xEdit source.
# This does NOT fully parse the file. It scans for subrecords named 'EDID',
# and for TES4 header subrecords 'CNAM' (Author) and 'SNAM' (Description).
# It keeps binary sizes intact by only allowing replacements that do not
# exceed the original allocated subrecord length. If the new text is shorter,
# it will be null-padded.
#
# Limitations:
# - Will not update container sizes. Therefore, new strings must be <= original length.
# - Does not handle compressed records (record flag 0x00040000). Such records' subrecords
#   are inside zlib-compressed data, which this tool does not decode. Most EDIDs of interest
#   are often uncompressed, but results may vary by plugin.
# - Only EDID, CNAM (author), SNAM (description) are modified.

EDID = b"EDID"
CNAM = b"CNAM"
SNAM = b"SNAM"
TES4 = b"TES4"

class ESPProcessor:
    def __init__(self, data: bytearray):
        self.data = data
        self.changed_edids = 0
        self.changed_author = False
        self.changed_description = False

    @staticmethod
    def _read_u16_le(b: bytes, off: int) -> int:
        return struct.unpack_from('<H', b, off)[0]

    @staticmethod
    def _read_u32_le(b: bytes, off: int) -> int:
        return struct.unpack_from('<I', b, off)[0]

    def _replace_in_subrecord(self, off: int, tag: bytes, new_text: str) -> bool:
        # subrecord layout: 4s tag, uint16 size, payload[size]
        # off points to tag start
        # returns True if modified
        if off + 6 > len(self.data):
            return False
        size = self._read_u16_le(self.data, off + 4)
        start = off + 6
        end = start + size
        if end > len(self.data):
            return False
        payload = bytes(self.data[start:end])
        # Decode as ASCII/UTF-8 tolerant; EDID/CNAM/SNAM typically ASCII, zero-terminated within size
        # Keep existing zero padding
        try:
            # Strip trailing zeros for meaningful text
            text = payload.rstrip(b"\x00").decode('utf-8', errors='ignore')
        except Exception:
            return False
        # Build new payload with null padding to exact length
        new_bytes = new_text.encode('utf-8')
        if len(new_bytes) > size:
            return False  # cannot expand
        padded = new_bytes + b"\x00" * (size - len(new_bytes))
        if padded == payload:
            return False
        self.data[start:end] = padded
        return True

    def replace_edid_occurrences(self, old_base: str, new_base: str):
        # we replace inside EDID subrecord texts: occurrences of old_base -> new_base
        # Not exceeding the allocated subrecord size
        idx = 0
        while True:
            idx = self.data.find(EDID, idx)
            if idx == -1:
                break
            if idx + 6 > len(self.data):
                break
            size = self._read_u16_le(self.data, idx + 4)
            start = idx + 6
            end = start + size
            if end > len(self.data):
                break
            payload = bytes(self.data[start:end])
            try:
                text = payload.rstrip(b"\x00").decode('utf-8', errors='ignore')
            except Exception:
                idx += 4
                continue
            if old_base in text:
                replaced = text.replace(old_base, new_base)
                new_bytes = replaced.encode('utf-8')
                if len(new_bytes) <= size:
                    padded = new_bytes + b"\x00" * (size - len(new_bytes))
                    self.data[start:end] = padded
                    self.changed_edids += 1
            idx = end  # move past this subrecord

    def replace_terms_with_base_in_edids(self, terms: List[str], new_base: str):
        # Replace occurrences of any of the given terms with new_base inside EDID texts.
        # Keeps sizes by null-padding and only commits if the result fits.
        if not terms:
            return
        # keep order, remove empties
        terms = [t for t in terms if t]
        if not terms:
            return
        idx = 0
        while True:
            idx = self.data.find(EDID, idx)
            if idx == -1:
                break
            if idx + 6 > len(self.data):
                break
            size = self._read_u16_le(self.data, idx + 4)
            start = idx + 6
            end = start + size
            if end > len(self.data):
                break
            payload = bytes(self.data[start:end])
            try:
                text = payload.rstrip(b"\x00").decode('utf-8', errors='ignore')
            except Exception:
                idx += 4
                continue
            new_text = text
            for term in terms:
                if term and term in new_text:
                    new_text = new_text.replace(term, new_base)
            if new_text != text:
                new_bytes = new_text.encode('utf-8')
                if len(new_bytes) <= size:
                    padded = new_bytes + b"\x00" * (size - len(new_bytes))
                    self.data[start:end] = padded
                    self.changed_edids += 1
            idx = end

    def set_author(self, author: str):
        # set CNAM in TES4 header if present and sufficient size
        # We'll first locate TES4 record at start of file (usually the first record)
        tes4_pos = self.data.find(TES4)
        if tes4_pos == -1:
            return
        idx = tes4_pos
        # Scan forward within a reasonable window (first few KB) for CNAM subrecord
        window_end = min(len(self.data), idx + 20000)
        search_pos = idx
        while True:
            search_pos = self.data.find(CNAM, search_pos, window_end)
            if search_pos == -1:
                break
            if self._replace_in_subrecord(search_pos, CNAM, author):
                self.changed_author = True
                break
            search_pos += 4

    def set_description(self, description: str):
        tes4_pos = self.data.find(TES4)
        if tes4_pos == -1:
            return
        idx = tes4_pos
        window_end = min(len(self.data), idx + 20000)
        search_pos = idx
        while True:
            search_pos = self.data.find(SNAM, search_pos, window_end)
            if search_pos == -1:
                break
            if self._replace_in_subrecord(search_pos, SNAM, description):
                self.changed_description = True
                break
            search_pos += 4


def process_esp(source_path: str, dest_path: str, new_basename: str, author: str, description: str, match_terms: List[str]) -> Tuple[int, bool, bool, str]:
    # returns: (num_edid_changed, author_changed, description_changed, message)
    if not os.path.isfile(source_path):
        return (0, False, False, f"Source file not found: {source_path}")

    with open(source_path, 'rb') as f:
        data = bytearray(f.read())

    new_basename_clean = os.path.splitext(new_basename)[0]

    # sanitize match terms: support comma and newline separated
    sanitized_terms: List[str] = []
    for term in (match_terms or []):
        # split by comma within each line if present
        parts = [p.strip() for p in term.split(',')]
        for p in parts:
            if p and p not in sanitized_terms:
                sanitized_terms.append(p)

    proc = ESPProcessor(data)
    if sanitized_terms and new_basename_clean:
        proc.replace_terms_with_base_in_edids(sanitized_terms, new_basename_clean)
    # CNAM/SNAM as requested
    if author:
        proc.set_author(author)
    if description:
        proc.set_description(description)

    # Write output
    os.makedirs(os.path.dirname(dest_path) or '.', exist_ok=True)
    with open(dest_path, 'wb') as f:
        f.write(proc.data)

    return (proc.changed_edids, proc.changed_author, proc.changed_description, "OK")


class ESPTemplaterWidget(BaseWidget):
    def __init__(self, parent=None, text=None):
        super().__init__(parent=parent, text=text, vertical=True)

        # Shared logger from toolbox
        self._logger = None

        # Persistent settings using qfluentwidgets ConfigItem
        self.src_cfg = ConfigItem("esp_renamer", "src", "")
        self.new_name_cfg = ConfigItem("esp_renamer", "new_name", "")

        # Setting cards
        self.src_card = PushSettingCard(
            self.tr("Template ESP"),
            CustomIcons.FILE.icon(stroke=True),
            self.tr("Select ESP Template"),
            self.src_cfg.value or ""
        )

        self.new_name_card = TextSettingCard(
            self.new_name_cfg,
            FIF.EDIT,
            self.tr("New plugin name (eg: B21_PepperShaker.esp)"),
            self.new_name_cfg.value or ""
        )
        self.author_card = TextSettingCard(
            cfg.author_cfg,
            FIF.ACCEPT,
            self.tr("Author (optional)"),
            cfg.author_cfg.value or ""
        )
        self.description_card = TextSettingCard(
            cfg.description_cfg,
            FIF.INFO,
            self.tr("Description (optional)"),
            cfg.description_cfg.value or ""
        )
        self.match_card = TextSettingCard(
            cfg.match_text_cfg,
            FIF.FILTER,
            self.tr("Replace Strings (comma separated). Defaults to source filename. Example: B21_Template_Gun -> B21_PepperShaker"),
            cfg.match_text_cfg.value or ""
        )

        self.run_btn = PrimaryPushButton(icon=FIF.RIGHT_ARROW, text=self.tr("Copy and Replace Template ESP"))

        self.addToFrame(self.src_card)
        self.addToFrame(self.new_name_card)
        self.addToFrame(self.author_card)
        self.addToFrame(self.description_card)
        self.addToFrame(self.match_card)
        self.boxLayout.addStretch(1)

        # Wire up
        self.src_card.clicked.connect(self.on_src_card)
        self.run_btn.clicked.connect(self.run)

        self.settings_widget = TemplateSettings(self)
        self.settings_drawer.addWidget(self.settings_widget)

        self.addButtonBarToBottom(self.run_btn)

        self.help_widget = TemplateHelp(self)
        self.help_drawer.addWidget(self.help_widget)


    def on_src_card(self):
        # Start from current config value; do not fall back to cwd
        current = (self.src_cfg.value or '').strip()
        start_dir = os.path.dirname(current) if current else ''
        path, _ = QFileDialog.getOpenFileName(self, self.tr('Select ESP file'), start_dir, 'Bethesda Plugin (*.esp)')
        if path:
            self.src_cfg.value = path
            self.src_card.setContent(path)
            base = os.path.splitext(os.path.basename(path))[0]
            # If new name empty, default to basename
            if not (self.new_name_cfg.value or '').strip():
                self.new_name_cfg.value = base
                self.new_name_card.setValue(base)
            # Default match list to the source basename if empty
            if not (cfg.match_text_cfg.value or '').strip():
                cfg.set(cfg.match_text_cfg,base)
                self.match_card.setValue(base)

    def run(self):
        src = (self.src_cfg.value or '').strip()
        new_name = (self.new_name_cfg.value or '').strip()
        author = (cfg.author_cfg.value or '').strip()
        desc = (cfg.description_cfg.value or '').strip()
        raw = (cfg.match_text_cfg.value or '')

        if not src:
            logger.debug('Please select a source ESP file.')
            return
        if not new_name:
            logger.debug('Please enter the new plugin name.')
            return
        if not os.path.isfile(src):
            logger.debug('Source file does not exist.')
            return

        new_base = os.path.splitext(os.path.basename(new_name))[0]

        # Build match terms (comma or newline separated)
        terms: List[str] = []
        for line in raw.splitlines() if isinstance(raw, str) else []:
            for part in line.split(','):
                p = part.strip()
                if p and p not in terms:
                    terms.append(p)
        # If no terms specified, default to source basename
        if not terms:
            base = os.path.splitext(os.path.basename(src))[0]
            if base:
                terms.append(base)

        dest_dir = os.path.dirname(src)
        dest_path = os.path.join(dest_dir, f"{new_base}.esp")
        if os.path.abspath(dest_path) == os.path.abspath(src):
            logger.debug('Error: Destination would overwrite the source file. Choose a different new name.')
            return

        # persist before processing - ConfigItem stores automatically on set
        self.src_cfg.value = src
        self.new_name_cfg.value = new_base

        changed_edids, changed_author, changed_desc, msg = process_esp(src, dest_path, new_base, author, desc, terms)
        if msg != 'OK':
            logger.debug(f'Failed: {msg}')
            return

        logger.debug(f'Created: {dest_path}')
        logger.debug(f'EDID entries changed: {changed_edids}')
        logger.debug(f'Match terms applied: {len(terms)}')
        if author:
            logger.debug(f'Author set: {"Yes" if changed_author else "No (CNAM not found or too short)"}')
        if desc:
            logger.debug(f'Description set: {"Yes" if changed_desc else "No (SNAM not found or too short)"}')

