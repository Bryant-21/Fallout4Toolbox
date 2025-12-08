import os
import subprocess
from typing import Optional

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QWidget, QLabel, QFileDialog, QHBoxLayout, QTextEdit
from qfluentwidgets import PushSettingCard, PrimaryPushButton

from src.utils.dds_utils import load_image
from src.utils.filesystem_utils import get_app_root
from src.utils.helpers import BaseWidget
from src.utils.icons import CustomIcons
from src.utils.logging_utils import logger


class DDSInspector(BaseWidget):
    """Simple DDS inspector.

    Left side: textual information from texdiag.
    Right side: image preview (using the shared DDS loader).
    """

    def __init__(self, parent: Optional[QWidget] = None, text: str = "DDS Inspector"):
        super().__init__(parent=parent, text=text, vertical=True)

        self.dds_path: Optional[str] = None
        self.dds_image: Optional[Image.Image] = None

        # --- File selector card ---
        self.dds_card = PushSettingCard(
            self.tr("DDS Texture"),
            CustomIcons.IMAGE.icon(stroke=True),
            self.tr("Select a DDS texture to inspect"),
            self.tr("No DDS selected"),
        )
        self.dds_card.clicked.connect(self.on_select_dds)
        self.addToFrame(self.dds_card)

        # Optional explicit refresh button (re-run texdiag / reload preview)
        self.refresh_button = PrimaryPushButton(
            text=self.tr("Refresh Info"),
        )
        self.refresh_button.clicked.connect(self.refresh)

        # --- Main content: left info, right preview ---
        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Left: texdiag output
        self.info_view = QTextEdit(self)
        self.info_view.setReadOnly(True)
        self.info_view.setPlaceholderText(self.tr("texdiag info output will appear here"))

        # Right: image preview
        self.preview_label = QLabel(self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(400, 400)
        self.preview_label.setText(self.tr("DDS preview"))

        layout.addWidget(self.info_view, 2)
        layout.addWidget(self.preview_label, 3)

        self.addToFrame(container)
        self.addButtonBarToBottom(self.refresh_button)


    # -------------- Actions --------------
    def on_select_dds(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select DDS Texture"),
            "",
            self.tr("DDS Images (*.dds);;All Files (*.*)"),
        )
        if not path:
            return

        self.dds_path = path
        base = os.path.basename(path)
        self.dds_card.setContent(base)

        # Load both sides
        self.refresh()

    def refresh(self):
        if not self.dds_path:
            return

        self._update_texdiag_info()
        self._update_preview()

    # -------------- Helpers --------------
    def _texdiag_exe(self) -> str:
        """Return full path to texdiag.exe located in the resource folder.

        This mirrors how texconv is resolved via app root.
        """
        # texdiag.exe is shipped in the resource directory (next to texconv.exe)
        exe_path = os.path.join(get_app_root(), "resource", "texdiag.exe")
        return exe_path

    def _update_texdiag_info(self):
        texdiag_path = self._texdiag_exe()
        if not os.path.isfile(texdiag_path):
            msg = self.tr(f"texdiag.exe not found at: {texdiag_path}")
            logger.error(msg)
            self.info_view.setPlainText(msg)
            return

        try:
            cmd = [texdiag_path, "info", self.dds_path]
            logger.debug("Running texdiag: %s", " ".join(cmd))
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            msg = self.tr("texdiag timed out while reading DDS")
            logger.error(msg)
            self.info_view.setPlainText(msg)
            return
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("Failed to run texdiag: %s", e)
            self.info_view.setPlainText(self.tr(f"Failed to run texdiag: {e}"))
            return

        output_parts = []
        if result.stdout.strip():
            output_parts.append(result.stdout.strip())
        if result.stderr.strip():
            output_parts.append("\n[Errors/Warnings]\n" + result.stderr.strip())

        if output_parts:
            self.info_view.setPlainText("\n".join(output_parts))
        else:
            self.info_view.setPlainText(self.tr("texdiag produced no output."))

    def _update_preview(self):
        try:
            img = load_image(self.dds_path, f="RGBA")
        except Exception as e:
            logger.exception("Failed to load DDS for preview: %s", e)
            self.preview_label.setText(self.tr("Failed to load DDS preview"))
            self.preview_label.setPixmap(QPixmap())
            return

        self.dds_image = img
        self._set_preview_from_pil(img)

    def _set_preview_from_pil(self, pil_img: Image.Image):
        if pil_img is None:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText(self.tr("No image"))
            return

        # Convert PIL image to QImage/QPixmap similar to PaletteApplier
        rgba_img = pil_img.convert("RGBA")
        w, h = rgba_img.size
        data = rgba_img.tobytes("raw", "RGBA")
        qimage = QImage(data, w, h, QImage.Format_RGBA8888)

        max_w = max(1, self.preview_label.width())
        max_h = max(1, self.preview_label.height())
        pix = QPixmap.fromImage(qimage).scaled(
            max_w,
            max_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(pix)
        self.preview_label.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Rescale preview on resize
        if self.dds_image is not None:
            self._set_preview_from_pil(self.dds_image)

