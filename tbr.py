#!/usr/bin/env python3
"""
TBR to PBR Metal/Roughness Converter
Converts Fallout 4 TBR specular maps to Substance Painter PBR values
"""

import sys

from PySide6.QtCore import Qt, QRegularExpression
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QLabel, QSpinBox, QGroupBox, QGridLayout, QLineEdit)


class TBRConverter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TBR to PBR Metal/Roughness Converter")
        self.setMinimumWidth(450)
        self._updating = False

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Title
        title = QLabel("Fallout 4 TBR Specular Map Converter")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        # Info
        info = QLabel(
            "TBR Format: RED = Specular Red channel | GREEN = Glossiness | BLUE = Unused\n"
            "Shows what Substance Painter Metallic/Roughness values produce these TBR values.\n"
            "Export preset: Specular Red -> TBR Red, Glossiness -> TBR Green"
        )
        info.setWordWrap(True)
        main_layout.addWidget(info)

        hex_validator = QRegularExpressionValidator(QRegularExpression("[0-9A-Fa-f]{0,6}"))

        # TBR input
        tbr_group = QGroupBox("TBR Texture Values (from existing texture)")
        tbr_layout = QGridLayout()

        tbr_layout.addWidget(QLabel("Hex (RG only):"), 0, 0)
        self.tbr_hex = QLineEdit()
        self.tbr_hex.setMaxLength(6)
        self.tbr_hex.setPlaceholderText("e.g. 3B6F00")
        self.tbr_hex.setValidator(hex_validator)
        self.tbr_hex.editingFinished.connect(self.on_tbr_hex_changed)
        tbr_layout.addWidget(self.tbr_hex, 0, 1)

        tbr_layout.addWidget(QLabel("RED (Specular):"), 1, 0)
        self.red_spin = QSpinBox()
        self.red_spin.setRange(0, 255)
        self.red_spin.setValue(59)
        self.red_spin.valueChanged.connect(self.on_tbr_rgb_changed)
        tbr_layout.addWidget(self.red_spin, 1, 1)

        tbr_layout.addWidget(QLabel("GREEN (Glossiness):"), 2, 0)
        self.green_spin = QSpinBox()
        self.green_spin.setRange(0, 255)
        self.green_spin.setValue(111)
        self.green_spin.valueChanged.connect(self.on_tbr_rgb_changed)
        tbr_layout.addWidget(self.green_spin, 2, 1)

        self.tbr_preview = QLabel()
        self.tbr_preview.setMinimumSize(60, 60)
        tbr_layout.addWidget(self.tbr_preview, 0, 2, 3, 1)

        tbr_group.setLayout(tbr_layout)
        main_layout.addWidget(tbr_group)

        # Base color input
        base_group = QGroupBox("Base Color / Albedo (from diffuse texture)")
        base_layout = QGridLayout()

        base_layout.addWidget(QLabel("Hex:"), 0, 0)
        self.base_hex = QLineEdit()
        self.base_hex.setMaxLength(6)
        self.base_hex.setPlaceholderText("e.g. 7B352B")
        self.base_hex.setValidator(hex_validator)
        self.base_hex.editingFinished.connect(self.on_base_hex_changed)
        base_layout.addWidget(self.base_hex, 0, 1)

        base_layout.addWidget(QLabel("RED:"), 1, 0)
        self.base_r_spin = QSpinBox()
        self.base_r_spin.setRange(0, 255)
        self.base_r_spin.setValue(123)
        self.base_r_spin.valueChanged.connect(self.on_base_rgb_changed)
        base_layout.addWidget(self.base_r_spin, 1, 1)

        base_layout.addWidget(QLabel("GREEN:"), 2, 0)
        self.base_g_spin = QSpinBox()
        self.base_g_spin.setRange(0, 255)
        self.base_g_spin.setValue(53)
        self.base_g_spin.valueChanged.connect(self.on_base_rgb_changed)
        base_layout.addWidget(self.base_g_spin, 2, 1)

        base_layout.addWidget(QLabel("BLUE:"), 3, 0)
        self.base_b_spin = QSpinBox()
        self.base_b_spin.setRange(0, 255)
        self.base_b_spin.setValue(43)
        self.base_b_spin.valueChanged.connect(self.on_base_rgb_changed)
        base_layout.addWidget(self.base_b_spin, 3, 1)

        self.base_preview = QLabel()
        self.base_preview.setMinimumSize(60, 60)
        base_layout.addWidget(self.base_preview, 0, 2, 4, 1)

        base_group.setLayout(base_layout)
        main_layout.addWidget(base_group)

        # Output
        output_group = QGroupBox("Substance Painter Values to Match TBR Output")
        output_layout = QGridLayout()

        output_layout.addWidget(QLabel("Metallic:"), 0, 0)
        self.metallic_label = QLabel()
        output_layout.addWidget(self.metallic_label, 0, 1)

        output_layout.addWidget(QLabel("Roughness:"), 1, 0)
        self.roughness_label = QLabel()
        output_layout.addWidget(self.roughness_label, 1, 1)

        output_layout.addWidget(QLabel("---"), 2, 0, 1, 2)

        output_layout.addWidget(QLabel("Specular Red (ref):"), 3, 0)
        self.specular_label = QLabel()
        output_layout.addWidget(self.specular_label, 3, 1)

        output_layout.addWidget(QLabel("Glossiness (ref):"), 4, 0)
        self.glossiness_label = QLabel()
        output_layout.addWidget(self.glossiness_label, 4, 1)

        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)

        notes = QLabel(
            "Notes:\n"
            "- All calculations happen in linear light (sRGB in/out)\n"
            "- metallic = (linear(spec_red) - 0.01683) / (linear(base_r) - 0.01683)\n"
            "- Roughness = 1 - Glossiness\n"
            "- Only base color RED channel matters for this export preset"
        )
        notes.setWordWrap(True)
        main_layout.addWidget(notes)

        main_layout.addStretch()

        self._sync_tbr_hex()
        self._sync_base_hex()
        self.calculate()

    # Hex/RGB sync

    def _sync_tbr_hex(self):
        r = self.red_spin.value()
        g = self.green_spin.value()
        self.tbr_hex.setText(f"{r:02X}{g:02X}00")

    def _sync_base_hex(self):
        r = self.base_r_spin.value()
        g = self.base_g_spin.value()
        b = self.base_b_spin.value()
        self.base_hex.setText(f"{r:02X}{g:02X}{b:02X}")

    def on_tbr_hex_changed(self):
        if self._updating:
            return
        text = self.tbr_hex.text().strip()
        if len(text) == 6:
            try:
                r = int(text[0:2], 16)
                g = int(text[2:4], 16)
                self._updating = True
                self.red_spin.setValue(r)
                self.green_spin.setValue(g)
                self._updating = False
                self.calculate()
            except ValueError:
                pass

    def on_tbr_rgb_changed(self):
        if self._updating:
            return
        self._sync_tbr_hex()
        self.calculate()

    def on_base_hex_changed(self):
        if self._updating:
            return
        text = self.base_hex.text().strip()
        if len(text) == 6:
            try:
                r = int(text[0:2], 16)
                g = int(text[2:4], 16)
                b = int(text[4:6], 16)
                self._updating = True
                self.base_r_spin.setValue(r)
                self.base_g_spin.setValue(g)
                self.base_b_spin.setValue(b)
                self._updating = False
                self.calculate()
            except ValueError:
                pass

    def on_base_rgb_changed(self):
        if self._updating:
            return
        self._sync_base_hex()
        self.calculate()

    # Colour-space helpers

    @staticmethod
    def srgb_to_linear(c):
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    @staticmethod
    def linear_to_srgb(c):
        c = max(0.0, c)
        if c <= 0.0031308:
            return c * 12.92
        return 1.055 * (c ** (1.0 / 2.4)) - 0.055

    # Main calculation

    def calculate(self):
        red    = self.red_spin.value()
        green  = self.green_spin.value()
        base_r = self.base_r_spin.value()
        base_g = self.base_g_spin.value()
        base_b = self.base_b_spin.value()

        self.tbr_preview.setStyleSheet(f"background-color: rgb({red}, {green}, 0);")
        self.base_preview.setStyleSheet(f"background-color: rgb({base_r}, {base_g}, {base_b});")

        roughness = 1.0 - green / 255.0

        spec_linear = self.srgb_to_linear(red    / 255.0)
        base_linear = self.srgb_to_linear(base_r / 255.0)
        DIELECTRIC_LINEAR = 0.016831

        if base_linear > DIELECTRIC_LINEAR + 0.001:
            metallic = (spec_linear - DIELECTRIC_LINEAR) / (base_linear - DIELECTRIC_LINEAR)
            metallic = max(0.0, min(1.0, metallic))
        else:
            metallic = 0.0

        self.metallic_label.setText(f"{metallic:.3f}")
        self.roughness_label.setText(f"{roughness:.3f}")
        self.specular_label.setText(f"{red   / 255.0:.3f}")
        self.glossiness_label.setText(f"{green / 255.0:.3f}")


def main():
    app = QApplication(sys.argv)
    window = TBRConverter()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()