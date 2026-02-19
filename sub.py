#!/usr/bin/env python3
"""
TBR to PBR Metal/Roughness Converter
Converts Fallout 4 TBR specular maps to Substance Painter PBR values.

The user provides TARGET values (what the game texture should look like):
  - Target TBR specular hex (R=specular, G=glossiness)
  - Target Diffuse color hex (the exported diffuse texture color)

The tool calculates what to set IN Substance Painter:
  - Metallic / Roughness sliders
  - Base color to use (which differs from diffuse because diffuse = base * (1-metal))
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
        self.setMinimumWidth(500)
        self._updating = False

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Title
        title = QLabel("Fallout 4 TBR → Substance Painter Converter")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        # Info
        info = QLabel(
            "Enter the TARGET values you want in the final game textures.\n"
            "The tool calculates what to set in Substance Painter to achieve them.\n\n"
            "Substance export: diffuse = base_color × (1 − metallic)\n"
            "                  spec_R  = lerp(dielectric, base_R, metallic)"
        )
        info.setWordWrap(True)
        main_layout.addWidget(info)

        hex_validator = QRegularExpressionValidator(QRegularExpression("[0-9A-Fa-f]{0,6}"))

        # --- TARGET TBR input ---
        tbr_group = QGroupBox("Target TBR Specular (from existing game texture)")
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

        # --- TARGET Diffuse color input ---
        diffuse_group = QGroupBox("Target Diffuse Color (exported game texture color)")
        diffuse_layout = QGridLayout()

        diffuse_layout.addWidget(QLabel("Hex:"), 0, 0)
        self.diffuse_hex = QLineEdit()
        self.diffuse_hex.setMaxLength(6)
        self.diffuse_hex.setPlaceholderText("e.g. 7B352B")
        self.diffuse_hex.setValidator(hex_validator)
        self.diffuse_hex.editingFinished.connect(self.on_diffuse_hex_changed)
        diffuse_layout.addWidget(self.diffuse_hex, 0, 1)

        diffuse_layout.addWidget(QLabel("RED:"), 1, 0)
        self.diffuse_r_spin = QSpinBox()
        self.diffuse_r_spin.setRange(0, 255)
        self.diffuse_r_spin.setValue(123)
        self.diffuse_r_spin.valueChanged.connect(self.on_diffuse_rgb_changed)
        diffuse_layout.addWidget(self.diffuse_r_spin, 1, 1)

        diffuse_layout.addWidget(QLabel("GREEN:"), 2, 0)
        self.diffuse_g_spin = QSpinBox()
        self.diffuse_g_spin.setRange(0, 255)
        self.diffuse_g_spin.setValue(53)
        self.diffuse_g_spin.valueChanged.connect(self.on_diffuse_rgb_changed)
        diffuse_layout.addWidget(self.diffuse_g_spin, 2, 1)

        diffuse_layout.addWidget(QLabel("BLUE:"), 3, 0)
        self.diffuse_b_spin = QSpinBox()
        self.diffuse_b_spin.setRange(0, 255)
        self.diffuse_b_spin.setValue(43)
        self.diffuse_b_spin.valueChanged.connect(self.on_diffuse_rgb_changed)
        diffuse_layout.addWidget(self.diffuse_b_spin, 3, 1)

        self.diffuse_preview = QLabel()
        self.diffuse_preview.setMinimumSize(60, 60)
        diffuse_layout.addWidget(self.diffuse_preview, 0, 2, 4, 1)

        diffuse_group.setLayout(diffuse_layout)
        main_layout.addWidget(diffuse_group)

        # --- Output: Substance Painter values ---
        output_group = QGroupBox("Set These in Substance Painter")
        output_layout = QGridLayout()

        output_layout.addWidget(QLabel("Metallic:"), 0, 0)
        self.metallic_label = QLabel()
        self.metallic_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        output_layout.addWidget(self.metallic_label, 0, 1)

        output_layout.addWidget(QLabel("Roughness:"), 1, 0)
        self.roughness_label = QLabel()
        self.roughness_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        output_layout.addWidget(self.roughness_label, 1, 1)

        output_layout.addWidget(QLabel("---"), 2, 0, 1, 2)

        output_layout.addWidget(QLabel("Substance Base Color:"), 3, 0)
        self.substance_color_label = QLabel()
        self.substance_color_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        output_layout.addWidget(self.substance_color_label, 3, 1)

        self.substance_preview = QLabel()
        self.substance_preview.setMinimumSize(60, 60)
        output_layout.addWidget(self.substance_preview, 3, 2, 1, 1)

        output_layout.addWidget(QLabel("---"), 4, 0, 1, 2)

        output_layout.addWidget(QLabel("Verification:"), 5, 0, 1, 2)

        output_layout.addWidget(QLabel("  Predicted Spec R:"), 6, 0)
        self.predicted_spec_label = QLabel()
        self.predicted_spec_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        output_layout.addWidget(self.predicted_spec_label, 6, 1)

        output_layout.addWidget(QLabel("  Predicted Diffuse R:"), 7, 0)
        self.predicted_diffuse_label = QLabel()
        self.predicted_diffuse_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        output_layout.addWidget(self.predicted_diffuse_label, 7, 1)

        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)

        notes = QLabel(
            "How it works:\n"
            "1. Roughness = 1 − (green/255)  (glossiness → roughness)\n"
            "2. Metallic is solved from: spec_R = lerp(dielectric, base_R, metallic)\n"
            "   where base_R = diffuse_R / (1 − metallic)  (iterative solve)\n"
            "3. Substance Base Color = diffuse / (1 − metallic)\n"
            "   This is the color to paint in Substance so the exported diffuse matches."
        )
        notes.setWordWrap(True)
        main_layout.addWidget(notes)

        main_layout.addStretch()

        self._sync_tbr_hex()
        self._sync_diffuse_hex()
        self.calculate()

    # Hex/RGB sync

    def _sync_tbr_hex(self):
        r = self.red_spin.value()
        g = self.green_spin.value()
        self.tbr_hex.setText(f"{r:02X}{g:02X}00")

    def _sync_diffuse_hex(self):
        r = self.diffuse_r_spin.value()
        g = self.diffuse_g_spin.value()
        b = self.diffuse_b_spin.value()
        self.diffuse_hex.setText(f"{r:02X}{g:02X}{b:02X}")

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

    def on_diffuse_hex_changed(self):
        if self._updating:
            return
        text = self.diffuse_hex.text().strip()
        if len(text) == 6:
            try:
                r = int(text[0:2], 16)
                g = int(text[2:4], 16)
                b = int(text[4:6], 16)
                self._updating = True
                self.diffuse_r_spin.setValue(r)
                self.diffuse_g_spin.setValue(g)
                self.diffuse_b_spin.setValue(b)
                self._updating = False
                self.calculate()
            except ValueError:
                pass

    def on_diffuse_rgb_changed(self):
        if self._updating:
            return
        self._sync_diffuse_hex()
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
        spec_red_srgb = self.red_spin.value()
        gloss_srgb    = self.green_spin.value()
        diff_r = self.diffuse_r_spin.value()
        diff_g = self.diffuse_g_spin.value()
        diff_b = self.diffuse_b_spin.value()

        self.tbr_preview.setStyleSheet(f"background-color: rgb({spec_red_srgb}, {gloss_srgb}, 0);")
        self.diffuse_preview.setStyleSheet(f"background-color: rgb({diff_r}, {diff_g}, {diff_b});")

        # Roughness from glossiness
        roughness = 1.0 - gloss_srgb / 255.0

        # Convert to linear
        spec_linear = self.srgb_to_linear(spec_red_srgb / 255.0)
        diff_r_linear = self.srgb_to_linear(diff_r / 255.0)
        diff_g_linear = self.srgb_to_linear(diff_g / 255.0)
        diff_b_linear = self.srgb_to_linear(diff_b / 255.0)

        DIELECTRIC_LINEAR = 0.016831

        # Iterative solve for metallic:
        # We know:
        #   diffuse_R = base_R * (1 - metallic)       => base_R = diffuse_R / (1 - metallic)
        #   spec_R    = dielectric + metallic * (base_R - dielectric)
        # Substituting:
        #   spec_R = dielectric + metallic * (diffuse_R / (1 - metallic) - dielectric)
        # Solve for metallic iteratively.

        metallic = 0.0
        if diff_r_linear > 0.001 and spec_linear > DIELECTRIC_LINEAR + 0.001:
            # Newton-style iteration
            for _ in range(50):
                if metallic >= 0.999:
                    metallic = 0.999
                    break
                one_minus_m = 1.0 - metallic
                base_r_lin = diff_r_linear / one_minus_m
                predicted_spec = DIELECTRIC_LINEAR + metallic * (base_r_lin - DIELECTRIC_LINEAR)
                error = predicted_spec - spec_linear
                # Derivative of predicted_spec w.r.t. metallic:
                # d/dm [D + m * (diff/(1-m) - D)]
                # = (diff/(1-m) - D) + m * diff/(1-m)^2
                # = diff/(1-m) - D + m*diff/(1-m)^2
                deriv = (diff_r_linear / one_minus_m - DIELECTRIC_LINEAR +
                         metallic * diff_r_linear / (one_minus_m ** 2))
                if abs(deriv) < 1e-10:
                    break
                metallic -= error / deriv
                metallic = max(0.0, min(0.999, metallic))
                if abs(error) < 1e-8:
                    break

        # Compute Substance base color = diffuse / (1 - metallic) for all channels
        one_minus_m = 1.0 - metallic
        warning = ""

        if metallic >= 0.998:
            # Metallic hit the cap — derive base_R from specular equation instead
            # spec_R = dielectric + metallic * (base_R - dielectric)
            # base_R = dielectric + (spec_R - dielectric) / metallic
            base_r_lin = DIELECTRIC_LINEAR + (spec_linear - DIELECTRIC_LINEAR) / metallic
            base_r_lin = max(0.0, min(1.0, base_r_lin))
            # Scale G/B proportionally from diffuse ratios
            if diff_r_linear > 0.001:
                ratio_g = diff_g_linear / diff_r_linear
                ratio_b = diff_b_linear / diff_r_linear
            else:
                ratio_g = 1.0
                ratio_b = 1.0
            base_g_lin = min(1.0, base_r_lin * ratio_g)
            base_b_lin = min(1.0, base_r_lin * ratio_b)
            warning = " ⚠ Spec too high for diffuse — metallic capped, diffuse may not match exactly"
        elif one_minus_m > 0.001:
            base_r_lin = diff_r_linear / one_minus_m
            base_g_lin = diff_g_linear / one_minus_m
            base_b_lin = diff_b_linear / one_minus_m
            # Check if any channel exceeds 1.0 (would clamp and lose accuracy)
            if base_r_lin > 1.0 or base_g_lin > 1.0 or base_b_lin > 1.0:
                warning = " ⚠ Base color clipped — diffuse may not match exactly"
            base_r_lin = min(1.0, base_r_lin)
            base_g_lin = min(1.0, base_g_lin)
            base_b_lin = min(1.0, base_b_lin)
        else:
            base_r_lin = diff_r_linear
            base_g_lin = diff_g_linear
            base_b_lin = diff_b_linear

        # Convert back to sRGB
        base_r_out = int(round(self.linear_to_srgb(base_r_lin) * 255))
        base_g_out = int(round(self.linear_to_srgb(base_g_lin) * 255))
        base_b_out = int(round(self.linear_to_srgb(base_b_lin) * 255))

        self.metallic_label.setText(f"{metallic:.3f}")
        self.roughness_label.setText(f"{roughness:.3f}")
        self.substance_color_label.setText(
            f"{base_r_out:02X}{base_g_out:02X}{base_b_out:02X}  "
            f"(R={base_r_out} G={base_g_out} B={base_b_out}){warning}"
        )
        self.substance_preview.setStyleSheet(
            f"background-color: rgb({base_r_out}, {base_g_out}, {base_b_out});"
        )

        # Verification: predict what the export will produce
        pred_spec_lin = DIELECTRIC_LINEAR + metallic * (base_r_lin - DIELECTRIC_LINEAR)
        pred_spec_srgb = int(round(self.linear_to_srgb(pred_spec_lin) * 255))
        pred_diff_r_lin = base_r_lin * one_minus_m
        pred_diff_r_srgb = int(round(self.linear_to_srgb(pred_diff_r_lin) * 255))

        self.predicted_spec_label.setText(
            f"{pred_spec_srgb} (0x{pred_spec_srgb:02X})  target={spec_red_srgb} (0x{spec_red_srgb:02X})"
        )
        self.predicted_diffuse_label.setText(
            f"{pred_diff_r_srgb} (0x{pred_diff_r_srgb:02X})  target={diff_r} (0x{diff_r:02X})"
        )


def main():
    app = QApplication(sys.argv)
    window = TBRConverter()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
