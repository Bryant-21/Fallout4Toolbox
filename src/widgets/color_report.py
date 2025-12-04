import json
from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QGridLayout
)

from src.utils.appconfig import cfg
from src.utils.logging_utils import logger


def generate_color_report_data(base_palette_array, color_distribution, image_height, image_width, palette_size):
    """Build a list of dicts describing the color report rows.

    Parameters:
        base_palette_array: np.ndarray of shape (palette_size, 3) containing RGB colors per grey_value
        color_distribution: dict mapping (R, G, B) tuples to pixel counts
        image_height: int
        image_width: int
        palette_size: int

    Returns:
        List[dict]: each with keys grey_value, color_rgb, frequency, frequency_percent
    """
    total_pixels = float(image_height * image_width) if image_height and image_width else 1.0
    rows = []
    for grey_value in range(palette_size):
        color = tuple(int(x) for x in base_palette_array[grey_value])
        frequency = int(color_distribution.get(color, 0))
        percent = float((frequency / total_pixels) * 100.0)
        rows.append({
            'grey_value': int(grey_value),
            'color_rgb': [int(color[0]), int(color[1]), int(color[2])],
            'frequency': frequency,
            'frequency_percent': percent
        })
    return rows


def save_color_report(color_report_data, file_path):
    """Save color report data as a JSON file.

    This keeps the previous structure for compatibility.
    """
    try:
        serializable_data = []
        for color_data in color_report_data:
            serializable_data.append({
                'grey_value': int(color_data['grey_value']),
                'color_rgb': [
                    int(color_data['color_rgb'][0]),
                    int(color_data['color_rgb'][1]),
                    int(color_data['color_rgb'][2])
                ],
                'frequency': int(color_data['frequency']),
                'frequency_percent': float(color_data['frequency_percent'])
            })

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_colors': len(serializable_data),
                'most_used_colors': sorted(serializable_data, key=lambda x: x['frequency'], reverse=True)[:10],
                'least_used_colors': sorted(serializable_data, key=lambda x: x['frequency'])[:10]
            },
            'color_mapping': serializable_data
        }

        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.debug(f"Color report saved to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving color report: {str(e)}")
        raise


class ColorReportWidget(QWidget):
    def __init__(self, color_report_data):
        super().__init__()
        self.color_report_data = color_report_data
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title = QLabel(
            f"Color to Greyscale Mapping Report - {cfg.get(cfg.ci_default_quant_method)} ({len(self.color_report_data)} colors)"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)

        # Description
        desc = QLabel(
            f"All {len(self.color_report_data)} colors with their assigned greyscale values. Colors are sorted perceptually by luminance and hue to keep similar colors together. Frequency shows how often each color appears in the image."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("margin: 5px; color: #666;")
        layout.addWidget(desc)

        # Create scroll area for the color grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QGridLayout(scroll_content)

        # Headers
        headers = ["Grey Value", "Color", "RGB Values", "Frequency", "Percent"]
        for col, header in enumerate(headers):
            label = QLabel(header)
            label.setStyleSheet("font-weight: bold; background-color: #e0e0e0; padding: 5px; border: 1px solid #ccc;")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            scroll_layout.addWidget(label, 0, col)

        # Add color rows
        for row, color_data in enumerate(self.color_report_data, 1):
            grey_value = color_data['grey_value']
            color_rgb = color_data['color_rgb']
            frequency = color_data['frequency']
            percent = color_data['frequency_percent']

            # Grey value
            grey_label = QLabel(str(grey_value))
            grey_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grey_label.setStyleSheet("padding: 5px; border: 1px solid #ccc;")
            scroll_layout.addWidget(grey_label, row, 0)

            # Color swatch
            color_widget = QLabel()
            color_widget.setMinimumSize(60, 30)
            color_widget.setStyleSheet(
                f"background-color: rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}); border: 1px solid #ccc;"
            )
            color_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            scroll_layout.addWidget(color_widget, row, 1)

            # RGB values
            rgb_label = QLabel(f"R: {color_rgb[0]:3d} G: {color_rgb[1]:3d} B: {color_rgb[2]:3d}")
            rgb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            rgb_label.setStyleSheet("padding: 5px; border: 1px solid #ccc; font-family: monospace;")
            scroll_layout.addWidget(rgb_label, row, 2)

            # Frequency
            freq_label = QLabel(f"{frequency:,}")
            freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            freq_label.setStyleSheet("padding: 5px; border: 1px solid #ccc; font-family: monospace;")
            scroll_layout.addWidget(freq_label, row, 3)

            # Percentage
            percent_label = QLabel(f"{percent:.2f}%")
            percent_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # Color code based on frequency
            if percent > 5:
                percent_label.setStyleSheet(
                    "padding: 5px; border: 1px solid #ccc; background-color: #ffcccc; font-family: monospace;"
                )
            elif percent > 1:
                percent_label.setStyleSheet(
                    "padding: 5px; border: 1px solid #ccc; background-color: #ffffcc; font-family: monospace;"
                )
            else:
                percent_label.setStyleSheet(
                    "padding: 5px; border: 1px solid #ccc; background-color: #ccffcc; font-family: monospace;"
                )
            scroll_layout.addWidget(percent_label, row, 4)

        scroll_layout.setColumnStretch(2, 1)
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        # Summary
        total_colors = len(self.color_report_data)
        dominant_colors = sorted(self.color_report_data, key=lambda x: x['frequency'], reverse=True)[:5]
        dominant_summary = " | ".join(
            [f"Grey {c['grey_value']}: {c['frequency_percent']:.1f}%" for c in dominant_colors]
        )

        summary = QLabel(f"Total colors: {total_colors} | Most used: {dominant_summary}")
        summary.setStyleSheet("margin: 10px; padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc;")
        layout.addWidget(summary)
