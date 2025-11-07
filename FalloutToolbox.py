import logging
import os
import sys
import traceback

from PySide6 import QtWidgets
from PySide6.QtCore import Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from qfluentwidgets import FluentIcon as FIF, StateToolTip, ProgressRing, IndeterminateProgressRing
from qfluentwidgets import Theme, setTheme, NavigationItemPosition

from src.utils.appconfig import VERSION
from src.utils.helpers import CustomFluentWindow
from src.utils.icons import CustomIcons


class FalloutToolboxMainWindow(CustomFluentWindow):

    def __init__(self):
        super().__init__()
        self.icon_path = "resource/icon.ico"
        self.icon = QIcon(self.icon_path)
        self.setupWindow()
        self.ring = IndeterminateProgressRing(self)
        self.ring.hide()
        self.show_progress()
        self.complete_loader()
        # Enable Fluent effects when available
        for attr in ('setMicaEffectEnabled', 'setAcrylicEffectEnabled'):
            try:
                fn = getattr(self, attr, None)
                if callable(fn):
                    fn(True)
            except Exception:
                pass

        # Add third_party folder to Python module search path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        third_party_path = os.path.join(base_dir, "third_party")
        if third_party_path not in sys.path:
            sys.path.insert(0, third_party_path)

        from src.widgets.palette_generator import PaletteGenerator
        from src.widgets.create_archlist import ArchlistWidget
        from src.widgets.esp_template_renamer_tool import ESPTemplaterWidget
        from src.widgets.dds_resizer import DDSResizerWindow
        from src.widgets.matfiles_copy import MaterialToolUI
        from src.settings.settings_widget import MainSettings
        from src.widgets.subgraph_maker import SubGraphMakerWindow
        from src.widgets.bulk_palette import BulkPaletteWidget
        from src.widgets.image_quantizer import ImageQuantizerWidget
        from src.widgets.combine_palettes import CombinePaletteGroupsWidget
        from src.widgets.nif_texture_edit import UVPaddingRemoverWidget
        from src.widgets.texture_to_greyscale import ConvertToPaletteWidget
        from src.widgets.add_colors_to_palette import AddColorsToPaletteWidget
        from src.widgets.mip_flooding import MipFloodingWidget

        self.addSubInterface(DDSResizerWindow(self, "DDS Bulk Resizer"), CustomIcons.BULK.icon(), "DDS Bulk Resizer",
                             NavigationItemPosition.TOP)
        self.addSubInterface(ESPTemplaterWidget(self, "ESP Template Mod Maker"), CustomIcons.PUZZLE.icon(),
                             "ESP Template Mod Maker",
                             NavigationItemPosition.TOP)
        self.addSubInterface(SubGraphMakerWindow(self, "SubGraph Maker"), CustomIcons.GRAPH.icon(stroke=True),
                             "SubGraph Maker",
                             NavigationItemPosition.TOP)
        self.addSubInterface(MaterialToolUI(self, "Material File Copier"), CustomIcons.BGSM.icon(),
                             "Material File Copier",
                             NavigationItemPosition.TOP)
        self.addSubInterface(ArchlistWidget(self, "Archlist Creator"), FIF.PENCIL_INK, "Archlist Creator",
                             NavigationItemPosition.TOP)
        self.navigationInterface.addSeparator()
        self.addSubInterface(PaletteGenerator(self, "Palette Generator"), CustomIcons.PALETTE.icon(),
                             "Palette Generator", NavigationItemPosition.TOP)
        self.addSubInterface(ConvertToPaletteWidget(self, "Greyscale Creator"), CustomIcons.GREYSCALE.icon(),
                             "Greyscale Creator", NavigationItemPosition.TOP)
        self.addSubInterface(AddColorsToPaletteWidget(self, "Add Colors To Palette"), CustomIcons.ADD_SOLID.icon(stroke=True),
                             "Add Colors To Palette", NavigationItemPosition.TOP)
        self.addSubInterface(CombinePaletteGroupsWidget(self, "Palette Group Combiner (WIP)"),
                             CustomIcons.COMBINE.icon(), "Palette Group Combiner (WIP)", NavigationItemPosition.TOP)
        self.addSubInterface(BulkPaletteWidget(self, "Texture Set Palette Generator (WIP)"), CustomIcons.SWATCH.icon(),
                             "Texture Set Palette Generator (WIP)", NavigationItemPosition.TOP)
        self.navigationInterface.addSeparator()
        self.addSubInterface(ImageQuantizerWidget(self, "Image Quantizer"), CustomIcons.QUANT.icon(), "Image Quantizer",
                             NavigationItemPosition.TOP)
        self.addSubInterface(MipFloodingWidget(self, "MIP Flooding"), CustomIcons.FLOOD.icon(), "MIP Flooding", NavigationItemPosition.TOP)
        self.addSubInterface(UVPaddingRemoverWidget(self, "NIF UV Cleaner (WIP)"),
                             CustomIcons.CUT.icon(), "NIF UV Cleaner (WIP)", NavigationItemPosition.TOP)
        self.addSubInterface(MainSettings(self), FIF.SETTING, 'Settings', NavigationItemPosition.BOTTOM)


    def show_progress(self):
        self.setEnabled(False)
        x = (self.width() - (self.ring.width() // 4)) // 2
        y = (self.height() - self.ring.height()) // 2
        self.ring.move(x, y)
        self.ring.show()

    def update_progress(self, value: int):
        #self.ring.setValue(value)
        pass

    def complete_loader(self):
        self.ring.setValue(0)
        self.ring.hide()
        self.setEnabled(True)


    def setupWindow(self):
        self.setWindowTitle(f'Fallout Tools - {VERSION}')
        self.setWindowIcon(self.icon)
        desktop = QApplication.screens()[0].availableGeometry()
        # Set size
        self.setMinimumSize(1280, 900)
        self.resize(1280, 900)
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)

        QApplication.processEvents()
        self.show()
        QApplication.processEvents()
        self.setMicaEffectEnabled(True)

    def _append_log(self, text: str):
        self.log_view.appendPlainText(text)


def main():
    from src.utils.logging_utils import setup_logging

    logger = None
    try:
        # Configure the logger
        root_logger = setup_logging()
        logger = logging.getLogger('main')
    except Exception as e:
        traceback.print_exc()
        logger = logging.getLogger('main')
        logger.debug("Unable to Start FallTalk Logging", e)

    if QtWidgets is None:
        print("PySide6 is not installed. Please install it with: pip install PySide6")
        sys.exit(1)
    app = QtWidgets.QApplication(sys.argv[:1])
    # Initialize Fluent theme (auto follows system light/dark)
    try:
        setTheme(Theme.AUTO)
    except Exception:
        pass
    w = FalloutToolboxMainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
