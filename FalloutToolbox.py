import logging
import os
import sys
import traceback

from PySide6 import QtWidgets
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from qfluentwidgets import FluentIcon as FIF, IndeterminateProgressRing, Dialog
from qfluentwidgets import Theme, setTheme, NavigationItemPosition

from src.utils.appconfig import VERSION
from src.utils.helpers import CustomFluentWindow
from src.utils.icons import CustomIcons


class FalloutToolboxMainWindow(CustomFluentWindow):

    def __init__(self):
        super().__init__()

        self.setupWindow()
        self.ring = IndeterminateProgressRing(self)
        self.ring.hide()
        self.show_progress()
        self.complete_loader()

        # Add third_party folder to Python module search path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        third_party_path = os.path.join(base_dir, "third_party")
        if third_party_path not in sys.path:
            sys.path.insert(0, third_party_path)

        from src.utils.capabilities import CAPABILITIES
        from src.utils.appconfig import cfg

        if not CAPABILITIES["mip_flooding"] and cfg.get(cfg.mipflood_check):
            w = Dialog("MIP Flooding Issue", "MIP Flooding Disabled, Unable to load", self)
            w.exec()
            cfg.set(cfg.mipflood_check, False)

        # Enable Fluent effects when available
        for attr in ('setMicaEffectEnabled', 'setAcrylicEffectEnabled'):
            try:
                fn = getattr(self, attr, None)
                if callable(fn):
                    fn(True)
            except Exception:
                pass

        #from src.widgets.palette_generator import PaletteGenerator
        from src.widgets.palette_creator import PaletteLUTGenerator

        from src.widgets.create_archlist import ArchlistWidget
        from src.widgets.esp_template_renamer_tool import ESPTemplaterWidget
        from src.widgets.dds_resizer import DDSResizerWindow
        from src.widgets.dds_inspector import DDSInspector
        from src.widgets.matfiles_copy import MaterialToolUI
        from src.settings.settings_widget import MainSettings
        from src.widgets.subgraph_maker import SubGraphMakerWindow
        from src.widgets.image_quantizer import ImageQuantizerWidget
        from src.widgets.bulk_nif_edit import UVPaddingRemoverWidget
        from src.widgets.nif_edit import SingleModelUVPadWidget
        from src.widgets.palette_applier import PaletteApplier
        from src.widgets.palette_adder import AddToPaletteWidget
        from src.widgets.palette_adjuster import PaletteAdjuster
        from src.widgets.bulk_palette_generator import BulkPaletteGeneratorWidget

        self.addSubInterface(DDSResizerWindow(self, "DDS Bulk Resizer"), CustomIcons.BULK.icon(), "DDS Bulk Resizer",
                             NavigationItemPosition.TOP)
        self.addSubInterface(DDSInspector(self, "DDS Inspector"), CustomIcons.IMAGE_VIEWER.icon(),
                             "DDS Inspector", NavigationItemPosition.TOP)
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
        self.addSubInterface(PaletteLUTGenerator(self, "Palette Generator"), CustomIcons.PALETTE_2.icon(stroke=True),
                             "Palette Generator", NavigationItemPosition.TOP)
        self.addSubInterface(BulkPaletteGeneratorWidget(self, "Bulk Palette Generator"), CustomIcons.BULK_EDIT.icon(),
                             "Bulk Palette Generator", NavigationItemPosition.TOP)
        self.addSubInterface(PaletteApplier(self, "Palette Preview"), CustomIcons.PREVIEW_FILE.icon(),
                             "Palette Preview", NavigationItemPosition.TOP)
        self.addSubInterface(AddToPaletteWidget(self, "Add To Palette"), CustomIcons.IMAGE_CIRCLE.icon(stroke=True),
                             "Add To Palette", NavigationItemPosition.TOP)
        self.addSubInterface(PaletteAdjuster(self, "Palette Adjuster"), CustomIcons.COLORS_SQUARE.icon(stroke=True),
                             "Palette Adjuster", NavigationItemPosition.TOP)
        self.navigationInterface.addSeparator()
        self.addSubInterface(ImageQuantizerWidget(self, "Image Quantizer"), CustomIcons.QUANT.icon(), "Image Quantizer",
                             NavigationItemPosition.TOP)

        if CAPABILITIES["mip_flooding"]:
            from src.widgets.mip_flooding import MipFloodingWidget
            self.addSubInterface(MipFloodingWidget(self, "MIP Flooding"), CustomIcons.FLOOD.icon(), "MIP Flooding", NavigationItemPosition.TOP)

        self.addSubInterface(UVPaddingRemoverWidget(self, "Bulk NIF UV Cleaner (WIP)"),
                             CustomIcons.CUT_FILM.icon(), "Bulk NIF UV Cleaner (WIP)", NavigationItemPosition.TOP)
        self.addSubInterface(SingleModelUVPadWidget(self, "Single NIF UV Cleaner"),
                             CustomIcons.CUBE.icon(stroke=True), "Single NIF UV Cleaner", NavigationItemPosition.TOP)

        if CAPABILITIES["ChaiNNer"]:
            from src.widgets.upscale import UpscaleWidget
            self.addSubInterface(UpscaleWidget(self, "Upscale"), CustomIcons.ENHANCE.icon(), "Upscale", NavigationItemPosition.TOP)

        self.addSubInterface(MainSettings(self), FIF.SETTING, 'Settings', NavigationItemPosition.BOTTOM)

        self.splashScreen.finish()


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

        screen = self.window().screen()
        desktop = screen.availableGeometry()

        self.setMinimumSize(1280, 900)
        self.setMaximumSize(desktop.width(), desktop.height())

        screen_w = desktop.width()
        screen_h = desktop.height()

        target_w = max(int(screen_w * 0.85), self.minimumWidth())
        target_h = max(int(screen_h * 0.85), self.minimumHeight())

        # Resize
        self.resize(target_w, target_h)

        # Center
        self.move(
            desktop.x() + (desktop.width() - self.width()) // 2,
            desktop.y() + (desktop.height() - self.height()) // 2,
        )

        self.show()
        QApplication.processEvents()
        self.setMicaEffectEnabled(True)

    def _append_log(self, text: str):
        self.log_view.appendPlainText(text)


def main():
    from src.utils.logging_utils import setup_logging

    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"

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
