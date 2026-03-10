import logging
import os
import sys
import traceback

from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout
from qfluentwidgets import (
    FluentIcon as FIF,
    IndeterminateProgressRing,
    Dialog,
    Theme,
    setTheme,
    NavigationItemPosition,
    CardWidget,
    IconWidget,
    BodyLabel,
    CaptionLabel,
    PushButton,
    TransparentToolButton, SingleDirectionScrollArea,
)

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

        # keep child windows referenced
        self._child_windows = []

        # HKX Watcher state
        self.hkx_watchers = {}  # Map of path -> HKXWatchWorker

        # Add third_party folder to Python module search path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        third_party_path = os.path.join(base_dir, "third_party")
        if third_party_path not in sys.path:
            sys.path.insert(0, third_party_path)

        from src.utils.capabilities import CAPABILITIES
        from src.utils.appconfig import cfg

        if not CAPABILITIES["mip_flooding"] and cfg.get(cfg.mipflood_check):
            w = Dialog(
                "MIP Flooding Issue", "MIP Flooding Disabled, Unable to load", self
            )
            w.exec()
            cfg.set(cfg.mipflood_check, False)

        # Enable Fluent effects when available
        for attr in ("setMicaEffectEnabled", "setAcrylicEffectEnabled"):
            try:
                fn = getattr(self, attr, None)
                if callable(fn):
                    fn(True)
            except Exception:
                pass

        # Build lightweight start page with cards
        start = self._build_start_page()
        self.addSubInterface(
            start,
            FIF.HOME,
            "Home",
            NavigationItemPosition.TOP,
        )
        # Collapse the navigation (burger menu) on the Home page
        try:
            self.navigationInterface.collapse()
        except Exception:
            pass
        self.splashScreen.finish()

        # Start HKX watchers if enabled
        if cfg.get(cfg.hkx_watch_enabled):
            watch_dirs_raw = cfg.get(cfg.hkx_watch_dirs)
            for item in watch_dirs_raw:
                if "|" in item:
                    path, sub = item.split("|", 1)
                    include_sub = sub.lower() == "true"
                    if os.path.isdir(path):
                        self.start_hkx_watcher(path, include_sub)

    def start_hkx_watcher(self, watch_dir: str, include_subdirs: bool):
        from src.widgets.hkx_watcher_widget import HKXWatchWorker
        watch_dir = os.path.normpath(watch_dir)
        if watch_dir in self.hkx_watchers:
            self.stop_hkx_watcher(watch_dir)
        
        worker = HKXWatchWorker(watch_dir, include_subdirs)
        worker.start()
        self.hkx_watchers[watch_dir] = worker

    def stop_hkx_watcher(self, watch_dir: str):
        watch_dir = os.path.normpath(watch_dir)
        if watch_dir in self.hkx_watchers:
            worker = self.hkx_watchers.pop(watch_dir)
            worker.abort()
            worker.wait()

    def stop_all_hkx_watchers(self):
        paths = list(self.hkx_watchers.keys())
        for path in paths:
            self.stop_hkx_watcher(path)

    class AppCard(CardWidget):
        def __init__(self, icon, title, content, parent=None):
            super().__init__(parent)
            self.iconWidget = IconWidget(icon)
            self.titleLabel = BodyLabel(title, self)
            self.contentLabel = CaptionLabel(content, self)
            self.openButton = PushButton("Open", self)
            self.moreButton = TransparentToolButton(FIF.MORE, self)

            self.hBoxLayout = QHBoxLayout(self)
            self.vBoxLayout = QVBoxLayout()

            self.setFixedHeight(73)
            self.iconWidget.setFixedSize(48, 48)
            self.contentLabel.setTextColor("#606060", "#d2d2d2")
            self.openButton.setFixedWidth(120)

            self.hBoxLayout.setContentsMargins(20, 11, 11, 11)
            self.hBoxLayout.setSpacing(15)
            self.hBoxLayout.addWidget(self.iconWidget)

            self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
            self.vBoxLayout.setSpacing(0)
            self.vBoxLayout.addWidget(self.titleLabel, 0, Qt.AlignVCenter)
            self.vBoxLayout.addWidget(self.contentLabel, 0, Qt.AlignVCenter)
            self.vBoxLayout.setAlignment(Qt.AlignVCenter)
            self.hBoxLayout.addLayout(self.vBoxLayout)

            self.hBoxLayout.addStretch(1)
            self.hBoxLayout.addWidget(self.openButton, 0, Qt.AlignRight)
            self.hBoxLayout.addWidget(self.moreButton, 0, Qt.AlignRight)

            self.moreButton.setFixedSize(32, 32)

    def _build_start_page(self) -> QWidget:
        scrollArea = SingleDirectionScrollArea(orient=Qt.Vertical)
        scrollArea.setWidgetResizable(True)

        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setAlignment(Qt.AlignTop)

        # Define cards: title, desc, icon, handler
        cards = [
            ("DDS Tools", "Resize, inspect and export DDS", CustomIcons.IMAGE_VIEWER.icon(), self._open_dds_window),
            ("Palette Tools", "Generate, apply and quantize color palettes", CustomIcons.PALETTE_2.icon(stroke=True), self._open_palette_window),
            ("Havok / Animation", "HKX pack/unpack and annotations", CustomIcons.FILE_CODE.icon(stroke=True), self._open_havok_window),
            ("Archives & Audio", "Audio and weapon firing tools", CustomIcons.FILE_SEARCH.icon(stroke=True), self._open_archive_audio_window),
            ("Models / NIF", "UV cleanup utilities", CustomIcons.CUBE.icon(stroke=True), self._open_models_window),
            ("Utilities", "BSA extractor, ESP templater, renamers", CustomIcons.PUZZLE.icon(), self._open_utilities_window),
            ("Substance TBR", "Texture baking runner", FIF.PALETTE, self._open_tbr_window),
            ("Upscale", "Image upscaling (if available)", CustomIcons.ENHANCE.icon(), self._open_upscale_window),
            ("Settings", "Configure Fallout Tools", FIF.SETTING, self._open_settings_window),
        ]

        for title, desc, icon, handler in cards:
            card = self.AppCard(icon=icon, title=title, content=desc)
            card.clicked.connect(handler)
            card.openButton.clicked.connect(handler)
            layout.addWidget(card)

        scrollArea.setWidget(view)
        scrollArea.setObjectName("startPage")
        scrollArea.title = "Home"
        return scrollArea

    # Lazy openers: create a new window and add relevant sub-interfaces
    def _spawn_window(self, title: str, builder_fn):
        w = CustomFluentWindow()
        w.setWindowTitle(title)
        try:
            w.setMicaEffectEnabled(True)
        except Exception:
            pass
        builder_fn(w)
        # Ensure the child window's splash screen is dismissed after building
        try:
            w.splashScreen.finish()
        except Exception:
            pass
        # Match size to main window and collapse the navigation in child windows
        try:
            w.resize(self.size())
            # Center the child window on the same screen as the main window
            screen = self.screen()
            if screen:
                desktop = screen.availableGeometry()
                w.move(
                    desktop.x() + (desktop.width() - w.width()) // 2,
                    desktop.y() + (desktop.height() - w.height()) // 2,
                )
        except Exception:
            pass
        try:
            w.navigationInterface.collapse()
        except Exception:
            pass
        w.show()
        self._child_windows.append(w)

    def _open_dds_window(self):
        def build(w: CustomFluentWindow):
            from src.widgets.dds_resizer import DDSResizerWindow
            from src.widgets.dds_png_exporter import DDSPNGExporterWindow
            from src.widgets.dds_inspector import DDSInspector
            w.addSubInterface(DDSResizerWindow(w, "DDS Bulk Resizer"), CustomIcons.BULK.icon(), "DDS Bulk Resizer", NavigationItemPosition.TOP)
            w.addSubInterface(DDSInspector(w, "DDS Inspector"), CustomIcons.IMAGE_VIEWER.icon(), "DDS Inspector", NavigationItemPosition.TOP)
            w.addSubInterface(DDSPNGExporterWindow(w, "DDS → PNG Exporter"), CustomIcons.IMAGE_VIEWER.icon(), "DDS → PNG Exporter", NavigationItemPosition.TOP)
        self._spawn_window("DDS Tools", build)

    def _open_palette_window(self):
        def build(w: CustomFluentWindow):
            from src.widgets.palette_creator import PaletteLUTGenerator
            from src.widgets.bulk_palette_generator import BulkPaletteGeneratorWidget
            from src.widgets.palette_applier import PaletteApplier
            from src.widgets.palette_adder import AddToPaletteWidget
            from src.widgets.palette_adjuster import PaletteAdjuster
            from src.widgets.image_quantizer import ImageQuantizerWidget
            w.addSubInterface(PaletteLUTGenerator(w, "Palette Generator"), CustomIcons.PALETTE_2.icon(stroke=True), "Palette Generator", NavigationItemPosition.TOP)
            w.addSubInterface(BulkPaletteGeneratorWidget(w, "Bulk Palette Generator"), CustomIcons.BULK_EDIT.icon(), "Bulk Palette Generator", NavigationItemPosition.TOP)
            w.addSubInterface(PaletteApplier(w, "Palette Preview"), CustomIcons.PREVIEW_FILE.icon(), "Palette Preview", NavigationItemPosition.TOP)
            w.addSubInterface(AddToPaletteWidget(w, "Add To Palette"), CustomIcons.IMAGE_CIRCLE.icon(stroke=True), "Add To Palette", NavigationItemPosition.TOP)
            w.addSubInterface(PaletteAdjuster(w, "Palette Adjuster"), CustomIcons.COLORS_SQUARE.icon(stroke=True), "Palette Adjuster", NavigationItemPosition.TOP)
            w.addSubInterface(ImageQuantizerWidget(w, "Image Quantizer"), CustomIcons.QUANT.icon(), "Image Quantizer", NavigationItemPosition.TOP)
        self._spawn_window("Palette Tools", build)

    def _open_havok_window(self):
        def build(w: CustomFluentWindow):
            from src.widgets.hkx_pack_widget import HKXPackWidget
            from src.widgets.hkx_watcher_widget import HKXWatchWidget
            from src.widgets.animation_annotation_extractor import AnnotationExtractorWidget
            w.addSubInterface(HKXPackWidget(w, "HKX Pack / Unpack"), CustomIcons.FILE_CODE.icon(), "HKX Pack / Unpack", NavigationItemPosition.TOP)
            w.addSubInterface(HKXWatchWidget(w, "HKX Auto-Packer"), CustomIcons.LOOP.icon(), "HKX Auto-Packer", NavigationItemPosition.TOP)
            w.addSubInterface(AnnotationExtractorWidget(w, "Annotation Extractor"), CustomIcons.FILE_EDIT.icon(stroke=True), "Annotation Extractor", NavigationItemPosition.TOP)
        self._spawn_window("Havok / Animation", build)

    def _open_archive_audio_window(self):
        def build(w: CustomFluentWindow):
            from src.widgets.audio_extractor_widget import AudioExtractorWidget
            from src.widgets.gun_fire_generator import GunFireGeneratorWidget
            from src.widgets.laser_beam_generator import LaserBeamGeneratorWidget
            w.addSubInterface(AudioExtractorWidget(w, "Audio Extractor"), CustomIcons.MUSIC.icon(stroke=True), "Audio Extractor", NavigationItemPosition.TOP)
            w.addSubInterface(GunFireGeneratorWidget(w, "Gun Fire Generator"), FIF.MUSIC, "Gun Fire Generator", NavigationItemPosition.TOP)
            w.addSubInterface(LaserBeamGeneratorWidget(w, "Laser Beam Generator"), FIF.MUSIC, "Laser Beam Generator", NavigationItemPosition.TOP)
        self._spawn_window("Archives & Audio", build)

    def _open_models_window(self):
        def build(w: CustomFluentWindow):
            from src.widgets.bulk_nif_edit import UVPaddingRemoverWidget
            from src.widgets.nif_edit import SingleModelUVPadWidget
            w.addSubInterface(UVPaddingRemoverWidget(w, "Bulk NIF UV Cleaner (WIP)"), CustomIcons.CUT_FILM.icon(), "Bulk NIF UV Cleaner (WIP)", NavigationItemPosition.TOP)
            w.addSubInterface(SingleModelUVPadWidget(w, "Single NIF UV Cleaner"), CustomIcons.CUBE.icon(stroke=True), "Single NIF UV Cleaner", NavigationItemPosition.TOP)
        self._spawn_window("Models / NIF", build)

    def _open_utilities_window(self):
        def build(w: CustomFluentWindow):
            from src.widgets.bsa_extractor_widget import BSAExtractorWidget
            from src.widgets.esp_template_renamer_tool import ESPTemplaterWidget
            from src.widgets.folder_renamer_widget import FolderRenamerWidget
            from src.widgets.modlist_merger_widget import ModlistMergerWidget
            from src.widgets.create_archlist import ArchlistWidget
            from src.widgets.matfiles_copy import MaterialToolUI
            from src.widgets.subgraph_maker import SubGraphMakerWindow
            from src.widgets.papyrus_decompiler_widget import PapyrusDecompilerWidget
            w.addSubInterface(BSAExtractorWidget(w, "BSA Extractor"), CustomIcons.FILE_SEARCH.icon(), "BSA Extractor", NavigationItemPosition.TOP)
            w.addSubInterface(ESPTemplaterWidget(w, "ESP Template Mod Maker"), CustomIcons.PUZZLE.icon(), "ESP Template Mod Maker", NavigationItemPosition.TOP)
            w.addSubInterface(FolderRenamerWidget(w, "Folder Renamer"), CustomIcons.FOLDERRIGHT.icon(stroke=True), "Folder Renamer", NavigationItemPosition.TOP)
            w.addSubInterface(ModlistMergerWidget(w, "Modlist Merger"), CustomIcons.COMBINE.icon(), "Modlist Merger", NavigationItemPosition.TOP)
            w.addSubInterface(ArchlistWidget(w, "Archlist Creator"), FIF.PENCIL_INK, "Archlist Creator", NavigationItemPosition.TOP)
            w.addSubInterface(MaterialToolUI(w, "Material File Copier"), CustomIcons.BGSM.icon(), "Material File Copier", NavigationItemPosition.TOP)
            w.addSubInterface(SubGraphMakerWindow(w, "SubGraph Maker"), CustomIcons.GRAPH.icon(stroke=True), "SubGraph Maker", NavigationItemPosition.TOP)
            w.addSubInterface(PapyrusDecompilerWidget(w, "Papyrus Decompiler"), FIF.DEVELOPER_TOOLS, "Papyrus Decompiler", NavigationItemPosition.TOP)
        self._spawn_window("Utilities", build)

    def _open_tbr_window(self):
        def build(w: CustomFluentWindow):
            from src.widgets.substance_tbr import SubstanceTBRWidget
            w.addSubInterface(SubstanceTBRWidget(w, "Substance TBR"), FIF.PALETTE, "Substance TBR", NavigationItemPosition.TOP)
        self._spawn_window("Substance TBR", build)

    def _open_upscale_window(self):
        from src.utils.capabilities import CAPABILITIES
        if not CAPABILITIES.get("ChaiNNer", False):
            # simple info dialog if not available
            w = Dialog("Upscale Unavailable", "ChaiNNer capability not detected.", self)
            w.exec()
            return

        def build(w: CustomFluentWindow):
            from src.widgets.upscale import UpscaleWidget
            w.addSubInterface(UpscaleWidget(w, "Upscale"), CustomIcons.ENHANCE.icon(), "Upscale", NavigationItemPosition.TOP)
        self._spawn_window("Upscale", build)

    def _open_settings_window(self):
        def build(w: CustomFluentWindow):
            from src.settings.settings_widget import MainSettings
            w.addSubInterface(MainSettings(w), FIF.SETTING, "Settings", NavigationItemPosition.TOP)
        self._spawn_window("Settings", build)

    def show_progress(self):
        self.setEnabled(False)
        x = (self.width() - (self.ring.width() // 4)) // 2
        y = (self.height() - self.ring.height()) // 2
        self.ring.move(x, y)
        self.ring.show()

    def update_progress(self, value: int):
        # self.ring.setValue(value)
        pass

    def complete_loader(self):
        self.ring.setValue(0)
        self.ring.hide()
        self.setEnabled(True)

    def setupWindow(self):
        self.setWindowTitle(f"Fallout Tools - {VERSION}")

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
        logger = logging.getLogger("main")
    except Exception as e:
        traceback.print_exc()
        logger = logging.getLogger("main")
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
