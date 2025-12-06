from src.settings.generic_settings import GenericSettings
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QFileDialog
from qfluentwidgets import FluentIcon as FIF, SettingCardGroup, isDarkTheme, PushSettingCard, InfoBar, HyperlinkCard, \
    CustomColorSettingCard, OptionsSettingCard, SwitchSettingCard
from qfluentwidgets import ScrollArea, ExpandLayout

from src.utils.appconfig import cfg, HELP_URL, NEXUS_URL, VERSION, YEAR, AUTHOR, KOFI_URL, DISCORD_URL
from src.utils.cards import SpinSettingCard, RadioSettingCard, RangeSettingCardScaled
from src.utils.icons import CustomIcons

class PaletteSettings(GenericSettings):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.palette_size_card = RadioSettingCard(
            cfg.ci_default_palette_size,
            CustomIcons.WIDTH.icon(),
            self.tr("Palette Size"),
            self.tr("Number of colors to quantize to"),
            texts=["256", "128", "64", "32"],
            parent=self
        )

        self.method_card = OptionsSettingCard(
            cfg.ci_default_quant_method,
            CustomIcons.QUANT.icon(),
            self.tr("Quantization Method"),
            self.tr("How do we reduce the color palette of the base images"),
            texts=[
                "median_cut - Good color relationships (default)",
                "max_coverage - Maximizes color variety",
                "fast_octree - Fast, good for photos",
                "libimagequant - High quality",
                "kmeans_adaptive - Adaptive color distribution",
                "uniform - Helps with color banding"
            ],
            parent=self
        )

        self.row_height_card = SpinSettingCard(
            cfg.ci_palette_row_height,
            CustomIcons.HEIGHT.icon(),
            self.tr("Palette row height (pixels)"),
            self.tr("2 to 8"), step=2
        )

        self.__initWidget()

    def __initWidget(self):

        self.settings_group.addSettingCard(self.palette_size_card)
        self.settings_group.addSettingCard(self.method_card)
        self.settings_group.addSettingCard(self.row_height_card)

        # add cards to group
        self.setupLayout()