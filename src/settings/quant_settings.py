from src.settings.generic_settings import GenericSettings
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QFileDialog
from qfluentwidgets import FluentIcon as FIF, SettingCardGroup, isDarkTheme, PushSettingCard, InfoBar, HyperlinkCard, \
    CustomColorSettingCard, OptionsSettingCard, SwitchSettingCard
from qfluentwidgets import ScrollArea, ExpandLayout

from src.utils.appconfig import cfg, HELP_URL, NEXUS_URL, VERSION, YEAR, AUTHOR, KOFI_URL, DISCORD_URL
from src.utils.cards import SpinSettingCard, RadioSettingCard
from src.utils.icons import CustomIcons

class QuantSettings(GenericSettings):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.palette_size_card = RadioSettingCard(
            cfg.ci_default_palette_size,
            CustomIcons.WIDTH.icon(),
            self.tr("Palette Size"),
            self.tr("Number of colors to quantize to"),
            texts=["128", "64", "32"],
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

        # Quantize step dithering
        self.quantize_dither_enable = SwitchSettingCard(
            icon=CustomIcons.SPARK.icon(),
            title=self.tr("Apply Dither To Quantized Image"),
            content=self.tr("Enable Floyd–Steinberg dithering during quantization for fewer banding artifacts"),
            configItem=cfg.ci_quantize_dither_enable
        )


        # self.advanced_quant= SwitchSettingCard(
        #     icon=FIF.CUT,
        #     title=self.tr("Oversample Colors - Uses LAB Reduction to keep more unique colors"),
        #     configItem=cfg.ci_advanced_quant
        # )


        self.__initWidget()

    def __initWidget(self):
        self.settings_group.addSettingCard(self.palette_size_card)
        self.settings_group.addSettingCard(self.method_card)
        self.settings_group.addSettingCard(self.quantize_dither_enable)
        # self.settings_group.addSettingCard(self.advanced_quant)
        # add cards to group
        self.setupLayout()