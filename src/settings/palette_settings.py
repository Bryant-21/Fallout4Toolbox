from src.settings.generic_settings import GenericSettings
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QFileDialog
from qfluentwidgets import FluentIcon as FIF, SettingCardGroup, isDarkTheme, PushSettingCard, InfoBar, HyperlinkCard, \
    CustomColorSettingCard, OptionsSettingCard, SwitchSettingCard
from qfluentwidgets import ScrollArea, ExpandLayout

from src.utils.appconfig import cfg, HELP_URL, NEXUS_URL, VERSION, YEAR, AUTHOR, KOFI_URL, DISCORD_URL
from src.utils.cards import SpinSettingCard, RadioSettingCard
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

        # Per-image quantization budget relative to final palette size
        self.lower_quant_factor_card = OptionsSettingCard(
            cfg.ci_lower_quant_factor,
            CustomIcons.QUANT.icon(),
            self.tr("Per-image color budget"),
            self.tr("Quantize each source image to fewer colors than the final palette (e.g., Half: 256 → 128)"),
            texts=[
                "Full (same as final)",
                "Half (50%)"
            ],
            parent=self
        )
        #
        # self.advanced_quant= SwitchSettingCard(
        #     icon=FIF.CUT,
        #     title=self.tr("Oversample Colors - Uses LAB Reduction to keep more unique colors"),
        #     configItem=cfg.ci_advanced_quant
        # )

        self.working_res_card = OptionsSettingCard(
            cfg.ci_default_working_res,
            CustomIcons.RESCALE.icon(stroke=True),
            self.tr("Rescale Images"),
            self.tr("Downscale base and additional images for processing (no upscaling)"),
            texts=["Original", "4k (4096)", "2k (2048)", "1k (1024)", "512"],
            parent=self
        )

        self.interation_card = SpinSettingCard(
            cfg.ci_palette_color_iteration,
            CustomIcons.HEIGHT.icon(),
            self.tr("Gradient Color Pass (more ordered for compression, takes longer)"),
            self.tr("100 to 2000"), step=10
        )

        self.row_height_card = SpinSettingCard(
            cfg.ci_palette_row_height,
            CustomIcons.HEIGHT.icon(),
            self.tr("Palette row height (pixels)"),
            self.tr("2 to 8"), step=2
        )


        self.sc_color_report = SwitchSettingCard(icon=CustomIcons.REPORT.icon(),
                                                 title=self.tr("Color Report Enabled"),
                                                 content = "Detailed Color Report, useful for debugging",
                                                 configItem=cfg.ci_produce_color_report)



        self.__initWidget()

    def __initWidget(self):

        self.settings_group.addSettingCard(self.palette_size_card)
        self.settings_group.addSettingCard(self.interation_card)
        self.settings_group.addSettingCard(self.method_card)
        self.settings_group.addSettingCard(self.lower_quant_factor_card)
        #self.settings_group.addSettingCard(self.advanced_quant)
        self.settings_group.addSettingCard(self.working_res_card)
        self.settings_group.addSettingCard(self.row_height_card)
        self.settings_group.addSettingCard(self.sc_color_report)

        # add cards to group
        self.setupLayout()