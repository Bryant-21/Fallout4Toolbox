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

        self.filter_type_card = OptionsSettingCard(
            cfg.ci_palette_filter_type,
            CustomIcons.QUANT.icon(),
            self.tr("Palette Filter Type"),
            self.tr("How colors are sampled when applying palette to greyscale"),
            texts=[
                "linear - Smooth color interpolation (default)",
                "nearest - Exact colors, no blending (better for game LUTs)",
                "cubic - Smoother transitions, reduces harsh jumps",
                "gaussian - Blurs transitions to reduce banding",
                "cubic_gaussian - Cubic + gaussian for smoothest results"
            ],
            parent=self
        )

        self.upscale_palette_switch = SwitchSettingCard(
            icon=CustomIcons.RESCALE.icon(stroke=True),
            title=self.tr("Upscale Palette To 256"),
            content=self.tr("If palette has < 256 colors, interpolate and smooth to 256 to reduce harsh transitions"),
            configItem=cfg.ci_palette_upscale_to_256
        )

        self.__initWidget()

    def __initWidget(self):

        self.settings_group.addSettingCard(self.palette_size_card)
        self.settings_group.addSettingCard(self.method_card)
        self.settings_group.addSettingCard(self.row_height_card)
        self.settings_group.addSettingCard(self.filter_type_card)
        self.settings_group.addSettingCard(self.upscale_palette_switch)

        # add cards to group
        self.setupLayout()