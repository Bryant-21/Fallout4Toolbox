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

        self.quantize_post_process = SwitchSettingCard(
            icon=CustomIcons.PALETTE.icon(),
            title=self.tr("Apply To Post Processing to Textures"),
            content=self.tr("Smooth out the colors to try and get a better palette"),
            configItem=cfg.ci_quantize_post_enable
        )

        self.advanced_color_sort= SwitchSettingCard(
            icon=CustomIcons.SPARK.icon(),
            title=self.tr("Use Faster Color Sorting"),
            content=self.tr("80% as good but much faster, disable to get maximum quality"),
            configItem=cfg.ci_use_faster_sort
        )

        self.prefer_original_colors = SwitchSettingCard(
            icon=CustomIcons.PALETTE.icon(),
            title=self.tr("Prefer Original Colors"),
            content=self.tr("With multiple images there can be small colors differences, we prefer original"),
            configItem=cfg.ci_palette_use_original_colors
        )

        self.prefer_original_colors_strength = SpinSettingCard(
            cfg.ci_palette_original_max_de,
            CustomIcons.STEPS.icon(),
            self.tr("Prefer Original Colors Strength"),
            self.tr("2 to 20"), step=1
        )

        self.greyscale_post_enabled = SwitchSettingCard(
            icon=CustomIcons.PALETTE.icon(),
            title=self.tr("Apply Dither To Greyscale"),
            content=self.tr("Makes the image a little less harsh"),
            configItem=cfg.ci_greyscale_post_enable
        )

        self.greyscale_post_method = OptionsSettingCard(
            cfg.ci_greyscale_post_method,
            CustomIcons.RESCALE.icon(stroke=True),
            self.tr("Dither Methods"),
            self.tr("Different supported methods"),
            texts=["none", "median", "gaussian", "dither", "median_dither", "gaussian_dither"],
            parent=self
        )

        # Extra greyscale post-process controls
        self.greyscale_median_size = SpinSettingCard(
            cfg.ci_greyscale_median_size,
            CustomIcons.STEPS.icon(),
            self.tr("Median Filter Size"),
            self.tr("3 to 9"), step=1
        )

        self.greyscale_blur_radius = RangeSettingCardScaled(
            cfg.ci_greyscale_blur_radius,
            CustomIcons.RESCALE.icon(stroke=True),
            self.tr("Gaussian Blur Radius"),
            self.tr("0.1 to 3.0"),
            step=1,
            scale=10,
        )

        self.greyscale_dither_amount = RangeSettingCardScaled(
            cfg.ci_greyscale_dither_amount,
            CustomIcons.SPARK.icon(),
            self.tr("Dither Amount"),
            self.tr("0.0 to 1.0"),
            step=1
        )

        self.greyscale_post_apply_to_textures = SwitchSettingCard(
            icon=CustomIcons.PALETTE.icon(),
            title=self.tr("Apply To Greyscale Textures"),
            content=self.tr("Also apply the post-process to greyscale conversion textures"),
            configItem=cfg.ci_greyscale_post_apply_to_textures
        )

        # Palette upscaling
        self.upscale_palette_switch = SwitchSettingCard(
            icon=CustomIcons.RESCALE.icon(stroke=True),
            title=self.tr("Upscale Palette To 256"),
            content=self.tr("If palette has < 256 colors, interpolate and smooth to 256 to reduce harsh transitions"),
            configItem=cfg.ci_palette_upscale_to_256
        )
        self.upscale_sigma = RangeSettingCardScaled(
            cfg.ci_palette_upscale_sigma,
            CustomIcons.SPARK.icon(),
            self.tr("Upscale Smoothness (Sigma)"),
            self.tr("0.0 to 10.0"),
            step=1,
            scale=10,
        )

        # Quantize step dithering
        self.quantize_dither_enable = SwitchSettingCard(
            icon=CustomIcons.SPARK.icon(),
            title=self.tr("Apply Dither To Quantized Image"),
            content=self.tr("Enable Floyd–Steinberg dithering during quantization for fewer banding artifacts"),
            configItem=cfg.ci_quantize_dither_enable
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
        self.settings_group.addSettingCard(self.quantize_post_process)
        #self.settings_group.addSettingCard(self.advanced_quant)
        self.settings_group.addSettingCard(self.working_res_card)
        self.settings_group.addSettingCard(self.row_height_card)
        self.settings_group.addSettingCard(self.sc_color_report)
        self.settings_group.addSettingCard(self.advanced_color_sort)
        self.settings_group.addSettingCard(self.prefer_original_colors)
        self.settings_group.addSettingCard(self.prefer_original_colors_strength)
        self.settings_group.addSettingCard(self.greyscale_post_enabled)
        self.settings_group.addSettingCard(self.greyscale_post_method)
        self.settings_group.addSettingCard(self.greyscale_median_size)
        self.settings_group.addSettingCard(self.greyscale_blur_radius)
        self.settings_group.addSettingCard(self.greyscale_dither_amount)
        self.settings_group.addSettingCard(self.greyscale_post_apply_to_textures)
        self.settings_group.addSettingCard(self.upscale_palette_switch)
        self.settings_group.addSettingCard(self.upscale_sigma)
        self.settings_group.addSettingCard(self.quantize_dither_enable)

        # add cards to group
        self.setupLayout()