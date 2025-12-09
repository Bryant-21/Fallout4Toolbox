from qfluentwidgets import OptionsSettingCard, SwitchSettingCard, RangeSettingCard

from src.settings.generic_settings import GenericSettings
from src.utils.appconfig import cfg
from src.utils.cards import SpinSettingCard, RadioSettingCard
from src.utils.icons import CustomIcons


class PaletteSettings(GenericSettings):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.palette_size_card = RadioSettingCard(
            cfg.ci_default_palette_size,
            CustomIcons.WIDTH.icon(),
            self.tr("Palette Size"),
            self.tr("Width of the Palette"),
            texts=["256", "128", "64", "32"],
            parent=self
        )

        self.quantize_size_card = RadioSettingCard(
            cfg.ci_default_quant_size,
            CustomIcons.WIDTH.icon(),
            self.tr("Quantize Amount"),
            self.tr("Number of Colors to Quantize image down"),
            texts=["256", "192", "128", "96", "64", "32"],
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

        self.semi_transparent_mode_card = OptionsSettingCard(
            cfg.ci_semi_transparent_mode,
            CustomIcons.QUANT.icon(),
            self.tr("Semi-Transparent Handling"),
            self.tr("How to treat semi-transparent pixels when generating palettes"),
            texts=[
                "mask - Treat semi-transparent as transparent (remove)",
                "nearest_fill - Replace with nearest opaque color then mask",
                "premultiply_snap - Premultiply RGB then snap alpha to 0/255"
            ],
            parent=self
        )

        self.upscale_palette_switch = SwitchSettingCard(
            icon=CustomIcons.RESCALE.icon(stroke=True),
            title=self.tr("Upscale Palette To 256"),
            content=self.tr("If palette has < 256 colors, interpolate and smooth to 256 to reduce harsh transitions"),
            configItem=cfg.ci_palette_upscale_to_256
        )

        self.greyscale_mapping_strategy_card = OptionsSettingCard(
            cfg.ci_greyscale_mapping_strategy,
            CustomIcons.QUANT.icon(),
            self.tr("Greyscale Mapping Strategy"),
            self.tr("How pixels are mapped to palette indices within islands"),
            texts=[
                "luminosity - Default brightness-based linear mapping",
                "guard_bands_quantile - Hybrid: guard bands + quantile (recommended for Fallout 4)",
                "quantile - Even distribution across palette range",
                "guard_bands - Simple guard bands with luminosity",
                "nearest_neighbor_reserve - Reserve first/last pixels, fill with nearest neighbor (simple approach)",
                "color_clustering - Hue-based (preserves color identity)",
                "perceptual - CIE Lab L* perceptual brightness",
                "reverse_luminosity - Inverted (dark=high, bright=low)",
                "alternating_luminosity - Alternates direction per island (island 0: high-to-low, island 1: low-to-high, etc.)"
            ],
            parent=self
        )

        self.guard_band_width_card = SpinSettingCard(
            cfg.ci_guard_band_width,
            CustomIcons.WIDTH.icon(),
            self.tr("Guard Band Width"),
            self.tr("Number of boundary indices to reserve (0-2) for interpolation smoothing")
        )

        self.palette_smooth_method_card = OptionsSettingCard(
            cfg.ci_palette_smooth_method,
            CustomIcons.QUANT.icon(),
            self.tr("Palette Smoothing Method"),
            self.tr("Smooth harsh color transitions in palette to reduce in-game interpolation artifacts"),
            texts=[
                "none - No smoothing (default)",
                "gaussian - Gaussian blur: smooth based on spatial proximity",
                "median - Median filter: preserves edges better while smoothing",
                "bilateral - Bilateral: edge-preserving smoothing (best quality, slower)"
            ],
            parent=self
        )

        self.palette_smooth_strength_card = RangeSettingCard(
            cfg.ci_palette_smooth_strength,
            CustomIcons.QUANT.icon(),
            self.tr("Palette Smoothing Strength"),
            self.tr("Control intensity of palette smoothing (0.0 = none, 1.0 = maximum)"),
            parent=self
        )

        self.fix_scaled_uv = SwitchSettingCard(
            configItem=cfg.scale_uvs,
            title=self.tr("Scale UV"),
            icon=CustomIcons.FIT.icon(),
            content=self.tr("Sometimes needed, not sure why."),
        )


        self.__initWidget()

    def __initWidget(self):
        self.settings_group.addSettingCard(self.quantize_size_card)
        self.settings_group.addSettingCard(self.palette_size_card)
        self.settings_group.addSettingCard(self.method_card)
        self.settings_group.addSettingCard(self.row_height_card)
        self.settings_group.addSettingCard(self.filter_type_card)
        self.settings_group.addSettingCard(self.semi_transparent_mode_card)
        self.settings_group.addSettingCard(self.greyscale_mapping_strategy_card)
        self.settings_group.addSettingCard(self.guard_band_width_card)
        self.settings_group.addSettingCard(self.palette_smooth_method_card)
        self.settings_group.addSettingCard(self.palette_smooth_strength_card)
        self.settings_group.addSettingCard(self.upscale_palette_switch)
        self.settings_group.addSettingCard(self.fix_scaled_uv)

        # add cards to group
        self.setupLayout()