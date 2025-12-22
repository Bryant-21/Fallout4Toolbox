from qfluentwidgets import OptionsSettingCard, SwitchSettingCard

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
                "cubic_gaussian - Cubic + gaussian for smoothest results",
                "anchored_linear"
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
                "premultiply_snap - Premultiply RGB then snap alpha to 0/255",
                "none - For Testing"
            ],
            parent=self
        )

        self.upscale_palette_switch = SwitchSettingCard(
            icon=CustomIcons.RESCALE.icon(stroke=True),
            title=self.tr("Upscale Palette To 256"),
            content=self.tr("If palette has < 256 colors, interpolate and smooth to 256 to reduce harsh transitions"),
            configItem=cfg.ci_palette_upscale_to_256
        )

        self.island_prequant_switch = SwitchSettingCard(
            icon=CustomIcons.QUANT.icon(),
            title=self.tr("Pre-Quantize Islands To Island Size"),
            content=self.tr("When enabled, each island's colors are quantized to its available slots before grayscale/palette. Transparent areas are re-masked after quantization."),
            configItem=cfg.ci_island_prequant_enable
        )

        self.island_autobalance_switch = SwitchSettingCard(
            icon=CustomIcons.WIDTH.icon(),
            title=self.tr("Auto Balance Islands"),
            content=self.tr("Shift palette index boundaries so under-used islands donate slots to overfull islands before quantizing."),
            configItem=cfg.ci_island_autobalance_enable
        )

        self.auto_preview_switch = SwitchSettingCard(
            icon=CustomIcons.PREVIEW_FILE.icon() if hasattr(CustomIcons, 'PREVIEW_FILE') else CustomIcons.IMAGE.icon(),
            title=self.tr("Auto Preview"),
            content=self.tr("After generating both images, automatically open Palette Preview and load them"),
            configItem=cfg.ci_auto_preview
        )

        # self.greyscale_mapping_strategy_card = OptionsSettingCard(
        #     cfg.ci_greyscale_mapping_strategy,
        #     CustomIcons.QUANT.icon(),
        #     self.tr("Greyscale Mapping Strategy"),
        #     self.tr("How pixels are mapped to palette indices within islands"),
        #     texts=[
        #         "luminosity - Default brightness-based linear mapping",
        #         "guard_bands_quantile - Hybrid: guard bands + quantile (recommended for Fallout 4)",
        #         "quantile - Even distribution across palette range",
        #         "guard_bands - Simple guard bands with luminosity",
        #         "nearest_neighbor_reserve - Reserve first/last pixels, fill with nearest neighbor (simple approach)",
        #         "color_clustering - Hue-based (preserves color identity)",
        #         "perceptual - CIE Lab L* perceptual brightness",
        #         "reverse_luminosity - Inverted (dark=high, bright=low)",
        #         "alternating_luminosity - Alternates direction per island (island 0: high-to-low, island 1: low-to-high, etc.)",
        #         "smoothed_quantile - Smoothed ECDF quantile with guard bands and linear tempering",
        #         "tempered_quantile - Blend quantile with linear to soften transitions",
        #         "spline_quantile - Monotone spline through data quantiles (smooth, rank-preserving)"
        #     ],
        #     parent=self
        # )

        # Collision resolver controls
        # self.collision_resolver_switch = SwitchSettingCard(
        #     icon=CustomIcons.QUANT.icon(),
        #     title=self.tr("Resolve Grayscale Collisions"),
        #     content=self.tr("Try small randomized tone-curve tweaks (N tries) to reduce grayscale collisions while staying in island ranges"),
        #     configItem=cfg.ci_enable_collision_resolver
        # )

        # self.collision_resolver_strategy = OptionsSettingCard(
        #     cfg.ci_collision_resolver_strategy,
        #     CustomIcons.QUANT.icon(),
        #     self.tr("Collision Resolver Strategy"),
        #     self.tr("How to search: grayscale curve, per-channel gamma, RGB weight mix, or hybrid"),
        #     texts=[
        #         "gray_curve - Adjust grayscale tone curve (default)",
        #         "per_channel_gamma - Tiny gamma/offset per R,G,B before luminance",
        #         "rgb_weight_mix - Slightly vary R/G/B luminance weights",
        #         "hybrid - Randomly mix all strategies"
        #     ],
        #     parent=self
        # )
        # self.collision_resolver_tries = SpinSettingCard(
        #     cfg.ci_collision_resolver_tries,
        #     CustomIcons.QUANT.icon(),
        #     self.tr("Collision Resolver: Tries"),
        #     self.tr("Number of random attempts (1–100)")
        # )
        # self.collision_resolver_per_island = SwitchSettingCard(
        #     icon=CustomIcons.QUANT.icon(),
        #     title=self.tr("Per-Island Fine Tuning"),
        #     content=self.tr("Allow tiny per-island gamma/offset tweaks (keeps mapping monotone)"),
        #     configItem=cfg.ci_collision_resolver_per_island
        # )
        # self.collision_resolver_naturalness = DoubleSpinSettingCard(
        #     cfg.ci_collision_resolver_naturalness_w,
        #     CustomIcons.QUANT.icon(),
        #     self.tr("Collision Resolver: Naturalness Weight (0..1)")
        # )
        # self.collision_resolver_collision_w = DoubleSpinSettingCard(
        #     cfg.ci_collision_resolver_collision_w,
        #     CustomIcons.QUANT.icon(),
        #     self.tr("Collision Resolver: Collision Weight (0..10)")
        # )

        self.guard_band_width_card = SpinSettingCard(
            cfg.ci_guard_band_width,
            CustomIcons.WIDTH.icon(),
            self.tr("Guard Band Width"),
            self.tr("Number of boundary indices to reserve (0-2) for interpolation smoothing")
        )

        # # Smoothed quantile controls
        # self.smoothed_bins_card = SpinSettingCard(
        #     cfg.ci_smoothed_quantile_bins,
        #     CustomIcons.WIDTH.icon(),
        #     self.tr("Smoothed Quantile: Histogram Bins"),
        #     self.tr("Number of bins used to compute the ECDF (16–2048)")
        # )
        # self.smoothed_sigma_card = DoubleSpinSettingCard(
        #     cfg.ci_smoothed_quantile_sigma,
        #     CustomIcons.QUANT.icon(),
        #     self.tr("Smoothed Quantile: Gaussian Sigma (bins)")
        # )
        # self.smoothed_alpha_card = RangeSettingCard(
        #     cfg.ci_smoothed_quantile_alpha,
        #     CustomIcons.QUANT.icon(),
        #     self.tr("Smoothed Quantile: Linear Blend %"),
        #     self.tr("Blend toward linear mapping (0 = pure quantile, 100 = pure linear)")
        # )
        #
        # # Tempered quantile controls
        # self.tempered_alpha_card = RangeSettingCard(
        #     cfg.ci_tempered_quantile_alpha,
        #     CustomIcons.QUANT.icon(),
        #     self.tr("Tempered Quantile: Linear Blend %"),
        #     self.tr("Blend toward linear mapping (0 = pure quantile, 100 = pure linear)")
        # )
        #
        # # Spline quantile controls
        # self.spline_profile_card = OptionsSettingCard(
        #     cfg.ci_spline_profile,
        #     CustomIcons.QUANT.icon(),
        #     self.tr("Spline Quantile: Profile"),
        #     self.tr("Shape of output spacing inside island range"),
        #     texts=[
        #         "even - Even spacing (default)",
        #         "compressed_ends - Slightly compress extremes",
        #         "expanded_ends - Slightly expand extremes"
        #     ],
        #     parent=self
        # )
        # self.spline_gamma_card = DoubleSpinSettingCard(
        #     cfg.ci_spline_gamma,
        #     CustomIcons.QUANT.icon(),
        #     self.tr("Spline Quantile: Gamma")
        # )

        # self.palette_smooth_method_card = OptionsSettingCard(
        #     cfg.ci_palette_smooth_method,
        #     CustomIcons.QUANT.icon(),
        #     self.tr("Palette Smoothing Method"),
        #     self.tr("Smooth harsh color transitions in palette to reduce in-game interpolation artifacts"),
        #     texts=[
        #         "none - No smoothing (default)",
        #         "gaussian - Gaussian blur: smooth based on spatial proximity",
        #         "median - Median filter: preserves edges better while smoothing",
        #         "bilateral - Bilateral: edge-preserving smoothing (best quality, slower)"
        #     ],
        #     parent=self
        # )
        #
        # self.palette_smooth_strength_card = RangeSettingCard(
        #     cfg.ci_palette_smooth_strength,
        #     CustomIcons.QUANT.icon(),
        #     self.tr("Palette Smoothing Strength"),
        #     self.tr("Control intensity of palette smoothing (0.0 = none, 1.0 = maximum)"),
        #     parent=self
        # )

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
        # self.settings_group.addSettingCard(self.greyscale_mapping_strategy_card)
        # self.settings_group.addSettingCard(self.collision_resolver_switch)
        # self.settings_group.addSettingCard(self.collision_resolver_strategy)
        # self.settings_group.addSettingCard(self.collision_resolver_tries)
        # self.settings_group.addSettingCard(self.collision_resolver_per_island)
        # self.settings_group.addSettingCard(self.collision_resolver_naturalness)
        # self.settings_group.addSettingCard(self.collision_resolver_collision_w)
        # self.settings_group.addSettingCard(self.guard_band_width_card)
        # self.settings_group.addSettingCard(self.smoothed_bins_card)
        # self.settings_group.addSettingCard(self.smoothed_sigma_card)
        # self.settings_group.addSettingCard(self.smoothed_alpha_card)
        # self.settings_group.addSettingCard(self.tempered_alpha_card)
        # self.settings_group.addSettingCard(self.spline_profile_card)
        # self.settings_group.addSettingCard(self.spline_gamma_card)
        # self.settings_group.addSettingCard(self.palette_smooth_method_card)
        # self.settings_group.addSettingCard(self.palette_smooth_strength_card)
        self.settings_group.addSettingCard(self.upscale_palette_switch)
        self.settings_group.addSettingCard(self.island_prequant_switch)
        self.settings_group.addSettingCard(self.island_autobalance_switch)
        self.settings_group.addSettingCard(self.auto_preview_switch)
        self.settings_group.addSettingCard(self.fix_scaled_uv)

        # add cards to group
        self.setupLayout()