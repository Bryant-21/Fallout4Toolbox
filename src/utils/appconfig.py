import os
from enum import Enum

from qfluentwidgets import ConfigItem, QConfig, FolderValidator, qconfig, ConfigValidator, RangeValidator, \
    RangeConfigItem, BoolValidator, EnumSerializer, OptionsValidator, OptionsConfigItem, ColorConfigItem

from src.utils.filesystem_utils import get_app_root


class FileValidator(ConfigValidator):

    def __init__(self, allowed_file_types=None):
        if allowed_file_types is None:
            allowed_file_types = ['csv', 'txt']
        self.allowed_file_types = allowed_file_types

    """ File validator """

    def validate(self, value):
        if os.path.exists(value) and os.path.isfile(value):
            _, file_extension = os.path.splitext(value)
            return file_extension in self.allowed_file_types

        return False


class QuantAlgorithm(Enum):
    median_cut= "median_cut"
    max_coverage= "max_coverage"
    fast_octree= "fast_octree"
    libimagequant = "libimagequant"
    kmeans_adaptive= "kmeans_adaptive"
    uniform= "uniform"

class ResType(Enum):
    Original= "Original"
    FOUR_K = 4096
    TWO_K = 2048
    ONE_K = 1024
    FIVE_TWELVE = 512



class Config(QConfig):

    #shared
    threads_cfg = RangeConfigItem("settings", "threads", max(1, os.cpu_count() or 1), RangeValidator(1, os.cpu_count()))

    #DDS
    sizes_cfg = ConfigItem("dds_resizer", "sizes", "1024,2048")
    ignore_cfg = ConfigItem("dds_resizer", "ignore_dirs", "")
    per_size_cfg = ConfigItem("dds_resizer", "per_size_subfolders", True, BoolValidator())
    no_upscale_cfg = ConfigItem("dds_resizer", "no_upscale", True, BoolValidator())
    mips_cfg = ConfigItem("dds_resizer", "generate_mips", True, BoolValidator())
    bc3_cfg = ConfigItem("dds_resizer", "convert_to_bc3", False, BoolValidator())
    dds_downscale_method = OptionsConfigItem(
        "dds_resizer",
        "downscale_method",
        "texconv",
        OptionsValidator(["texconv", "nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"])
    )

    #Palette
    ci_default_quant_method = OptionsConfigItem("palette", "default_quantization_method",
                                                     default=QuantAlgorithm.libimagequant,
                                                     validator=OptionsValidator(QuantAlgorithm),
                                                     serializer=EnumSerializer(QuantAlgorithm))
    ci_default_palette_size = OptionsConfigItem("palette", "default_palette_size", 128,
                                                 OptionsValidator([256, 128, 64, 32]))
    ci_default_working_res = OptionsConfigItem("palette", "default_working_resolution",
                                                    default=ResType.Original,
                                                    validator=OptionsValidator(ResType),
                                                    serializer=EnumSerializer(ResType))
    # Auto grouping sensitivity (1-200): >100 enables extra-lenient matching for mixed regions
    ci_grouping_threshold = RangeConfigItem("palette", "grouping_threshold", 33, RangeValidator(1, 200))


    ci_produce_color_report = ConfigItem("palette", "produce_color_report", False, BoolValidator())
    ci_produce_metadata_json = ConfigItem("palette", "produce_metadata_json", False, BoolValidator())
    ci_palette_row_height = RangeConfigItem("palette", "palette_row_height", 4, RangeValidator(4, 16))
    ci_palette_color_iteration = RangeConfigItem("palette", "palette_color_iteration", 1000, RangeValidator(100, 2000))
    ci_suffix = ConfigItem("palette", "suffix", "_d")
    ci_exclude = ConfigItem("palette", "exclude", "")
    ci_advanced_quant = ConfigItem("palette", "advanced_quant", False, BoolValidator())
    ci_include_subdirs = ConfigItem("palette", "include_subdirs", True, BoolValidator())
    ci_group_name = ConfigItem("palette", "group_name", "")
    ci_lower_quant_factor = OptionsConfigItem("palette", "lower_quant_factor", 1.0, OptionsValidator([1.0, 0.5]))
    ci_use_faster_sort = ConfigItem("palette", "faster_color_sort", True, BoolValidator())
    ci_replace_existing = ConfigItem("palette", "replace_existing", False, BoolValidator())
    ci_single_palette = ConfigItem("palette", "single_palette", True, BoolValidator())
    ci_palette_use_original_colors = ConfigItem("palette", "palette_use_original_colors", True, BoolValidator())
    ci_palette_original_max_de = RangeConfigItem("palette", "palette_original_max_de", 2.0, RangeValidator(0.0, 20.0))
    ci_extra_logging = ConfigItem("palette", "extra_logging", False, BoolValidator())
    # Greyscale post-processing
    ci_greyscale_post_enable = ConfigItem("palette", "greyscale_post_enable", False, BoolValidator())
    ci_greyscale_post_method = OptionsConfigItem(
        "palette", "greyscale_post_method", "median",
        OptionsValidator(["none", "median", "gaussian", "dither", "median_dither", "gaussian_dither"]))
    ci_greyscale_median_size = RangeConfigItem("palette", "greyscale_median_size", 3, RangeValidator(1, 9))
    ci_greyscale_blur_radius = RangeConfigItem("palette", "greyscale_blur_radius", 6, RangeValidator(1, 30))
    ci_greyscale_dither_amount = RangeConfigItem("palette", "greyscale_dither_amount", 2, RangeValidator(0, 100))
    ci_greyscale_post_apply_to_textures = ConfigItem("palette", "greyscale_post_apply_to_textures", False, BoolValidator())
    ci_quantize_post_enable = ConfigItem("palette", "quantize_post_enable", True, BoolValidator())
    # Palette upscaling
    ci_palette_upscale_to_256 = ConfigItem("palette", "palette_upscale_to_256", False, BoolValidator())
    ci_palette_upscale_sigma = RangeConfigItem("palette", "palette_upscale_sigma", 10, RangeValidator(0, 100))
    # Quantized image dithering
    ci_quantize_dither_enable = ConfigItem("palette", "quantize_dither_enable", False, BoolValidator())
    # Palette filtering type: "linear" interpolates colors smoothly, "nearest" preserves exact colors
    ci_palette_filter_type = OptionsConfigItem("palette", "palette_filter_type", "linear",
                                                OptionsValidator(["linear", "nearest", "cubic", "gaussian", "cubic_gaussian"]))

    #theme
    themeColor = ColorConfigItem("QFluentWidgets", "ThemeColor", '#ffa11d', restart=True)
    dpiScale = OptionsConfigItem(
        "MainWindow", "DpiScale", 1, OptionsValidator([0.75, 0.95, 1, 1.1, 1.25, 1.5, 1.75, 2, "Auto"]), restart=True)

    #matfiles
    tex_diffuse_cfg = ConfigItem("material", "tex_diffuse", True, BoolValidator())
    tex_normal_cfg = ConfigItem("material", "tex_normal", True, BoolValidator())
    tex_smoothspec_cfg = ConfigItem("material", "tex_smoothspec", True, BoolValidator())
    tex_greyscale_cfg = ConfigItem("material", "tex_greyscale", False, BoolValidator())
    tex_envmap_cfg = ConfigItem("material", "tex_envmap", False, BoolValidator())
    tex_glow_cfg = ConfigItem("material", "tex_glow", False, BoolValidator())
    tex_inner_cfg = ConfigItem("material", "tex_inner", False, BoolValidator())
    tex_wrinkles_cfg = ConfigItem("material", "tex_wrinkles", False, BoolValidator())
    bgsm_cfg = ConfigItem("material", "bgsm", True, BoolValidator())
    bgem_cfg = ConfigItem("material", "bgem", False, BoolValidator())
    input_dir = ConfigItem("material", "input_dir", "")
    output_root = ConfigItem("material", "output_root", "")
    folders_cfg = ConfigItem("material", "folders", "")
    excludes_cfg = ConfigItem("material", "excludes", "")
    grayscale_to_palette_scale_cfg = RangeConfigItem("material", "grayscale_to_palette_scale", 0.0, RangeValidator(0.0, 10.0))
    clean_output_cfg = ConfigItem("material", "clean_output", False, BoolValidator())


    #esp_renamer
    author_cfg = ConfigItem("esp_renamer", "author", "")
    description_cfg = ConfigItem("esp_renamer", "description", "")
    match_text_cfg = ConfigItem("esp_renamer", "match_text", "")

    #Subgraph Maker
    cfg_human = ConfigItem("subgraph_maker", "human", True, BoolValidator())
    cfg_power = ConfigItem("subgraph_maker", "powerarmor", True, BoolValidator())
    cfg_mutant = ConfigItem("subgraph_maker", "supermutant", True, BoolValidator())
    cfg_target_anim = ConfigItem("subgraph_maker", "target_anim", "")
    cfg_target_folder = ConfigItem("subgraph_maker", "target_folder", "")
    cfg_target_prepend = ConfigItem("subgraph_maker", "target_prepend", True, BoolValidator())

    #NIF
    data_root_cfg = ConfigItem("nif", "data_root", "")
    last_open_nif = ConfigItem("nif", "last_open_nif", "")
    do_ai_upscale = ConfigItem("nif", "ai_upscaler", False, BoolValidator())
    textures_dir_cfg = ConfigItem("nif", "textures_dir", "")
    output_dir_cfg = ConfigItem("nif", "output_dir", "")
    mip_flooding = ConfigItem("nif", "mip_flooding", False, BoolValidator())
    color_fill = ConfigItem("nif", "color_fill", False, BoolValidator())
    scale_uvs = ConfigItem("nif", "scale_uvs", False, BoolValidator())

    #Convert To Palette
    base_palette_cfg = ConfigItem("convert", "base_palette", "")
    textures = ConfigItem("convert", "textures", "")
    convert_output_dir_cfg = ConfigItem("convert", "output_dir", "")
    convert_dir_cfg = ConfigItem("convert", "input_dir", "")


    mipflood_check = ConfigItem("validation", "mipflood", True, BoolValidator())

    #Upscaler
    upscale_normals_cfg = ConfigItem("upscaler", "normals", "4x-Normal-RG0-BC7", OptionsValidator([
        "4x-Normal-RG0-BC1",
        "4x-Normal-RG0-BC7",
        "4x-Normal-RG0"
    ]))

    upscale_textures_cfg = ConfigItem("upscaler", "textures", "4x-PBRify_UpscalerV4", OptionsValidator([
        "4x-PBRify_UpscalerV4",
        "4xTextures_GTAV_rgt-s_dither",
        "4x-PBRify_UpscalerSIR-M_V2",
        "UltraSharpV2",
        "4xNomosWebPhoto_RealPLKSR"
    ]))

YEAR = 2025
AUTHOR = "Bryant21"
VERSION = '2.0.0'
NEXUS_URL = "https://next.nexusmods.com/profile/Bryant21"
HELP_URL = "https://github.com/Bryant-21/Fallout4Toolbox"
FEEDBACK_URL = "https://github.com/Bryant-21/Fallout4Toolbox/issues"
RELEASE_URL = "https://github.com/Bryant-21/Fallout4Toolbox/releases/latest"
KOFI_URL = "https://ko-fi.com/bryant21"
DISCORD_URL = "https://discord.gg/FgKrxdnQdG"
TEXCONV_EXE = os.path.join(get_app_root(), "resource", "texconv.exe")

cfg = Config()
qconfig.load(os.path.join(get_app_root(), 'config', 'config.json'), cfg)


