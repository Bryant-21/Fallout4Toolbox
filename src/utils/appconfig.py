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
    texconv_file = ConfigItem("settings", "texconv", "Please Select a texconv.exe location", FileValidator("exe"))
    threads_default = max(1, os.cpu_count() or 1)
    threads_cfg = RangeConfigItem("settings", "threads", threads_default, RangeValidator(1, os.cpu_count()))

    #DDS
    sizes_cfg = ConfigItem("dds_resizer", "sizes", "1024,2048")
    ignore_cfg = ConfigItem("dds_resizer", "ignore_dirs", "")
    per_size_cfg = ConfigItem("dds_resizer", "per_size_subfolders", True, BoolValidator())
    no_upscale_cfg = ConfigItem("dds_resizer", "no_upscale", True, BoolValidator())
    mips_cfg = ConfigItem("dds_resizer", "generate_mips", True, BoolValidator())
    bc3_cfg = ConfigItem("dds_resizer", "convert_to_bc3", False, BoolValidator())

    #Palette
    ci_default_quant_method = OptionsConfigItem("palette", "default_quantization_method",
                                                     default=QuantAlgorithm.libimagequant,
                                                     validator=OptionsValidator(QuantAlgorithm),
                                                     serializer=EnumSerializer(QuantAlgorithm))
    ci_default_palette_size = OptionsConfigItem("palette", "default_palette_size", 256,
                                                 OptionsValidator([256, 128, 64, 32]))
    ci_default_working_res = OptionsConfigItem("palette", "default_working_resolution",
                                                    default=ResType.Original,
                                                    validator=OptionsValidator(ResType),
                                                    serializer=EnumSerializer(ResType))
    ci_produce_color_report = ConfigItem("palette", "produce_color_report", False, BoolValidator())
    ci_produce_metadata_json = ConfigItem("palette", "produce_metadata_json", False, BoolValidator())
    ci_palette_row_height = RangeConfigItem("palette", "palette_row_height", 2, RangeValidator(2, 8))
    ci_palette_color_iteration = RangeConfigItem("palette", "palette_color_iteration", 1000, RangeValidator(100, 2000))
    ci_suffix = ConfigItem("palette", "suffix", "_d")
    ci_exclude = ConfigItem("palette", "exclude", "")
    ci_advanced_quant = ConfigItem("palette", "advanced_quant", False, BoolValidator())
    ci_include_subdirs = ConfigItem("palette", "include_subdirs", True, BoolValidator())
    ci_group_name = ConfigItem("palette", "group_name", "")
    ci_lower_quant_factor = OptionsConfigItem("palette", "lower_quant_factor", 1.0, OptionsValidator([1.0, 0.5]))

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
    data_root_cfg = ConfigItem("uvpad", "data_root", "")
    textures_dir_cfg = ConfigItem("uvpad", "textures_dir", "")
    output_dir_cfg = ConfigItem("uvpad", "output_dir", "")
    mip_flooding = ConfigItem("uvpad", "mip_flooding", False, BoolValidator())
    color_fill = ConfigItem("uvpad", "color_fill", False, BoolValidator())

    #Convert To Palette
    base_palette_cfg = ConfigItem("convert", "base_palette", "")
    textures = ConfigItem("convert", "textures", "")
    convert_output_dir_cfg = ConfigItem("convert", "output_dir", "")

YEAR = 2025
AUTHOR = "Bryant21"
VERSION = '1.0.0'
NEXUS_URL = "https://next.nexusmods.com/profile/Bryant21"
HELP_URL = "https://github.com/Bryant-21/Fallout4Toolbox"
FEEDBACK_URL = "https://github.com/Bryant-21/Fallout4Toolbox/issues"
RELEASE_URL = "https://github.com/Bryant-21/Fallout4Toolbox/releases/latest"
KOFI_URL = "https://ko-fi.com/bryant21"
DISCORD_URL = "https://discord.gg/FgKrxdnQdG"

cfg = Config()
qconfig.load(os.path.join(get_app_root(), 'config', 'config.json'), cfg)


