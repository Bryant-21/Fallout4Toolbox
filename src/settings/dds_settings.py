from qfluentwidgets import SwitchSettingCard

from src.settings.generic_settings import GenericSettings
from src.utils.appconfig import cfg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QFileDialog
from qfluentwidgets import FluentIcon as FIF, SettingCardGroup, isDarkTheme, PushSettingCard, InfoBar, HyperlinkCard, \
    CustomColorSettingCard, OptionsSettingCard
from qfluentwidgets import ScrollArea, ExpandLayout

from src.utils.appconfig import cfg, HELP_URL, NEXUS_URL, VERSION, YEAR, AUTHOR, KOFI_URL, DISCORD_URL
from src.utils.cards import SpinSettingCard
from src.utils.icons import CustomIcons

class DDSSettings(GenericSettings):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.sc_per_size = SwitchSettingCard(
            icon=FIF.TRANSPARENT,
            title=self.tr("Create per-size subfolders"),
            configItem=cfg.per_size_cfg
        )
        self.sc_no_upscale = SwitchSettingCard(
            icon=FIF.TRANSPARENT,
            title=self.tr("Don't upscale smaller images (copy as-is)"),
            configItem=cfg.no_upscale_cfg
        )
        self.sc_mips = SwitchSettingCard(
            icon=FIF.TRANSPARENT,
            title=self.tr("Generate mipmaps"),
            configItem=cfg.mips_cfg
        )
        self.sc_bc3 = SwitchSettingCard(
            icon=FIF.TRANSPARENT,
            title=self.tr("Convert BC7 to BC3 (linear)"),
            configItem=cfg.bc3_cfg
        )

        # Downscale method selection
        self.sc_downscale_method = OptionsSettingCard(
            cfg.dds_downscale_method,
            CustomIcons.RESCALE.icon(stroke=True),
            self.tr("Downscale Method"),
            self.tr("Choose how images are resized when downscaling"),
            texts=["texconv", "nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"],
            parent=self
        )

        self.__initWidget()

    def __initWidget(self):
        # add cards to group

        self.settings_group.addSettingCard(self.sc_per_size)
        self.settings_group.addSettingCard(self.sc_no_upscale)
        self.settings_group.addSettingCard(self.sc_mips)
        self.settings_group.addSettingCard(self.sc_bc3)
        self.settings_group.addSettingCard(self.sc_downscale_method)

        self.setupLayout()