from qfluentwidgets import FluentIcon as FIF, SwitchSettingCard

from src.settings.generic_settings import GenericSettings
from src.utils.appconfig import cfg


class MatFilesSettings(GenericSettings):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.cb_bgsm_card = SwitchSettingCard(
            icon=FIF.TRANSPARENT,
            title="BGSM",
            configItem=cfg.bgsm_cfg
        )
        self.cb_bgem_card = SwitchSettingCard(
            icon=FIF.TRANSPARENT,
            title="BGEM",
            configItem=cfg.bgem_cfg
        )

        self.sc_diffuse = SwitchSettingCard(icon=FIF.TRANSPARENT, title="DiffuseTexture", configItem=cfg.tex_diffuse_cfg)
        self.sc_normal = SwitchSettingCard(icon=FIF.TRANSPARENT, title="NormalTexture", configItem=cfg.tex_normal_cfg)
        self.sc_smoothspec = SwitchSettingCard(icon=FIF.TRANSPARENT, title="SmoothSpecTexture", configItem=cfg.tex_smoothspec_cfg)
        self.sc_greyscale = SwitchSettingCard(icon=FIF.TRANSPARENT, title="GreyscaleTexture", configItem=cfg.tex_greyscale_cfg)
        self.sc_envmap = SwitchSettingCard(icon=FIF.TRANSPARENT, title="EnvmapTexture", configItem=cfg.tex_envmap_cfg)
        self.sc_glow = SwitchSettingCard(icon=FIF.TRANSPARENT, title="GlowTexture", configItem=cfg.tex_glow_cfg)
        self.sc_inner = SwitchSettingCard(icon=FIF.TRANSPARENT, title="InnerLayerTexture", configItem=cfg.tex_inner_cfg)
        self.sc_wrinkles = SwitchSettingCard(icon=FIF.TRANSPARENT, title="WrinklesTexture", configItem=cfg.tex_wrinkles_cfg)

        self.__initWidget()

    def __initWidget(self):
        self.settings_group.addSettingCard(self.cb_bgsm_card)
        self.settings_group.addSettingCard(self.cb_bgem_card)
        self.settings_group.addSettingCard(self.sc_diffuse)
        self.settings_group.addSettingCard(self.sc_normal)
        self.settings_group.addSettingCard(self.sc_smoothspec)
        self.settings_group.addSettingCard(self.sc_greyscale)
        self.settings_group.addSettingCard(self.sc_envmap)
        self.settings_group.addSettingCard(self.sc_glow)
        self.settings_group.addSettingCard(self.sc_inner)
        self.settings_group.addSettingCard(self.sc_wrinkles)

        # add cards to group
        self.setupLayout()