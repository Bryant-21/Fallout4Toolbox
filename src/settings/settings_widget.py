from qfluentwidgets import FluentIcon as FIF, SettingCardGroup, InfoBar, HyperlinkCard, \
    CustomColorSettingCard, OptionsSettingCard

from src.settings.generic_settings import GenericSettings
from src.utils.appconfig import cfg, HELP_URL, NEXUS_URL, VERSION, YEAR, AUTHOR, KOFI_URL, DISCORD_URL
from src.utils.icons import CustomIcons


class MainSettings(GenericSettings):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('Settings')
        self.personalGroup = SettingCardGroup(self.tr('Personalization'), self.scroll_widget)

        self.themeColorCard = CustomColorSettingCard(
            cfg.themeColor,
            FIF.PALETTE,
            self.tr('Theme color'),
            self.tr('Change the theme color of you application'),
            self.personalGroup
        )

        self.zoomCard = OptionsSettingCard(
            cfg.dpiScale,
            FIF.ZOOM,
            self.tr("Interface zoom"),
            self.tr("Change the size of widgets and fonts"),
            texts=[
                "75%", "95%", "100%", "105%", "125%", "150%", "175%", "200%",
                self.tr("Use system setting")
            ],
            parent=self.personalGroup
        )

        self.aboutGroup = SettingCardGroup(self.tr('About'), self.scroll_widget)

        self.helpCard = HyperlinkCard(
            HELP_URL,
            self.tr('Github'),
            FIF.GITHUB,
            self.tr('GitHub'),
            self.tr('Source Page for reporting issues and Downloading the latest release'),
            self.aboutGroup
        )

        self.aboutCard = HyperlinkCard(
            NEXUS_URL,
            self.tr('Nexus Mods'),
            CustomIcons.NEXUS.icon(),
            self.tr('Nexus'),
            '© ' + self.tr('Copyright') + f" {YEAR}, {AUTHOR}. " +
            self.tr('Version') + f" {VERSION}",
            self.aboutGroup
        )

        self.supportCard = HyperlinkCard(
            KOFI_URL,
            self.tr('ko-fi'),
            CustomIcons.KO_FI.icon(),
            self.tr('Support Us'),
            'Anything donated will support the app development ❤️',
            self.aboutGroup
        )

        self.discord = HyperlinkCard(
            DISCORD_URL,
            self.tr('Discord'),
            CustomIcons.DISCORD.icon(),
            self.tr('Join my discord today!'),
            'My discord is the best place to reach me',
            self.aboutGroup
        )

        self.__initWidget()

    def __initWidget(self):
        self.personalGroup.addSettingCard(self.themeColorCard)
        self.personalGroup.addSettingCard(self.zoomCard)
        self.aboutGroup.addSettingCard(self.helpCard)
        self.aboutGroup.addSettingCard(self.supportCard)
        self.aboutGroup.addSettingCard(self.aboutCard)
        self.aboutGroup.addSettingCard(self.discord)

        self.expand_layout.addWidget(self.personalGroup)
        self.expand_layout.addWidget(self.aboutGroup)

        cfg.appRestartSig.connect(self.__showRestartTooltip)

        # add cards to group
        self.setupLayout()


    def __showRestartTooltip(self):
        """ show restart tooltip """
        InfoBar.warning(
            '',
            self.tr('Configuration takes effect after restart'),
            parent=self.window(),
            duration=3000
        )