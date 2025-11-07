import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QFileDialog
from qfluentwidgets import FluentIcon as FIF, SettingCardGroup, isDarkTheme, PushSettingCard, InfoBar, HyperlinkCard, \
    CustomColorSettingCard, OptionsSettingCard
from qfluentwidgets import ScrollArea, ExpandLayout

from src.utils.appconfig import cfg, HELP_URL, NEXUS_URL, VERSION, YEAR, AUTHOR, KOFI_URL, DISCORD_URL
from src.utils.cards import SpinSettingCard
from src.utils.icons import CustomIcons
from utils.filesystem_utils import get_app_root


class MainSettings(ScrollArea):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('Settings')
        self.title = self.tr('Settings')
        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
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
        self.resize(1000, 800)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setViewportMargins(0, 0, 0, 20)
        self.setWidget(self.scroll_widget)
        self.setWidgetResizable(True)

        # initialize style sheet
        self.__setQss()

        # initialize layout
        self.__initLayout()

    def __initLayout(self):


        self.personalGroup.addSettingCard(self.themeColorCard)
        self.personalGroup.addSettingCard(self.zoomCard)

        self.aboutGroup.addSettingCard(self.helpCard)
        self.aboutGroup.addSettingCard(self.supportCard)
        self.aboutGroup.addSettingCard(self.aboutCard)
        self.aboutGroup.addSettingCard(self.discord)

        # add setting card group to layout
        self.expand_layout.setSpacing(28)
        self.expand_layout.setContentsMargins(15, 0, 15, 0)

        self.expand_layout.addWidget(self.personalGroup)
        self.expand_layout.addWidget(self.aboutGroup)

        cfg.appRestartSig.connect(self.__showRestartTooltip)

    def __setQss(self):
        """ set style sheet """
        self.scroll_widget.setObjectName('scrollWidget')

        theme = 'dark' if isDarkTheme() else 'light'
        with open(os.path.join(get_app_root(), 'resource', 'qss', theme, 'setting_interface.qss'), encoding='utf-8') as f:
            self.setStyleSheet(f.read())


    def __showRestartTooltip(self):
        """ show restart tooltip """
        InfoBar.warning(
            '',
            self.tr('Configuration takes effect after restart'),
            parent=self.window(),
            duration=3000
        )