from src.settings.generic_settings import GenericSettings
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QFileDialog
from qfluentwidgets import FluentIcon as FIF, SettingCardGroup, isDarkTheme, PushSettingCard, InfoBar, HyperlinkCard, \
    CustomColorSettingCard, OptionsSettingCard
from qfluentwidgets import ScrollArea, ExpandLayout

from src.utils.appconfig import cfg, HELP_URL, NEXUS_URL, VERSION, YEAR, AUTHOR, KOFI_URL, DISCORD_URL
from src.utils.cards import SpinSettingCard
from src.utils.icons import CustomIcons

class TemplateSettings(GenericSettings):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.__initWidget()

    def __initWidget(self):
        # add cards to group
        self.setupLayout()