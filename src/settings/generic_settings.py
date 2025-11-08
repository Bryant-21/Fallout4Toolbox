import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QFileDialog
from qfluentwidgets import (
    FluentIcon as FIF, SettingCardGroup, isDarkTheme, PushSettingCard
)
from qfluentwidgets import ScrollArea, ExpandLayout

from src.utils.appconfig import cfg
from src.utils.cards import SpinSettingCard
from utils.filesystem_utils import get_app_root


class GenericSettings(ScrollArea):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr('Settings'), self.scroll_widget)

        self.threads_card = SpinSettingCard(
            cfg.threads_cfg,
            FIF.SPEED_OFF,
            self.tr("Threads"),
            self.tr("Parallel workers (default = CPU cores)"),
            step=1
        )

    def setupLayout(self):
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
        self.settings_group.addSettingCard(self.threads_card)
        # add setting card group to layout
        self.expand_layout.setSpacing(28)
        self.expand_layout.setContentsMargins(15, 0, 15, 0)
        self.expand_layout.addWidget(self.settings_group)

    def __setQss(self):
        """ set style sheet """
        self.scroll_widget.setObjectName('scrollWidget')

        theme = 'dark' if isDarkTheme() else 'light'
        with open(os.path.join(get_app_root(), 'resource', 'qss', theme, 'setting_interface.qss'), encoding='utf-8') as f:
            self.setStyleSheet(f.read())


