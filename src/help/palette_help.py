from __future__ import annotations

import os
from typing import TYPE_CHECKING


from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    SettingCardGroup, isDarkTheme
)
from qfluentwidgets import ScrollArea, ExpandLayout



from src.utils.cards import TextCard
from src.utils.filesystem_utils import get_app_root


class PaletteHelp(ScrollArea):

    def __init__(self, parent):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr('About'), self.scroll_widget)

        self.info = TextCard(
            text=
            """
            So this will take any texture generate a Greyscale version of it and the Palette to go with it. 
            
            A Palette (also called a lut) is basically all the colors from your original texture reduced to 256 and mapped to greyscale.
            
            This will allow you to change the colors in photoshop or substance painter, with no extra texture files. Great for file size and xbox mods
            
            The real power of this tool comes from the additional images picker. You can export the same texture from substance (or blender) multiple times with different colors.
            Then you select these, and the tool will automatically add them to the palette. Then in your BGSM you can choose which colors you want to use.
            
            The greyscale option allows you to take the original texture, but with new details on it and convert it to the same mapping.
            
            So say you want to add the brotherhood of steel logo, you just add it to the first texture with a color that ALREADY EXISTS on that texture and export it.
            
            Repeat, it CANNOT contain any new colors, and MUST only use the same colors as the "base" texture. 
            
            https://fallout.wiki/wiki/Resource:Using_pallettes_for_recoloring_in_Fallout_4
            
            Dither Methods
            •"none" — leave indices unchanged.
            •"median" — median filter on indices (removes isolated outliers while preserving edges).
            •"gaussian" — mild blur of indices (softens edges; more “anti‑aliased” look).
            •"dither" — ordered Bayer dithering on normalized indices (adds subtle texture to mask banding).
            • "median_dither" — median filter first, then ordered dithering (clean up specks, then add gentle texture).
            • "gaussian_dither" — gaussian blur first, then ordered dithering (soften edges, then add gentle texture).
             
            """,
            height=300
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
        self.__connectSignalToSlot()

    def __initLayout(self):
        # add cards to group
        self.settings_group.addSettingCard(self.info)

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

    def __connectSignalToSlot(self):
        pass