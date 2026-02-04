import os.path
from enum import Enum

from PySide6.QtGui import QIcon, QColor
from qfluentwidgets import FluentIconBase, Theme, getIconColor
from qfluentwidgets.common.icon import SvgIconEngine, writeSvg

from src.utils.filesystem_utils import get_app_root


class CustomIcons(FluentIconBase, Enum):
    """ Custom icons """
    # Regular icons
    GPU = "gpu"
    VAULT_BOY = "vault_boy"
    KO_FI = "ko-fi"
    NEXUS = "nexus"
    DISCORD = "discord"
    LOOP = "loop"
    RAM = "ram"
    STEPS = "steps"
    SCALE = "scale"
    RECORD = "record"
    CPU = "cpu"
    IMPORTANT = "important"
    HUGGING_FACE = "huggingface"
    NEW = "NEW"
    VOICE_OVER = "voice-over"
    VOICE_SQUARE = "voice-square"
    STYLE = "style"
    TTS = "tts"
    VOICE = "voice"
    G = "g"
    FROG = "frog"
    BULK = "bulk"
    REPLACE = "replace"
    FX = "fx"
    ENHANCE = "enhance"
    SINE = "sine"
    API = "api"
    LLAMA = "llama"
    PILLS = "pills"
    TRIANGLE = "triangle"
    DIA = "dia"
    F5 = "f5"
    FISH = "fish"
    MAGIC = "magic"
    UP = "up"
    SPARK = "spark"
    CSM = "sesame"
    HIGGS = "higgs"
    CHATTERBOX = "chatterbox"
    DMO2 = "dmo2"
    PADDING = "padding"
    ROBOTS = "robots"
    TEAM = "team"
    GROUP = "group"
    MULTI = "multi"
    BETHESDA = "bethesda"
    MICROSOFT = "microsoft"
    PUZZLE = "puzzle"
    PAINT = "paint"
    # Stroke-only icons
    ALPHA = "alpha"
    BETA = "beta"
    BUG = "bug"
    DELETE = "delete"
    MUSIC = "music"
    IMAGE = "image"
    IMAGEADD = "image-add"
    GREYSCALE = "greyscale"
    FOLDERRIGHT = "folder-right"
    WIDTH = "width"
    HEIGHT = "height"
    RESCALE = "rescale"
    PALETTE = "palette"
    PALETTE_2 = "palette-2"
    SWATCH = "swatch"
    GRAPH = "graph"
    BGSM = "bgsm"
    BGEM = "bgem"
    PERSON_WALKING = "person-walking"
    SHIELD = "shield"
    MUTANT = "mutant"
    REPORT = "report"
    QUANT = "quant"
    SUB = "sub"
    COMBINE = "combine"
    CUBE = "cube"
    FILL = "fill"
    CUT = "cut"
    FLOOD = "flood"
    INFINITY = "infinity"
    ADD_SOLID = "add-solid"
    ADD_BASIC = "add-basic"
    CUT_FILM = "cut-film"
    ARROW_UP = "arrow-up"
    FOLDER_IMAGE = "folder-image"
    SELECTION = "selection"
    SATURATION = "saturation"
    CONTRAST = "contrast"
    FILTER_ADD ="filter-add"
    FILTER_REMOVE = "filter-remove"
    FOLDERS_LINE ="folders-line"
    COLOR_BALANCE ="color-balance"
    COLORS_SQUARE = "colors-square"
    IMAGE_CIRCLE = "image-circle"
    LINE_ADD ="line-add"
    LINE_REMOVE = "line-remove"
    PREVIEW_FILE = "preview-file"
    BULK_EDIT = "bulk-edit"
    ROWS = "rows"
    COLUMNS = "columns"
    SCALE_POLY = "scale-poly"
    COPY_POLY = "copy-poly"
    FILE = "file"
    FILE_ADD = "file-add"
    FILE_MINUS = "file-minus"
    FILE_CLOSE ="file-close"
    FILE_SEARCH = "file-search"
    FILE_CODE = "file-code"
    FILE_EDIT = "file-edit"
    FILE_DIGIT = "file-digit"
    IMAGE_VIEWER = "image-viewer"
    FIT = "fit"

    def icon(self, theme=Theme.AUTO, color: QColor = None, stroke: bool = False) -> QIcon:
        """ create a fluent icon

        Parameters
        ----------
        theme: Theme
            the theme of icon
            * `Theme.Light`: black icon
            * `Theme.DARK`: white icon
            * `Theme.AUTO`: icon color depends on `qconfig.theme`

        color: QColor | Qt.GlobalColor | str
            icon color, only applicable to svg icon
            
        stroke: bool
            whether to render the icon as a stroke instead of a fill
        """
        path = self.path(theme)

        if color:
            color = color.name()
        elif not color:
            color = getIconColor(theme)

        if stroke:
            style = f"fill: none; stroke: rgb(0, 0, 0); stroke-linecap: round; stroke-linejoin: round; stroke-width: 2; stroke: {color}"
        else:
            style = f"fill: {color}"

        return QIcon(SvgIconEngine(writeSvg(path, style=style)))

    def path(self, theme=Theme.AUTO):
        return os.path.abspath(os.path.join(get_app_root(), 'resource', 'icons', f'{self.value}.svg'))