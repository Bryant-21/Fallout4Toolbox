import os

from utils.filesystem_utils import get_app_root


def try_import(module_name):
    try:
        __import__(module_name)
        return True
    except Exception:
        return False

CAPABILITIES = {
    "nif_tools": try_import("io_scene_nifly"),
    "mip_flooding": try_import("mipflooding.wrapper"),
    "skimage": try_import("skimage"),
    "ChaiNNer": os.path.join(get_app_root(), "ChaiNNer", "ChaiNNer.exe")
}

def missing():
    return [name for name, ok in CAPABILITIES.items() if not ok]