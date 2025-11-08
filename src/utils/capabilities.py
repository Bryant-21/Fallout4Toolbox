import os
import traceback

from utils.filesystem_utils import get_app_root
from utils.logging_utils import logger

def try_import(module_name):
    try:
        __import__(module_name)
        return True
    except Exception:
        logger.error("Failed to import '{}' module".format(module_name))
        return False

def try_load_mipflooding():
    try:
        from mipflooding.wrapper import image_processing
        return True
    except Exception:
        logger.error("Failed to import 'mipflooding' module")
        traceback.print_exc()
        return False

CAPABILITIES = {
    "pythonnet": try_import("pythonnet"),
    "nif_tools": try_import("io_scene_nifly"),
    "mip_flooding": try_import("mipflooding.wrapper") and try_load_mipflooding(),
    "skimage": try_import("skimage"),
    "ChaiNNer": os.path.join(get_app_root(), "ChaiNNer", "ChaiNNer.exe")
}

def missing():
    return [name for name, ok in CAPABILITIES.items() if not ok]