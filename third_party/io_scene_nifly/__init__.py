"""NIF format export/import for Blender using Nifly"""

# Copyright © 2021, Bad Dog.

bl_info = {
    "name": "NIF format",
    "description": "Nifly Import/Export for Skyrim, Skyrim SE, and Fallout 4 NIF files (*.nif)",
    "author": "Bad Dog",
    "blender": (4, 5, 0),
    "version": (21, 0, 0),
    "location": "File > Import-Export",
    "support": "COMMUNITY",
    "category": "Import-Export"
}

import os.path
# System libraries
import sys

# Locate the DLL and other files we need either in their development or install locations.
nifly_path = None
hkxcmd_path = None
pynifly_dev_root = None
pynifly_dev_path = None
asset_path = None

if 'PYNIFLY_DEV_ROOT' in os.environ:
    pynifly_dev_root = os.environ['PYNIFLY_DEV_ROOT']
    pynifly_dev_path = os.path.join(pynifly_dev_root, r"pynifly\pynifly")
    nifly_path = os.path.join(pynifly_dev_root, r"PyNifly\NiflyDLL\x64\Debug\NiflyDLL.dll")
    hkxcmd_path = os.path.join(pynifly_dev_path, "hkxcmd.exe")
    asset_path = os.path.join(pynifly_dev_path, "blender_assets")

if nifly_path and os.path.exists(nifly_path):
    if pynifly_dev_path not in sys.path:
        sys.path.insert(0, pynifly_dev_path)
else:
    # Load from install location
    py_addon_path = os.path.dirname(os.path.realpath(__file__))
    if py_addon_path not in sys.path:
        sys.path.append(py_addon_path)
    nifly_path = os.path.join(py_addon_path, "NiflyDLL.dll")
    hkxcmd_path = os.path.join(py_addon_path, "hkxcmd.exe")
    asset_path = os.path.join(py_addon_path, "blender_assets")

# Pynifly tools
from niflytools import *
from nifdefs import *
from pynifly import *
import xmltools