# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs
from glob import glob

# -------------------------
# Hidden imports & data files
# -------------------------
hiddenimports = []
datas = []

# Scientific stack
hiddenimports += collect_submodules('numpy')
hiddenimports += collect_submodules('scipy')
hiddenimports += collect_submodules('sklearn')
hiddenimports += collect_submodules('skimage')
hiddenimports += collect_submodules('joblib')
hiddenimports += collect_submodules('imagequant')
hiddenimports += collect_submodules('pycparser')
hiddenimports += collect_submodules('pythonnet')
hiddenimports += collect_submodules('numba')
hiddenimports += collect_submodules('clr')
hiddenimports += collect_submodules('clr_loader')

datas += collect_data_files('numpy')
datas += collect_data_files('scipy')
datas += collect_data_files('sklearn')
datas += collect_data_files('skimage')
datas += collect_data_files('pythonnet')
datas += collect_data_files('clr_loader')

def get_third_party_files():
    third_party_files = []
    third_party_dir = './third_party'

    if os.path.exists(third_party_dir):
        for root, dirs, files in os.walk(third_party_dir):
            for file in files:
                full_path = os.path.join(root, file)
                # Calculate relative path for target directory
                rel_path = os.path.relpath(root, third_party_dir)
                if rel_path == '.':
                    target_dir = 'third_party'
                else:
                    target_dir = os.path.join('third_party', rel_path)
                third_party_files.append((full_path, target_dir))

    return third_party_files

datas += get_third_party_files()



# -------------------------
# Analysis
# -------------------------
a = Analysis(
    ['FalloutToolbox.py'],
    pathex=['src', 'third_party'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['numpy_preload_hook.py'],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# -------------------------
# Executable (NOT onefile)
# -------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Fallout4Toolbox',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon='icon.ico',
)

# -------------------------
# Onedir distribution folder
# -------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='Fallout4Toolbox',
)
