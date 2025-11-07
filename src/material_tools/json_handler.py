from __future__ import annotations
import json
from typing import Tuple, Literal

MaterialType = Literal["BGSM", "BGEM"]


def detect_material_type_from_json(obj: dict) -> MaterialType:
    if "sDiffuseTexture" in obj or "sSmoothSpecTexture" in obj:
        return "BGSM"
    if "sBaseTexture" in obj:
        return "BGEM"
    # Fallback: prefer BGSM if ambiguous
    return "BGSM"


def load_json(path: str) -> Tuple[MaterialType, dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    mtype = detect_material_type_from_json(obj)
    return mtype, obj


def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def prefix_texture(p: str | None, folder: str) -> str | None:
    """Insert folder immediately before the filename, keeping prior directories.
    Examples:
      - p="Weapons/B21_GaussMinigun/barrel1_d.dds", folder="asd" ->
        "Weapons/B21_GaussMinigun/asd/barrel1_d.dds"
      - p="barrel1_d.dds", folder="asd" -> "asd/barrel1_d.dds"
    Avoid double insertion if the folder is already directly before the filename.
    """
    if not p:
        return p
    q = p.replace('\\', '/')
    if '/' in q:
        dirpart, fname = q.rsplit('/', 1)
    else:
        dirpart, fname = '', q
    if dirpart and dirpart.endswith('/' + folder):
        return q
    if not dirpart and q.startswith(folder + '/'):
        return q
    if dirpart:
        return f"{dirpart}/{folder}/{fname}"
    else:
        return f"{folder}/{fname}"


def update_textures_json(obj: dict, mat_type: MaterialType, folder: str, selected_paths: set[str] | None = None) -> dict:
    obj = dict(obj)  # shallow copy
    # Map BGSM logical names to JSON keys for each material type
    if mat_type == "BGSM":
        bgsm_map = {
            "DiffuseTexture": "sDiffuseTexture",
            "NormalTexture": "sNormalTexture",
            "SmoothSpecTexture": "sSmoothSpecTexture",
            "GreyscaleTexture": "sGreyscaleTexture",
            "EnvmapTexture": "sEnvmapTexture",
            "GlowTexture": "sGlowTexture",
            "InnerLayerTexture": "sInnerLayerTexture",
            "WrinklesTexture": "sWrinklesTexture",
        }
        keys_to_update = bgsm_map.items()
        if selected_paths:
            keys_to_update = [(k, v) for k, v in bgsm_map.items() if k in selected_paths]
        for logical, json_key in keys_to_update:
            if json_key in obj and isinstance(obj[json_key], str) and obj[json_key]:
                obj[json_key] = prefix_texture(obj[json_key], folder)
    else:  # BGEM
        # Only a subset applies to BGEM
        bgem_map = {
            "DiffuseTexture": "sBaseTexture",
            "NormalTexture": "sNormalTexture",
            "SmoothSpecTexture": "sSpecularTexture",
            "GreyscaleTexture": "sGrayscaleTexture",
            "EnvmapTexture": "sEnvmapTexture",
            "GlowTexture": "sGlowTexture",
            # InnerLayerTexture, WrinklesTexture do not exist in BGEM
        }
        keys_to_update = bgem_map.items()
        if selected_paths:
            keys_to_update = [(k, v) for k, v in bgem_map.items() if k in selected_paths]
        for logical, json_key in keys_to_update:
            if json_key in obj and isinstance(obj[json_key], str) and obj[json_key]:
                obj[json_key] = prefix_texture(obj[json_key], folder)
    return obj
