import os
import json
import tempfile
import subprocess
import shutil
import urllib.request
from typing import Dict, Optional, List, Tuple

from PIL import Image

from src.utils.logging_utils import logger
from src.utils.filesystem_utils import get_app_root

# Supported file extensions for counting/matching inputs
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".dds")

# Paths and constants shared across UIs
CHAINNER_EXE = os.path.join(get_app_root(), "ChaiNNer", "ChaiNNer.exe")
MODELS_DIR = os.path.join(get_app_root(), "ChaiNNer", "models")
# External chain path (do not embed the chain). Users can edit/replace this JSON.
CHAIN_FILE = os.path.join(get_app_root(), "resource", "chain", "upscale.chn")
# Multiple-files chain path
CHAIN_FILE_MULTIPLE = os.path.join(get_app_root(), "resource", "chain", "upscale_multiple.chn")

# Known model URLs. Extend as needed.
MODEL_URLS = {
    "UltraSharpV2": "https://huggingface.co/Kim2091/UltraSharpV2/resolve/main/4x-UltraSharpV2.safetensors?download=true",
    "4x-Normal-RG0-BC1": "https://github.com/RunDevelopment/ESRGAN-models/raw/main/normals/4x-Normal-RG0-BC1.pth?download=true",
    "4x-Normal-RG0-BC7": "https://github.com/RunDevelopment/ESRGAN-models/raw/main/normals/4x-Normal-RG0-BC7.pth?download=true",
    "4x-Normal-RG0": "https://github.com/RunDevelopment/ESRGAN-models/raw/main/normals/4x-Normal-RG0.pth?download=true",
    "4x-PBRify_UpscalerV4": "https://github.com/Kim2091/Kim2091-Models/releases/download/4x-PBRify_UpscalerV4/4x-PBRify_UpscalerV4.pth?download=true",
    "4xTextures_GTAV_rgt-s_dither": "https://huggingface.co/Phips/4xTextures_GTAV_rgt-s_dither/resolve/main/4xTextures_GTAV_rgt-s_dither.safetensors?download=true",
    "4x-PBRify_UpscalerSIR-M_V2": "https://github.com/Kim2091/Kim2091-Models/releases/download/4x-PBRify_UpscalerSIR-M_V2/4x-PBRify_UpscalerSIR-M_V2.pth?download=true",
    "4xNomosWebPhoto_RealPLKSR": "https://github.com/Phhofm/models/releases/download/4xNomosWebPhoto_RealPLKSR/4xNomosWebPhoto_RealPLKSR.pth?download=true"
}

# Known model file extensions. Do not guess.
MODEL_EXTS = {
    "UltraSharpV2": ".safetensors",
    "4x-Normal-RG0-BC1": ".pth",
    "4x-Normal-RG0-BC7": ".pth",
    "4x-Normal-RG0": ".pth",
    "4x-PBRify_UpscalerV4": ".pth",
    "4xTextures_GTAV_rgt-s_dither": ".safetensors",
    "4x-PBRify_UpscalerSIR-M_V2": ".pth",
    "4xNomosWebPhoto_RealPLKSR": ".pth",
}


def json_safe_path(path: str) -> str:
    """Return a JSON-safe absolute path by converting to forward slashes."""
    try:
        abspath = os.path.abspath(path)
        return abspath.replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def ensure_models_dir() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)


def model_download_path(model_name: str) -> str:
    """Return the default local filename under MODELS_DIR for the model.
    Preserves given extension; if none, uses known extension (MODEL_EXTS) otherwise defaults to .onnx.
    """
    ensure_models_dir()
    base = model_name
    lower = base.lower()
    if not lower.endswith((".onnx", ".pth", ".safetensors")):
        # Use known extension when available, do NOT guess
        ext = MODEL_EXTS.get(model_name)
        if ext is None:
            ext = ".onnx"
        base += ext
    return os.path.join(MODELS_DIR, base)


def resolve_model_path(model_name: str) -> Optional[str]:
    """Resolve a local model path by trying known extension first, then common ones.
    Returns None if not found.
    """
    ensure_models_dir()
    # If absolute path provided and exists
    if os.path.isabs(model_name) and os.path.exists(model_name):
        return model_name

    # If model_name already contains an extension, check it directly under MODELS_DIR
    given_path = os.path.join(MODELS_DIR, model_name)
    name_has_ext = os.path.splitext(model_name)[1] != ""

    candidates: List[str] = []
    if name_has_ext:
        candidates.append(given_path)
    else:
        base_no_ext = os.path.join(MODELS_DIR, model_name)
        # Prefer the known extension for this model name, if any
        known_ext = MODEL_EXTS.get(model_name)
        if known_ext:
            candidates.append(base_no_ext + known_ext)
        # Then try typical extensions
        for ext in (".pth", ".safetensors", ".onnx"):
            if known_ext != ext:  # avoid duplicate
                candidates.append(base_no_ext + ext)
        # Finally, a bare path (in case file was saved without ext)
        candidates.append(base_no_ext)

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def download_model(model_name: str) -> str:
    """Download the model by name to MODELS_DIR using MODEL_URLS.
    Returns the local path on success, raises Exception on failure.
    """
    ensure_models_dir()
    url = MODEL_URLS.get(model_name)
    if not url:
        raise Exception(f"No download URL configured for model '{model_name}'.")
    target = model_download_path(model_name)
    try:
        tmp_path, _ = urllib.request.urlretrieve(url)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        # If an old partial exists, remove it first
        if os.path.exists(target):
            try:
                os.remove(target)
            except Exception:
                pass
        shutil.move(tmp_path, target)
        logger.info("Downloaded model %s -> %s", model_name, target)
        return target
    except Exception as e:
        logger.exception("Failed to download model %s: %s", model_name, e)
        raise


def get_or_download_model(model_name: str) -> str:
    """Return local path to model; if not present, attempt to download it.
    Raises Exception if cannot be resolved/downloaded.
    """
    local = resolve_model_path(model_name)
    if local and os.path.exists(local):
        return local
    # Fallback to download
    return download_model(model_name)


def parse_chain_for_ids(chain_path: str) -> Dict[str, str]:
    """Parse a ChaiNNer .chn (JSON) file and extract node IDs for
    - chainner:image:load -> key 'load'
    - chainner:image:load_images -> key 'load_images'
    - chainner:pytorch:load_model -> key 'load_model'
    - chainner:image:save -> key 'save'
    - chainner:pytorch:upscale_image -> key 'upscale'

    Returns a dict mapping keys to node IDs. Missing keys will not be present.
    """
    try:
        with open(chain_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        nodes = data.get('content', {}).get('nodes', [])
        result: Dict[str, str] = {}
        for node in nodes:
            d = node.get('data', {})
            schema = (d.get('schemaId') or '').lower()
            node_id = d.get('id') or node.get('id')
            if not schema or not node_id:
                continue
            if schema == 'chainner:image:load' and 'load' not in result:
                result['load'] = node_id
            elif schema == 'chainner:image:load_images' and 'load_images' not in result:
                result['load_images'] = node_id
            elif schema == 'chainner:pytorch:load_model' and 'load_model' not in result:
                result['load_model'] = node_id
            elif schema == 'chainner:image:save' and 'save' not in result:
                result['save'] = node_id
            elif schema == 'chainner:pytorch:upscale_image' and 'upscale' not in result:
                result['upscale'] = node_id
        return result
    except Exception as e:
        logger.exception('Failed parsing chain file %s: %s', chain_path, e)
        return {}


def run_chainner(input_png: str, model_path: str, out_dir: str, expected_output_png: str) -> bool:
    """Invoke ChaiNNer CLI using single-image chain and overrides. Returns True on success.

    - Reads node IDs from CHAIN_FILE.
    - Computes and passes an integer upscale factor to target ~4K on the longest side
      when an upscale node exists in the chain.
    """
    try:
        chain_path = CHAIN_FILE
        if not os.path.exists(chain_path):
            raise Exception(f"ChaiNNer chain not found. Please place it at: {chain_path}")

        ids = parse_chain_for_ids(chain_path)
        SAVE_ID = ids.get("save")
        LOAD_ID = ids.get("load")
        LOAD_MODEL_ID = ids.get("load_model")
        UPSCALE_ID = ids.get("upscale")
        if not (SAVE_ID and LOAD_ID and LOAD_MODEL_ID):
            raise Exception("Required nodes not found in chain (need image:load, pytorch:load_model, image:save).")

        # Compute integer upscale factor to target ~4K on the longest side.
        try:
            with Image.open(input_png) as _im:
                w, h = _im.size
            longest = max(w, h)
        except Exception:
            longest = 0
        factor = 1
        if longest > 0:
            factor = max(1, int(round(4096 / float(longest))))
        logger.info("Upscale factor for %s -> %dx (longest=%d)", os.path.basename(input_png), factor, longest)

        # Build overrides using '#<id>:<field>' format.
        inputs_map = {
            f"#{LOAD_ID}:0": json_safe_path(input_png),
            f"#{LOAD_MODEL_ID}:0": json_safe_path(model_path),
            # Do not set output name here to avoid double suffixing; only set the directory
            f"#{SAVE_ID}:1": json_safe_path(out_dir),
        }
        if UPSCALE_ID:
            inputs_map[f"#{UPSCALE_ID}:5"] = factor

        overrides = {"inputs": inputs_map}
        ov_fd, ov_path = tempfile.mkstemp(prefix="overrides_", suffix=".json")
        os.close(ov_fd)
        with open(ov_path, "w", encoding="utf-8") as f:
            json.dump(overrides, f)

        cmd = [CHAINNER_EXE, "run", chain_path, "--override", ov_path]
        logger.info("Running ChaiNNer: %s", " ".join(cmd))
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, creationflags=0)
            if proc.stdout is not None:
                for line in proc.stdout:
                    logger.info("[ChaiNNer] %s", line.rstrip())
            proc.wait()
            last_rc = proc.returncode
        except Exception as e:
            logger.exception("Failed to launch ChaiNNer: %s", e)
            return False

        if last_rc == 0 and os.path.exists(expected_output_png):
            return True
        else:
            logger.warning("ChaiNNer CLI failed or output not found Last rc=%s", last_rc)
            return False
    except Exception as e:
        logger.warning("ChaiNNer invocation failed: %s", e)
        return False


def run_chainner_directory(folder: str, model_name: str, out_dir: str, glob_pattern: str) -> bool:
    """Invoke ChaiNNer CLI using the multi-image chain (CHAIN_FILE_MULTIPLE) targeting a folder.

    Overrides used:
    - load_images:0 -> directory path
    - load_images:3 -> WCMatch glob (e.g., '**/*_d*')
    - load_model:0 -> model path (resolved/downloaded automatically)
    - save:1 -> output directory

    Returns True when the CLI exits with code 0.
    """
    try:
        chain_path = CHAIN_FILE_MULTIPLE
        if not os.path.exists(chain_path):
            raise Exception(f"ChaiNNer multi-image chain not found. Please place it at: {chain_path}")

        ids = parse_chain_for_ids(chain_path)
        SAVE_ID = ids.get("save")
        LOAD_IMAGES_ID = ids.get("load_images")
        LOAD_MODEL_ID = ids.get("load_model")
        if not (SAVE_ID and LOAD_IMAGES_ID and LOAD_MODEL_ID):
            raise Exception("Required nodes not found in multi chain (need image:load_images, pytorch:load_model, image:save).")

        # Ensure model exists locally or download it
        model_path = get_or_download_model(model_name)

        inputs_map = {
            f"#{LOAD_IMAGES_ID}:0": json_safe_path(folder),
            f"#{LOAD_IMAGES_ID}:3": glob_pattern,
            f"#{LOAD_MODEL_ID}:0": json_safe_path(model_path),
            f"#{SAVE_ID}:1": json_safe_path(out_dir),
        }
        overrides = {"inputs": inputs_map}
        ov_fd, ov_path = tempfile.mkstemp(prefix="overrides_dir_", suffix=".json")
        os.close(ov_fd)
        with open(ov_path, "w", encoding="utf-8") as f:
            json.dump(overrides, f)

        cmd = [CHAINNER_EXE, "run", chain_path, "--override", ov_path]
        logger.info("Running ChaiNNer (dir): %s | glob=%s | model=%s", folder, glob_pattern, model_name)
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, creationflags=0)
            if proc.stdout is not None:
                for line in proc.stdout:
                    logger.info("[ChaiNNer] %s", line.rstrip())
            proc.wait()
            last_rc = proc.returncode
        except Exception as e:
            logger.exception("Failed to launch ChaiNNer (dir): %s", e)
            return False
        return last_rc == 0
    except Exception as e:
        logger.warning("ChaiNNer (dir) invocation failed: %s", e)
        return False


def count_images_by_suffix(folder: str, include_subdirs: bool) -> Tuple[int, int]:
    """Return counts for (_d, _n) images in folder.
    Match is by filename stem ending with '_d' or '_n' (case-insensitive) and extension in IMAGE_EXTS.
    """
    d_count = 0
    n_count = 0
    try:
        if include_subdirs:
            walker = os.walk(folder)
            for root, _, files in walker:
                for fn in files:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in IMAGE_EXTS:
                        continue
                    stem = os.path.splitext(fn)[0].lower()
                    if stem.endswith("_d"):
                        d_count += 1
                    elif stem.endswith("_n"):
                        n_count += 1
        else:
            for fn in os.listdir(folder):
                full = os.path.join(folder, fn)
                if not os.path.isfile(full):
                    continue
                ext = os.path.splitext(fn)[1].lower()
                if ext not in IMAGE_EXTS:
                    continue
                stem = os.path.splitext(fn)[0].lower()
                if stem.endswith("_d"):
                    d_count += 1
                elif stem.endswith("_n"):
                    n_count += 1
    except Exception as e:
        logger.warning("Failed counting images in %s: %s", folder, e)
    return d_count, n_count


def upscale_directory_two_pass(folder: str, out_dir: Optional[str], include_subdirs: bool,
                               textures_model_name: str, normals_model_name: str) -> Tuple[int, int, int]:
    """Run ChaiNNer directory chain twice: first diffuse (_d) with textures model, then normals (_n) with normals model.

    Returns (saved, skipped, failed) as best-effort estimates based on counts and CLI status.
    """
    if not out_dir:
        out_dir = folder
    os.makedirs(out_dir, exist_ok=True)

    base = "**/*" if include_subdirs else "*"
    glob_d = f"{base}_d*"
    glob_n = f"{base}_n*"

    d_count, n_count = count_images_by_suffix(folder, include_subdirs)
    saved = skipped = failed = 0

    if d_count > 0:
        ok_d = run_chainner_directory(folder, textures_model_name, out_dir, glob_d)
        if ok_d:
            saved += d_count
        else:
            failed += d_count
    else:
        logger.info("No '_d' images found in %s", folder)

    if n_count > 0:
        ok_n = run_chainner_directory(folder, normals_model_name, out_dir, glob_n)
        if ok_n:
            saved += n_count
        else:
            failed += n_count
    else:
        logger.info("No '_n' images found in %s", folder)

    return saved, skipped, failed
