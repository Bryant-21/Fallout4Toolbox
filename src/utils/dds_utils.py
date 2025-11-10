import os
import os
import subprocess
from pathlib import Path

from PIL import Image

from src.utils.appconfig import TEXCONV_EXE
from src.utils.logging_utils import logger



def load_image(path, f='RGBA'):
    """Load an image path into a PIL Image.

    - For non-DDS formats, uses PIL directly and converts to requested 'format' (default RGB).
    - For .dds, uses texconv.exe to convert to a temporary PNG, then loads it and converts to requested 'format'.

    Args:
        path (str): Input image path.
        f (str): Desired PIL mode for the returned image, e.g., 'RGB' or 'RGBA'.

    Returns:
        PIL.Image.Image: Loaded image in the requested mode.

    Raises:
        Exception: If texconv is required but not set or conversion fails.
    """
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext != '.dds':
            # Use PIL for regular formats
            with Image.open(path) as im:
                return im.convert(f)

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run texconv to output PNG into tmpdir
            if os.path.splitext(path)[0].endswith("_n"):
                cmd = [
                    TEXCONV_EXE,
                    '-ft', 'PNG',
                    '-f', 'RGBA',
                    '-y',
                    '-m', '1',
                    '-srgb', # Force 32-bit RGBA to handle BC5/other special formats
                    path,
                    '-o', tmpdir
                ]
            else:
                cmd = [
                    TEXCONV_EXE,
                    '-ft', 'PNG',
                    '-y',
                    '-m', '1',
                    path,
                    '-o', tmpdir
                ]
            logger.debug(f"Running texconv for DDS load: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.error(f"texconv failed while reading DDS: {result.stderr}")
                raise Exception(f"texconv failed while reading DDS: {result.stderr}")

            # texconv outputs a PNG with same basename but .PNG extension
            base = os.path.splitext(os.path.basename(path))[0]
            # texconv may output uppercase .PNG; try both cases
            cand_paths = [
                os.path.join(tmpdir, base + '.PNG'),
                os.path.join(tmpdir, base + '.png')
            ]
            out_png = next((p for p in cand_paths if os.path.exists(p)), None)
            if not out_png:
                # Sometimes texconv may create subfolders; scan tmpdir for first PNG
                for fn in os.listdir(tmpdir):
                    if fn.lower().endswith('.png'):
                        out_png = os.path.join(tmpdir, fn)
                        break
            if not out_png:
                raise Exception("texconv did not produce a PNG output when reading DDS")

            with Image.open(out_png) as im:
                return im.convert(f)
    except subprocess.TimeoutExpired:
        logger.error("texconv timed out while reading DDS")
        raise Exception("texconv timed out while reading DDS")
    except Exception as e:
        logger.error(f"Failed to load image '{path}': {e}")
        raise

def convert_to_dds(input_path, output_path, is_palette=False, generate_mips=False, target_format=None):
    """Convert image to DDS format using texconv.exe.

    Notes:
    - texconv derives the output filename from the INPUT filename and only lets us pick the output directory via -o.
      To ensure the final filename matches `output_path`, we rename (move) the produced file to `output_path` after conversion.
    - On Windows (case-insensitive FS), renaming a file that differs only by letter case (e.g., .DDS → .dds) can fail.
      We therefore treat paths equal in a case-insensitive way and skip the rename when only case differs.
    """
    logger.debug(f"Converting to DDS: {input_path} -> {output_path}")
    try:
        out_dir = os.path.dirname(output_path) or '.'
        os.makedirs(out_dir, exist_ok=True)

        if is_palette:
            # Use provided palette dimensions
            cmd = [
                TEXCONV_EXE,
                '-f', 'B8G8R8A8_UNORM',
                '-y',
                '-m', '1',
                input_path,
                '-o', out_dir
            ]
        else:
            # Determine output format
            out_fmt = target_format or 'BC7_UNORM'
            cmd = [TEXCONV_EXE, '-f', out_fmt, '-y']
            if not generate_mips:
                cmd.extend(['-m', '1'])
            cmd.extend([input_path, '-o', out_dir])

        logger.debug(f"Running texconv command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.error(f"texconv failed: {result.stderr}")
            raise Exception(f"texconv failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        logger.error("texconv timed out")
        raise Exception("texconv timed out")
    except Exception as e:
        logger.error(f"DDS conversion error: {str(e)}")
        raise Exception(f"DDS conversion error: {str(e)}")

def save_image(img, path, is_palette=False):
    if "dds" in path:
        png_path = os.path.splitext(path)[0] + ".png"
        try:
            img.save(png_path)
            convert_to_dds(png_path, path, is_palette)
        finally:
            try:
                if os.path.exists(png_path):
                    os.remove(png_path)
            except Exception as _cleanup_ex:
                logger.warning(f"Failed to remove temp file {png_path}: {_cleanup_ex}")
    else:
        img.save(path)


def add_temp_to_filename(path):
    p = Path(path)
    new_name = p.stem + "_temp.png"
    return str(p.with_name(new_name))