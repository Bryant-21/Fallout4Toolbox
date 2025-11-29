import tempfile
from pathlib import Path

from PIL import Image

from src.utils.logging_utils import logger
from src.utils.capabilities import CAPABILITIES

if CAPABILITIES["mip_flooding"]:
    from mipflooding.wrapper import image_processing as _mip_image_processing

def _apply_mip_flooding_to_png(out_path: Path, rgba_img: Image.Image) -> bool:
    """Apply mip flooding to a transparent PNG using the mipflooding package.

    Returns True if processed, False otherwise.
    """
    if _mip_image_processing is None:
        return False
    try:
        with tempfile.TemporaryDirectory(prefix="mipflood_") as tmpdir:
            tmpdir_p = Path(tmpdir)
            color_path = tmpdir_p / "color_C.png"
            mask_path = tmpdir_p / "mask_A.png"
            # Ensure RGBA
            if rgba_img.mode != 'RGBA':
                rgba_img = rgba_img.convert('RGBA')
            r, g, b, a = rgba_img.split()
            color_img = Image.merge('RGB', (r, g, b))
            # Save intermediates
            color_img.save(color_path, format='PNG')
            a.save(mask_path, format='PNG')
            # Run mip flooding
            _mip_image_processing.run_mip_flooding(str(color_path), str(mask_path), str(out_path))
            return True
    except Exception as e:
        logger.warning(f"Mip flooding failed for {out_path}: {e}")
        return False