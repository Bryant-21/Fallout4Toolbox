from pathlib import Path

from PIL import Image, ImageChops

from src.utils.logging_utils import logger

def dilation_fill_static(out_path: Path, rgba_img: Image.Image, max_iters: int = 64) -> bool:
    try:
        if rgba_img.mode != 'RGBA':
            rgba_img = rgba_img.convert('RGBA')
        r, g, b, a = rgba_img.split()
        base_rgb = Image.merge('RGB', (r, g, b))
        known = a.point(lambda v: 255 if v > 0 else 0)
        unknown = a.point(lambda v: 0 if v > 0 else 255)
        unknown_initial = unknown.copy()
        if unknown.getbbox() is None:
            return False
        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

        # Helper: non-wrapping shift (pads with zeros) to avoid edge wrap-around blending
        def shift_no_wrap(img: Image.Image, dx: int, dy: int) -> Image.Image:
            w, h = img.size
            result = Image.new(img.mode, (w, h))
            # Compute source and destination rectangles
            src_x0 = max(0, -dx)
            src_y0 = max(0, -dy)
            src_x1 = min(w, w - dx) if dx >= 0 else w
            src_y1 = min(h, h - dy) if dy >= 0 else h
            if src_x0 >= src_x1 or src_y0 >= src_y1:
                return result
            region = img.crop((src_x0, src_y0, src_x1, src_y1))
            dst_x = max(0, dx)
            dst_y = max(0, dy)
            result.paste(region, (dst_x, dst_y))
            return result

        filled_any = False
        for _ in range(max_iters):
            iter_filled = False
            for dx, dy in neighbors:
                # Use non-wrapping shift so we never blend/copy colors from the opposite edge
                shifted_known = shift_no_wrap(known, dx, dy)
                fill_mask = ImageChops.multiply(shifted_known, unknown)
                if fill_mask.getbbox() is None:
                    continue
                shifted_rgb = shift_no_wrap(base_rgb, dx, dy)
                base_rgb.paste(shifted_rgb, mask=fill_mask)
                unknown = ImageChops.subtract(unknown, fill_mask)
                known = ImageChops.lighter(known, fill_mask)
                iter_filled = True
                filled_any = True
            if not iter_filled:
                break
        filled_mask = ImageChops.subtract(unknown_initial, unknown)
        new_alpha = ImageChops.lighter(a, filled_mask)
        nr, ng, nb = base_rgb.split()
        out_img = Image.merge('RGBA', (nr, ng, nb, new_alpha))
        out_img.save(out_path, format='PNG')
        if filled_any:
            logger.info(f"Color fill applied (alpha made opaque in filled regions): {out_path}")
        return filled_any
    except Exception as e:
        logger.warning(f"Color fill failed for {out_path}: {e}")
        return False