from PIL import Image, ImageDraw, ImageFont
import os

# Configuration
GRID_SIZE = 16  # 16x16 = 256 grayscale values (0-255)
TILE_SIZE = 250  # 4000 / 16 = 250
CANVAS_SIZE = 4000  # 4000x4000 image
FONT_SIZE = 80  # Large font for clear cutouts

# Create transparent RGBA image
img = Image.new('RGBA', (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Try to load a bold/truetype font for better clarity
font = None
possible_fonts = [
    "Arial Bold",
    "Helvetica-Bold",
    "DejaVuSans-Bold",
    "Arial",
    "Helvetica",
    "DejaVuSans",
]

# Try system fonts first
for font_name in possible_fonts:
    try:
        font = ImageFont.truetype(font_name, FONT_SIZE)
        break
    except OSError:
        continue

# Fallback: load default font (but it's small, so we'll generate a large bitmap manually if needed)
if font is None:
    try:
        # On some systems, you can use a path
        if os.name == 'nt':  # Windows
            font = ImageFont.truetype("arialbd.ttf", FONT_SIZE)
        elif os.name == 'posix':
            # Try common Linux/macOS paths
            for path in [
                "/System/Library/Fonts/Helvetica.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
            ]:
                if os.path.exists(path):
                    font = ImageFont.truetype(path, FONT_SIZE)
                    break
    except Exception:
        pass

# Final fallback: use default font (but note: it may not scale well)
if font is None:
    font = ImageFont.load_default()
    # Since default font is tiny, we'll adjust by drawing bigger via scaling later if needed
    # But for 250px tile, we really need a vector font. Consider installing one if missing.

# Generate all 256 grayscale squares
for i in range(GRID_SIZE):  # rows
    for j in range(GRID_SIZE):  # columns
        gray_val = i * GRID_SIZE + j  # 0 to 255

        # Position of this tile
        x0 = j * TILE_SIZE
        y0 = i * TILE_SIZE
        x1 = x0 + TILE_SIZE
        y1 = y0 + TILE_SIZE

        # Fill square with grayscale color
        draw.rectangle([x0, y0, x1, y1], fill=(gray_val, gray_val, gray_val, 255))

        # Prepare number text
        text = str(gray_val)

        # Get text size
        try:
            # Modern Pillow (>=8.0.0)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except:
            # Older Pillow
            tw, th = draw.textsize(text, font=font)

        # Center the text
        tx = x0 + (TILE_SIZE - tw) // 2
        ty = y0 + (TILE_SIZE - th) // 2

        # Draw TRANSPARENT text (alpha=0) → creates cutout
        draw.text((tx, ty), text, fill=(0, 0, 0, 0), font=font)

# Save as transparent PNG
output_path = "grayscale_4k_cutout.png"
img.save(output_path, "PNG", optimize=True)
print(f"✅ 4000×4000 transparent PNG saved as: {output_path}")
print(f"   Contains all 256 grayscale values (0–255) in 250×250 px tiles with large cutout numbers.")