"""
Generate ComfyUI-ZeroCopy-Continuity 512x512 Icon Asset.

Architectural Context:
    Generates a professional, high-contrast, high-DPI (512x512) square icon
    for the ComfyUI Registry and ComfyUI-Manager node browser.
    Features:
    - Deep graphite / obsidian background (#0a0d14)
    - NVIDIA Blackwell / Cyber-green glowing neon accents (#76b900 / #00ff88)
    - Electric cyan data bridge / continuity rays (#00d2ff)
    - Central GPU Die / VRAM Pinning emblem with zero-copy interconnects
    - Anti-aliased geometry and smooth radial glow gradients
"""

import math
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def create_zerocopy_icon(output_path: str = "zerocopy-icon.png", size: int = 512) -> None:
    """Creates a high-tech modern icon for ComfyUI-ZeroCopy-Continuity.

    Args:
        output_path: Destination file path for PNG output.
        size: Width and height in pixels (default 512).
    """
    # Create master image with alpha
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 1. Base Rounded Background with dark carbon gradient
    corner_radius = int(size * 0.18)
    bg_margin = int(size * 0.04)
    bg_box = [bg_margin, bg_margin, size - bg_margin, size - bg_margin]

    # Gradient base
    base_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    base_draw = ImageDraw.Draw(base_layer)

    # Vertical gradient from #12161f to #07090d
    for y in range(bg_margin, size - bg_margin):
        ratio = (y - bg_margin) / (size - 2 * bg_margin)
        r = int(18 * (1 - ratio) + 7 * ratio)
        g = int(22 * (1 - ratio) + 9 * ratio)
        b = int(31 * (1 - ratio) + 13 * ratio)
        base_draw.line([(bg_margin, y), (size - bg_margin, y)], fill=(r, g, b, 255))

    # Mask to rounded rectangle
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle(bg_box, radius=corner_radius, fill=255)
    base_layer.putalpha(mask)
    img.alpha_composite(base_layer)

    # 2. Glowing Neon Border (Dual Tone: Cyan to NVIDIA Green)
    border_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    b_draw = ImageDraw.Draw(border_layer)
    b_draw.rounded_rectangle(bg_box, radius=corner_radius, outline=(0, 229, 255, 180), width=4)
    img.alpha_composite(border_layer)

    # Inner subtle border
    inner_margin = bg_margin + 6
    inner_box = [inner_margin, inner_margin, size - inner_margin, size - inner_margin]
    inner_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    in_draw = ImageDraw.Draw(inner_layer)
    in_draw.rounded_rectangle(inner_box, radius=corner_radius - 4, outline=(118, 185, 0, 100), width=2)
    img.alpha_composite(inner_layer)

    # 3. Ambient Glow in Center (NVIDIA Green & Electric Cyan)
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    cx, cy = size // 2, int(size * 0.46)
    r_glow = int(size * 0.35)
    for r in range(r_glow, 0, -4):
        alpha = int(45 * (1 - r / r_glow) ** 1.5)
        # Blend from cyan at core to green at edges
        mix = r / r_glow
        g_col = (int(0 * mix + 118 * (1 - mix)), int(229 * mix + 255 * (1 - mix)), int(255 * mix + 0 * (1 - mix)), alpha)
        glow_draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=g_col)
    glow = glow.filter(ImageFilter.GaussianBlur(12))
    img.alpha_composite(glow)

    # 4. Central Graphic: GPU Die & Memory Ring / Continuity Loop
    # Draw PCB Circuit Traces
    trace_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    t_draw = ImageDraw.Draw(trace_layer)

    # Radiating circuit lines
    angles = [30, 60, 120, 150, 210, 240, 300, 330]
    for ang in angles:
        rad = math.radians(ang)
        x1 = cx + math.cos(rad) * 110
        y1 = cy + math.sin(rad) * 110
        x2 = cx + math.cos(rad) * 170
        y2 = cy + math.sin(rad) * 170
        t_draw.line([(x1, y1), (x2, y2)], fill=(0, 229, 255, 90), width=2)
        # Node dot at end
        t_draw.ellipse([x2 - 4, y2 - 4, x2 + 4, y2 + 4], fill=(118, 185, 0, 220))

    # Outer memory nodes (simulating VRAM chips pinned around GPU die)
    chip_w, chip_h = 36, 20
    chip_coords = [
        (cx - chip_w // 2, cy - 135),  # Top
        (cx - chip_w // 2, cy + 115),  # Bottom
        (cx - 135 - chip_w // 2, cy - chip_h // 2),  # Left
        (cx + 135 - chip_w // 2, cy - chip_h // 2),  # Right
    ]
    for ch_x, ch_y in chip_coords:
        t_draw.rectangle([ch_x, ch_y, ch_x + chip_w, ch_y + chip_h], fill=(20, 30, 45, 230), outline=(0, 229, 255, 200), width=2)
        # Chip core pin
        t_draw.rectangle([ch_x + 6, ch_y + 4, ch_x + chip_w - 6, ch_y + chip_h - 4], fill=(118, 185, 0, 180))

    img.alpha_composite(trace_layer)

    # 5. Central GPU Core & Infinity / Continuity Symbol
    core_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    c_draw = ImageDraw.Draw(core_layer)

    # GPU Core square (rotated 45 deg or rounded)
    core_box = [cx - 65, cy - 65, cx + 65, cy + 65]
    c_draw.rounded_rectangle(core_box, radius=16, fill=(10, 15, 24, 240), outline=(118, 185, 0, 255), width=3)
    c_draw.rounded_rectangle([cx - 56, cy - 56, cx + 56, cy + 56], radius=10, outline=(0, 229, 255, 140), width=1)

    # Zero-Copy Fast Transfer Lightning / Infinite Continuity Infinity Glyph
    # Smooth infinity symbol in the center
    inf_pts = []
    a = 38
    for t in range(0, 360, 3):
        rad = math.radians(t)
        # Lemniscate of Bernoulli
        denom = 1 + (math.sin(rad) ** 2)
        x = cx + (a * math.sqrt(2) * math.cos(rad)) / denom
        y = cy + (a * math.sqrt(2) * math.cos(rad) * math.sin(rad)) / denom
        inf_pts.append((x, y))

    if len(inf_pts) > 1:
        for i in range(len(inf_pts)):
            p1 = inf_pts[i]
            p2 = inf_pts[(i + 1) % len(inf_pts)]
            c_draw.line([p1, p2], fill=(0, 255, 200, 255), width=5)

    img.alpha_composite(core_layer)

    # 6. Typography: "ZERO COPY" and "CONTINUITY"
    text_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    tx_draw = ImageDraw.Draw(text_layer)

    # Font setup
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    t1 = "ZERO-COPY"
    t2 = "CONTINUITY"

    bb1 = tx_draw.textbbox((0, 0), t1, font=font_large)
    w1 = bb1[2] - bb1[0]
    tx_draw.text(((size - w1) // 2, size - 105), t1, font=font_large, fill=(255, 255, 255, 245))

    bb2 = tx_draw.textbbox((0, 0), t2, font=font_small)
    w2 = bb2[2] - bb2[0]
    tx_draw.text(((size - w2) // 2, size - 68), t2, font=font_small, fill=(118, 185, 0, 230))

    img.alpha_composite(text_layer)

    # Save output
    img.save(output_path, "PNG", optimize=True)
    print(f"Successfully generated high-res icon: {output_path} ({size}x{size})")


if __name__ == "__main__":
    create_zerocopy_icon("zerocopy-icon.png", 512)
