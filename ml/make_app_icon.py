"""Generate a clean brand app icon + splash logo for the Flutter app.
Green rounded square + a white rice-leaf (vesica) with a central vein.
Outputs into rice_blast_detector/assets/.
"""
import math
from PIL import Image, ImageDraw, ImageChops, ImageFilter

OUT = r"C:\Users\KESHAV UPADHYAY\Desktop\Major project\rice_blast_detector\assets"
GREEN = (30, 122, 77, 255)
GREEN_DARK = (17, 92, 58, 255)
WHITE = (255, 255, 255, 255)


def leaf_mask(S, cx, cy, R, o, angle_deg):
    """Vesica (pointed-oval) leaf via intersection of two circles, then rotated."""
    m1 = Image.new("L", (S, S), 0)
    m2 = Image.new("L", (S, S), 0)
    ImageDraw.Draw(m1).ellipse([cx - o - R, cy - R, cx - o + R, cy + R], fill=255)
    ImageDraw.Draw(m2).ellipse([cx + o - R, cy - R, cx + o + R, cy + R], fill=255)
    lens = ImageChops.multiply(m1, m2)          # vertical pointed lens
    return lens.rotate(angle_deg, resample=Image.BICUBIC, center=(cx, cy))


def vertical_gradient(S, top, bot):
    g = Image.new("RGBA", (S, S))
    for y in range(S):
        t = y / (S - 1)
        r = int(top[0] * (1 - t) + bot[0] * t)
        gg = int(top[1] * (1 - t) + bot[1] * t)
        b = int(top[2] * (1 - t) + bot[2] * t)
        for_row = Image.new("RGBA", (S, 1), (r, gg, b, 255))
        g.paste(for_row, (0, y))
    return g


def build(S, rounded=True, pad=0.0):
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    # gradient background
    bg = vertical_gradient(S, (34, 138, 86), (17, 92, 58))
    if rounded:
        mask = Image.new("L", (S, S), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [0, 0, S - 1, S - 1], radius=int(S * 0.22), fill=255)
        img.paste(bg, (0, 0), mask)
    else:
        img.paste(bg, (0, 0))

    cx = cy = S // 2
    scale = 1.0 - pad
    R = int(S * 0.42 * scale)
    o = int(S * 0.245 * scale)
    lm = leaf_mask(S, cx, cy, R, o, -35)

    # white leaf
    white = Image.new("RGBA", (S, S), WHITE)
    img.paste(white, (0, 0), lm)

    # central vein (green) along the leaf's long axis (rotated -35 from vertical)
    d = ImageDraw.Draw(img)
    ang = math.radians(-35)
    half = int(S * 0.255 * scale)
    dx, dy = math.sin(ang) * half, -math.cos(ang) * half
    d.line([(cx - dx, cy - dy), (cx + dx, cy + dy)],
           fill=GREEN, width=max(6, int(S * 0.014)))
    # a couple of side veins
    for f in (-0.42, 0.0, 0.42):
        bx = cx + dx * f
        by = cy + dy * f
        sang = ang + math.radians(52)
        sl = int(S * 0.11 * scale)
        d.line([(bx, by), (bx + math.sin(sang) * sl, by - math.cos(sang) * sl)],
               fill=GREEN, width=max(4, int(S * 0.009)))
        sang2 = ang - math.radians(52)
        d.line([(bx, by), (bx + math.sin(sang2) * sl, by - math.cos(sang2) * sl)],
               fill=GREEN, width=max(4, int(S * 0.009)))
    return img


# App icon (rounded, full-bleed brand)
build(1024, rounded=True).save(f"{OUT}\\icon.png")
# Foreground for Android adaptive (transparent bg, leaf padded so it's not clipped)
fg = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
lm = leaf_mask(1024, 512, 512, int(1024 * 0.32), int(1024 * 0.185), -35)
fg.paste(Image.new("RGBA", (1024, 1024), WHITE), (0, 0), lm)
fg.save(f"{OUT}\\icon_fg.png")
# Splash logo (transparent, leaf only on transparent for centering on green splash)
build(512, rounded=True).save(f"{OUT}\\splash_logo.png")
print("Saved icon.png, icon_fg.png, splash_logo.png to", OUT)
