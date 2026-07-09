#!/usr/bin/env python3
"""Generate assets/icon.png (1024x1024) — Kagami's app icon.

A modern macOS-style squircle (true superellipse) with a soft sky→water
gradient and a glassy "liquid glass" sheen. The glyph is a minimal scene —
a sun and mountains — mirrored below a horizon line as a faded reflection,
echoing the name *Kagami* (鏡, "mirror"). Drawn supersampled and downsampled
for clean anti-aliasing.

Requires: Pillow, numpy.  Run: python3 assets/make_icon.py
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 1024
SS = S * 4                       # 4x supersample
C = SS / 2.0
WHITE = (255, 255, 255)


# ---------------------------------------------------------------- background
# True superellipse ("squircle") mask — closer to Apple's continuous corners
# than a plain rounded rectangle. macOS's icon grid expects the shape to fill
# only the central ~80% of the tile (the 824-in-1024 convention), with ~10%
# transparent padding on every side; otherwise the icon reads larger than the
# system apps sitting next to it in the Dock and Finder.
margin = SS * 0.098
half = (SS - 2 * margin) / 2.0
n = 5.0
yy, xx = np.mgrid[0:SS, 0:SS].astype(np.float64)
u = (xx - C) / half
v = (yy - C) / half
d = np.abs(u) ** n + np.abs(v) ** n
# Smooth 1px-ish edge for anti-aliasing before the 4x downscale.
edge = 0.012
alpha = np.clip((1.0 - d) / edge + 0.5, 0.0, 1.0)
mask = Image.fromarray((alpha * 255).astype(np.uint8))

# Vertical sky→water gradient with a soft top-center glow (glass highlight).
top = (122, 178, 255)            # light sky blue
bot = (60, 70, 196)             # deep indigo
t = (yy / (SS - 1))[..., None]
grad = np.array(top) * (1 - t) + np.array(bot) * t
glow = np.exp(-(((xx - C) / (0.62 * SS)) ** 2 + ((yy - 0.16 * SS) / (0.42 * SS)) ** 2))
grad += (glow * 46)[..., None]
# Gentle darkening toward the bottom edge for depth.
shade = np.clip((yy - 0.6 * SS) / (0.4 * SS), 0, 1)
grad -= (shade * 26)[..., None]
grad = np.clip(grad, 0, 255).astype(np.uint8)

img = Image.new("RGBA", (SS, SS), (0, 0, 0, 0))
img.paste(Image.fromarray(grad), (0, 0), mask)


# ------------------------------------------------------------------- glyph
# A single glassy orb facing the viewer, centered on the tile: a smooth
# reflective gradient with one soft specular highlight. An abstract reading of
# *Kagami* (鏡, "mirror") — a clean reflective surface, no literal object.
R = SS * 0.300                    # disc radius, centered on the tile
disc = Image.new("L", (SS, SS), 0)
ImageDraw.Draw(disc).ellipse([C - R, C - R, C + R, C + R], fill=255)
disc = np.array(disc, np.float64) / 255.0

# Convex-glass shading: treat the disc as a sphere lit from the upper-left, so
# the surface reads as a polished, reflective mirror with real depth instead of
# a flat fill. A faint Fresnel rim hugs the edge like the lip of glass.
nx, ny = (xx - C) / R, (yy - C) / R
nz = np.sqrt(np.clip(1.0 - nx ** 2 - ny ** 2, 0.0, 1.0))
L = np.array([-0.45, -0.55, 0.70])
L = L / np.linalg.norm(L)
diff = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0.0, 1.0)
shade = 0.5 + 0.5 * diff                       # ambient + directional light
light, dark = np.array((251, 253, 255)), np.array((150, 172, 230))
rgb = dark + (light - dark) * shade[..., None]
rad = np.sqrt(nx ** 2 + ny ** 2)
rim = np.clip((rad - 0.85) / 0.15, 0.0, 1.0) ** 1.5
rgb = rgb + (255.0 - rgb) * (rim * 0.40)[..., None]
rgb = np.clip(rgb, 0, 255)
mirror = np.dstack([rgb, disc * 255]).astype(np.uint8)
mirror = Image.fromarray(mirror)

# Soft drop shadow beneath the disc for depth.
shadow = Image.new("RGBA", (SS, SS), (0, 0, 0, 0))
shadow.paste((18, 22, 56, 95), (0, 0),
             Image.fromarray((disc * 255).astype(np.uint8)))
shadow = shadow.filter(ImageFilter.GaussianBlur(SS * 0.016))
shadow = shadow.transform((SS, SS), Image.AFFINE,
                          (1, 0, 0, 0, 1, -int(SS * 0.009)))

scene = Image.new("RGBA", (SS, SS), (0, 0, 0, 0))
scene = Image.alpha_composite(scene, shadow)
scene = Image.alpha_composite(scene, mirror)

# Clip the whole scene to the squircle and composite onto the background.
scene.putalpha(Image.fromarray(
    (np.array(scene.split()[3], np.float64) * (np.array(mask) / 255.0)
     ).astype(np.uint8)))
img = Image.alpha_composite(img, scene)


# --------------------------------------------------------- top glass sheen
# A faint white highlight hugging the inside of the top edge.
sheen = Image.new("RGBA", (SS, SS), (0, 0, 0, 0))
sg = np.clip(1.0 - (yy - margin) / (SS * 0.26), 0, 1) ** 2
sa = (sg * 60 * (np.array(mask) / 255.0)).astype(np.uint8)
sheen = Image.merge("RGBA", [Image.new("L", (SS, SS), 255)] * 3 +
                    [Image.fromarray(sa)])
img = Image.alpha_composite(img, sheen)


out = os.path.join(os.path.dirname(__file__), "icon.png")
img.resize((S, S), Image.LANCZOS).save(out)
print("wrote", out)
