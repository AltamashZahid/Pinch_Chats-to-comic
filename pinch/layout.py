"""Assemble panel images into a comic page with emotion-shaped speech bubbles.

Bubble outlines are drawn as polygons around a superellipse (a rounded rectangle that
hugs the text better than an ellipse), with a radius modulation per bubble style:
spikes for anger, a starburst for surprise, a shaky edge for fear, a sagging bottom
for sadness.
"""
import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .render import RenderedPanel

PANEL = 512
GUTTER = 22
BORDER = 5
COLUMNS = 2
FONT_PATHS = [r"C:\Windows\Fonts\comicbd.ttf", r"C:\Windows\Fonts\comic.ttf", r"C:\Windows\Fonts\arial.ttf"]


def _font(size: int):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _superellipse(t: float, a: float, b: float, n: float = 3.0) -> tuple[float, float]:
    c, s = math.cos(t), math.sin(t)
    return (a * math.copysign(abs(c) ** (2 / n), c), b * math.copysign(abs(s) ** (2 / n), s))


def bubble_outline(style: str, cx: float, cy: float, a: float, b: float) -> list[tuple[float, float]]:
    """Polygon for a bubble of the given style centred on (cx, cy) with semi-axes a, b."""
    spikes = {"jagged": (32, 1.16), "burst": (22, 1.42)}
    if style in spikes:
        count, out = spikes[style]
        pts = []
        for i in range(count * 2):
            t = math.pi * i / count
            x, y = _superellipse(t, a, b)
            k = out if i % 2 else 1.0
            pts.append((cx + x * k, cy + y * k))
        return pts

    pts = []
    for i in range(160):
        t = 2 * math.pi * i / 160
        x, y = _superellipse(t, a, b)
        k = 1.0
        if style == "wobbly":
            k = 1 + 0.045 * math.sin(9 * t) + 0.03 * math.sin(17 * t + 1)
        elif style == "wavy":
            k = 1 + 0.05 * math.sin(14 * t)
        elif style == "droopy" and y > 0:
            y *= 1 + 0.35 * (y / b) ** 4      # bottom edge sags
        pts.append((cx + x * k, cy + y * k))
    return pts


def _wrap(text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    for width in range(28, 6, -2):
        lines = textwrap.wrap(text, width) or [text]
        if max(draw.textlength(l, font=font) for l in lines) <= max_width:
            return lines
    return textwrap.wrap(text, 6)


def draw_bubble(img: Image.Image, text: str, style: str, side: str, top: int, scale: int = 1) -> int:
    """Draw one bubble near the top of a panel. Returns the y where the next bubble may start.
    `scale` > 1 when drawing on a supersampled canvas (PIL polygons aren't anti-aliased)."""
    draw = ImageDraw.Draw(img)
    font = _font(22 * scale)
    lines = _wrap(text, font, int(img.width * 0.5), draw)
    line_h = font.getbbox("Ag")[3] + 4 * scale
    tw = max(draw.textlength(l, font=font) for l in lines)
    th = line_h * len(lines)
    a, b = tw / 2 * 1.18 + 16 * scale, th / 2 * 1.3 + 12 * scale
    reach = 1.42 if style == "burst" else 1.16 if style == "jagged" else 1.1
    cx = 14 * scale + a * reach if side == "left" else img.width - 14 * scale - a * reach
    cy = top + b * reach
    stroke = 3 * scale

    # Tail points down toward the speaker's side of the panel.
    tail_x = cx - a * 0.3 if side == "left" else cx + a * 0.3
    tail = [(tail_x - 12 * scale, cy + b * 0.7), (tail_x + 12 * scale, cy + b * 0.7),
            (tail_x + (-30 if side == "left" else 30) * scale, cy + b + 34 * scale)]
    outline = bubble_outline(style, cx, cy, a, b)
    draw.polygon(tail, fill="white", outline="black", width=stroke)
    draw.polygon(outline, fill="white", outline="black", width=stroke)
    draw.polygon([(x, y - stroke) for x, y in tail[:2]] + [tail[2]], fill="white")  # hide seam

    y = cy - th / 2
    for line in lines:
        draw.text((cx - draw.textlength(line, font=font) / 2, y), line, font=font, fill="black")
        y += line_h
    return int(cy + b * reach + 34 * scale)


def compose_page(panels: list[RenderedPanel], out_path: Path) -> Path:
    rows = math.ceil(len(panels) / COLUMNS)
    cols = min(COLUMNS, len(panels))
    W = cols * PANEL + (cols + 1) * GUTTER
    H = rows * PANEL + (rows + 1) * GUTTER
    page = Image.new("RGB", (W, H), "white")
    for i, rp in enumerate(panels):
        ss = 2  # supersample for smooth bubble outlines
        art = Image.open(rp.image).convert("RGB").resize((PANEL * ss, PANEL * ss), Image.LANCZOS)
        top = 10 * ss
        for line in rp.panel.lines:
            side = "right" if line.speaker == "me" else "left"
            top = draw_bubble(art, line.text, line.bubble, side, top, scale=ss)
        art = art.resize((PANEL, PANEL), Image.LANCZOS)
        r, c = divmod(i, COLUMNS)
        in_row = min(COLUMNS, len(panels) - r * COLUMNS)
        offset = (cols - in_row) * (PANEL + GUTTER) // 2  # centre a short last row
        x, y = offset + GUTTER + c * (PANEL + GUTTER), GUTTER + r * (PANEL + GUTTER)
        page.paste(art, (x, y))
        ImageDraw.Draw(page).rectangle([x - BORDER // 2, y - BORDER // 2, x + PANEL + BORDER // 2,
                                        y + PANEL + BORDER // 2], outline="black", width=BORDER)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(out_path)
    return out_path
