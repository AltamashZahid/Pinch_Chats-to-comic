"""Draw fake chat screenshots so tests know the ground truth."""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

THEMES = {
    # name: (background, left bubble, right bubble, text) as RGB
    "whatsapp_light": ((236, 229, 221), (255, 255, 255), (220, 248, 198), (17, 27, 33)),
    "whatsapp_dark": ((11, 20, 26), (32, 44, 51), (0, 92, 75), (233, 237, 239)),
    "imessage": ((255, 255, 255), (233, 233, 235), (10, 132, 255), (0, 0, 0)),
}


def _font(size: int):
    path = r"C:\Windows\Fonts\arial.ttf"
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def render_chat(messages: list[tuple[str, str]], theme: str = "whatsapp_light",
                width: int = 720, header: bool = True) -> np.ndarray:
    """messages: [(side, text)] with side in {"left", "right"}. Returns a BGR array."""
    bg, left_fill, right_fill, text_color = THEMES[theme]
    font = _font(30)
    pad, margin, gap = 22, 24, 18
    height = 140 + sum(30 + 2 * pad + gap for _ in messages) + 100
    img = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(img)

    y = 30
    if header:
        d.rectangle([0, 0, width, 90], fill=(7, 94, 84))
        d.text((24, 28), "Contact", font=font, fill=(255, 255, 255))
        y = 110
    d.text((width // 2 - 30, y), "Today", font=_font(22), fill=(120, 120, 120))
    y += 50

    for side, text in messages:
        tw = d.textlength(text, font=font)
        bw, bh = int(tw) + 2 * pad, 30 + 2 * pad
        x = margin if side == "left" else width - margin - bw
        fill = left_fill if side == "left" else right_fill
        d.rounded_rectangle([x, y, x + bw, y + bh], radius=16, fill=fill)
        color = (255, 255, 255) if theme == "imessage" and side == "right" else text_color
        d.text((x + pad, y + pad - 2), text, font=font, fill=color)
        y += bh + gap

    return np.array(img)[:, :, ::-1].copy()
