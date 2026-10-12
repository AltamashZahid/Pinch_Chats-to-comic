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
EMOJI_FONT = r"C:\Windows\Fonts\seguiemj.ttf"


def _font(size: int):
    path = r"C:\Windows\Fonts\arial.ttf"
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def _runs(text: str) -> list[tuple[str, bool]]:
    """Split text into (chunk, is_emoji) runs."""
    from pinch.emoji import EMOTION_OF
    runs: list[tuple[str, bool]] = []
    for ch in text:
        e = ch in EMOTION_OF
        if runs and runs[-1][1] == e:
            runs[-1] = (runs[-1][0] + ch, e)
        else:
            runs.append((ch, e))
    return runs


def _draw_mixed(d, xy, text, font, efont, fill) -> None:
    x, y = xy
    for chunk, is_emoji in _runs(text):
        if is_emoji:
            d.text((x, y - 4), chunk, font=efont, embedded_color=True)
            x += d.textlength(chunk, font=efont, embedded_color=True)
        else:
            d.text((x, y), chunk, font=font, fill=fill)
            x += d.textlength(chunk, font=font)


def _mixed_len(d, text, font, efont) -> float:
    return sum(d.textlength(c, font=efont, embedded_color=True) if e else d.textlength(c, font=font)
               for c, e in _runs(text))


def render_chat(messages: list[tuple[str, str]], theme: str = "whatsapp_light",
                width: int = 720, header: bool = True) -> np.ndarray:
    """messages: [(side, text)] with side in {"left", "right"}. Returns a BGR array.
    Text may contain emoji (drawn in colour). Side "left-big"/"right-big" sends big emoji
    on their own, with no bubble, the way chat apps show emoji-only messages."""
    bg, left_fill, right_fill, text_color = THEMES[theme]
    font = _font(30)
    efont = ImageFont.truetype(EMOJI_FONT, 30)
    big = ImageFont.truetype(EMOJI_FONT, 64)
    pad, margin, gap = 22, 24, 18
    height = 140 + sum((100 if s.endswith("big") else 30 + 2 * pad) + gap for s, _ in messages) + 100
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
        if side.endswith("big"):
            w = d.textlength(text, font=big, embedded_color=True)
            x = margin + 10 if side.startswith("left") else width - margin - 10 - w
            d.text((x, y), text, font=big, embedded_color=True)
            y += 100 + gap
            continue
        tw = _mixed_len(d, text, font, efont)
        bw, bh = int(tw) + 2 * pad, 30 + 2 * pad
        x = margin if side == "left" else width - margin - bw
        fill = left_fill if side == "left" else right_fill
        d.rounded_rectangle([x, y, x + bw, y + bh], radius=16, fill=fill)
        color = (255, 255, 255) if theme == "imessage" and side == "right" else text_color
        _draw_mixed(d, (x + pad, y + pad - 2), text, font, efont, color)
        y += bh + gap

    return np.array(img)[:, :, ::-1].copy()
