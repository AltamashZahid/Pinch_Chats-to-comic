"""Find chat bubbles in a screenshot and work out which side (speaker) each is on.

Chat apps draw messages as solid-filled rounded rectangles on a mostly uniform
background. We find the background colour, mask everything that differs from it,
and keep the connected regions that look like bubbles: big enough, not full-width
(header / input bar), and dominated by one fill colour that isn't the background
(which rejects loose text like timestamps and date separators).
"""
import cv2
import numpy as np

from .models import Box, Bubble

QUANT = 4           # colour quantisation step when looking for dominant colours
BG_TOLERANCE = 14   # max per-channel difference still counted as background
MIN_FILL_RATIO = 0.75  # text bubbles measure 0.81-0.91, emoji ~0.67


def _dominant_color(pixels: np.ndarray) -> tuple[np.ndarray, float]:
    """Most common quantised colour in an (N, 3) array, and the share of pixels it covers."""
    q = (pixels // QUANT).astype(np.int32)
    keys = (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    i = counts.argmax()
    k = values[i]
    color = np.array([(k >> 16) & 0xFF, (k >> 8) & 0xFF, k & 0xFF]) * QUANT + QUANT // 2
    return color, counts[i] / len(keys)


def background_color(img: np.ndarray) -> np.ndarray:
    return _dominant_color(img.reshape(-1, 3))[0]


def foreground_mask(img: np.ndarray, bg: np.ndarray) -> np.ndarray:
    diff = np.abs(img.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    mask = (diff > BG_TOLERANCE).astype(np.uint8) * 255
    # No morphological closing here: stacked bubbles can sit only 1-2 px apart.
    # Fill holes so each bubble becomes one solid blob.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def detect_bubbles(img: np.ndarray) -> list[Bubble]:
    """Return bubbles sorted top to bottom. `img` is a BGR image."""
    H, W = img.shape[:2]
    bg = background_color(img)
    mask = foreground_mask(img, bg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 0.04 * W or h < max(16, 0.012 * H):
            continue
        if w > 0.92 * W:  # header, input bar, full-width banners
            continue
        if y <= 1 or y + h >= H - 1:  # clipped by the screenshot edge, text is cut off
            continue
        region = img[y:y + h, x:x + w][mask[y:y + h, x:x + w] > 0]
        fill, ratio = _dominant_color(region)
        if ratio < MIN_FILL_RATIO:
            continue  # no solid fill: avatar, image, loose text
        if np.abs(fill.astype(int) - bg.astype(int)).max() <= BG_TOLERANCE:
            continue
        side = "right" if (W - (x + w)) < x else "left"
        dark = float(fill.mean()) < 110
        bubbles.append(Bubble(Box(x, y, w, h), side, tuple(int(v) for v in fill), dark))

    bubbles = _drop_off_palette(bubbles)
    bubbles.sort(key=lambda b: (b.box.y, b.box.x))
    return bubbles


def _drop_off_palette(bubbles: list[Bubble]) -> list[Bubble]:
    """Each side of a chat uses one bubble colour. Anything else (big emoji,
    stickers, link previews) is not a text bubble."""
    kept = []
    for side in ("left", "right"):
        group = [b for b in bubbles if b.side == side]
        if not group:
            continue
        weights: dict[tuple, int] = {}
        for b in group:
            weights[b.fill] = weights.get(b.fill, 0) + b.box.w * b.box.h
        palette = max(weights, key=weights.get)
        kept += [b for b in group
                 if max(abs(a - c) for a, c in zip(b.fill, palette)) <= 2 * BG_TOLERANCE]
    return kept
