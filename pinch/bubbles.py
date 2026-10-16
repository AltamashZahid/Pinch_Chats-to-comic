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
FILL_TOLERANCE = 20  # JPEG noise and colour fringes: text on saturated purple measured 0.74 at 12, 0.89 at 20
GAP_COVERAGE = 0.15  # rows this empty between two bubbles split a merged region
MIN_RECTANGULARITY = 0.84  # share of the bounding box a bubble fills: bubbles measured 0.85-0.98,
                           # big emoji 0.82, avatars 0.76-0.78 (three real screenshots)


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


def uniform_mask(img: np.ndarray, k: int = 5, max_std: float = 8.0) -> np.ndarray:
    """Pixels whose k x k neighbourhood is one flat colour (per-channel std below `max_std`;
    JPEG noise is ~2-4). Bubble interiors are flat; wallpaper doodles are thin lines whose
    neighbourhoods are not."""
    f = img.astype(np.float32)
    mean = cv2.blur(f, (k, k))
    var = cv2.blur(f * f, (k, k)) - mean * mean
    return (var.max(axis=2) < max_std ** 2)


def foreground_mask(img: np.ndarray, bg: np.ndarray) -> np.ndarray:
    diff = np.abs(img.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    mask = (diff > BG_TOLERANCE).astype(np.uint8) * 255
    # No morphological closing here: stacked bubbles can sit only 1-2 px apart.
    # Fill holes so each bubble becomes one solid blob.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def blend_distance(pixels: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """Distance of each pixel from the colour blends fill->white and fill->black.

    Chat text is white or black on the bubble fill, so text and its anti-aliased (or JPEG
    blurred) edges lie on those blends, whatever the theme. `pixels` is (..., 3); `fill`
    broadcasts against it (one colour, or one per row for gradients)."""
    p = pixels.astype(np.float32)
    f = np.asarray(fill, np.float32)
    best = np.full(p.shape[:-1], np.inf, np.float32)
    for end in (255.0, 0.0):
        d = end - f
        t = np.clip(((p - f) * d).sum(-1) / np.maximum((d * d).sum(-1), 1.0), 0, 1)[..., None]
        best = np.minimum(best, np.linalg.norm(p - (f + t * d), axis=-1))
    return best


def _plane_fill(sub: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Fill colour as a plane over the region (a + b*x + c*y per channel), fitted to the
    fill pixels: refitting without the worst-fitting pixels drops the text. Solid fills come
    out flat; Instagram's diagonal pink-to-blue gradient fits as a slope."""
    ys, xs = np.nonzero(inside)
    px = sub[inside].astype(np.float32)
    A = np.stack([np.ones(len(xs), np.float32), xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    keep = np.ones(len(px), bool)
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A[keep], px[keep], rcond=None)
        resid = np.abs(px - A @ coef).max(axis=1)
        keep = resid <= max(2 * FILL_TOLERANCE, float(np.percentile(resid, 60)))
    yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    return coef[0] + xx[..., None] * coef[1] + yy[..., None] * coef[2]


def _fill_share(sub: np.ndarray, sub_mask: np.ndarray) -> tuple[np.ndarray, float, np.ndarray | None]:
    """How much of a region is bubble: (fill, share, fill_map or None).

    share = fraction of the region's pixels explained by a text bubble: the fill colour, or
    text drawn on it (on the fill->white/black blend). Exact colour bins measured only
    0.61-0.73 on a JPG phone screenshot (noise, blurred text). Gradient bubbles (Instagram)
    have no single fill, so a colour plane is fitted too; when it explains the region
    clearly better, the per-pixel fill map is returned (emoji detection needs it). Photos and
    avatars fit neither."""
    inside = sub_mask > 0
    fill, _ = _dominant_color(sub[inside])
    flat = float((blend_distance(sub[inside], fill) <= FILL_TOLERANCE).mean())
    plane = _plane_fill(sub, inside)
    by_plane = float((blend_distance(sub, plane)[inside] <= FILL_TOLERANCE).mean())
    if by_plane > flat + 0.1:  # a gradient
        centre = plane[sub.shape[0] // 2, sub.shape[1] // 2]
        return np.clip(centre, 0, 255).astype(np.int32), by_plane, plane
    return fill, flat, None


def _split_stacked(raw: np.ndarray, x: int, y: int, w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Bubbles stacked 2-3 px apart can merge into one region: JPEG blurs the gap shut. Split
    at nearly empty rows, and where a row's left or right edge jumps (stacked bubbles have
    different widths; inside one bubble the edges move only a few px per row, at the corners)."""
    sub = raw[y:y + h, x:x + w] > 0
    cover = sub.mean(axis=1)
    jump = max(12, int(0.06 * raw.shape[1]))
    has = sub.any(axis=1)
    left = np.where(has, sub.argmax(axis=1), w)
    right = np.where(has, w - 1 - sub[:, ::-1].argmax(axis=1), -1)
    parts, start = [], None
    for r in range(h + 1):
        gap = r == h or cover[r] < GAP_COVERAGE
        edge = (not gap and start is not None and r > start
                and (abs(int(left[r]) - int(left[r - 1])) > jump or abs(int(right[r]) - int(right[r - 1])) > jump))
        if start is not None and (gap or edge):
            parts.append((start, r))
            start = None
        if start is None and not gap:
            start = r
    if len(parts) <= 1:
        return [(x, y, w, h)]
    out = []
    for r0, r1 in parts:
        cols = np.where(sub[r0:r1].mean(axis=0) >= GAP_COVERAGE)[0]
        if len(cols):
            out.append((x + int(cols[0]), y + r0, int(cols[-1] - cols[0] + 1), r1 - r0))
    return out


def _filled(mask: np.ndarray) -> np.ndarray:
    """Fill holes (the text) so each bubble becomes one solid blob."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _candidates(img: np.ndarray, mask: np.ndarray, raw: np.ndarray, bg: np.ndarray) -> list[Bubble]:
    """Bubbles among the blobs of a filled mask (`raw`: the same mask before hole filling)."""
    H, W = img.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        for x, y, w, h in _split_stacked(raw, *cv2.boundingRect(c)):
            if w < 0.04 * W or h < max(16, 0.012 * H):
                continue
            if w > 0.92 * W:  # header, input bar, full-width banners
                continue
            if y <= 1 or y + h >= H - 1:  # clipped by the screenshot edge, text is cut off
                continue
            if x <= 1 or x + w >= W - 1:  # bubbles never touch the side edges (wallpaper does)
                continue
            if y + h > 0.93 * H and w > 0.7 * W:  # the message input bar
                continue
            if (mask[y:y + h, x:x + w] > 0).mean() < MIN_RECTANGULARITY:
                continue  # not a rounded rectangle: big emoji, avatar, icon
            fill, ratio, fill_map = _fill_share(img[y:y + h, x:x + w], mask[y:y + h, x:x + w])
            if ratio < MIN_FILL_RATIO:
                continue  # no solid fill: avatar, image, loose text
            if np.abs(fill.astype(int) - bg.astype(int)).max() <= BG_TOLERANCE:
                continue
            out.append(Bubble(Box(x, y, w, h), "", tuple(int(v) for v in fill), float(fill.mean()) < 110, fill_map))
    return out


def _fill_colours(img: np.ndarray, bg: np.ndarray, top: int = 4) -> list[np.ndarray]:
    """The most common flat colours other than the background: bubble fills (and, on a
    patterned wallpaper, the doodle colour, whose blobs fail the shape checks)."""
    flat = uniform_mask(img) & (np.abs(img.astype(np.int16) - bg.astype(np.int16)).max(axis=2) > BG_TOLERANCE)
    px = img[flat]
    if len(px) == 0:
        return []
    q = (px // (QUANT * 4)).astype(np.int32)
    keys = (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    out = []
    for k in values[np.argsort(-counts)][:top]:
        sel = px[keys == k]
        if len(sel) > 0.005 * img.shape[0] * img.shape[1]:
            out.append(np.median(sel, axis=0))
    return out


def _overlap(a: Box, b: Box) -> float:
    ix = max(0, min(a.right, b.right) - max(a.x, b.x))
    iy = max(0, min(a.bottom, b.bottom) - max(a.y, b.y))
    return ix * iy / max(1, min(a.w * a.h, b.w * b.h))


def detect_bubbles(img: np.ndarray) -> list[Bubble]:
    """Return bubbles sorted top to bottom. `img` is a BGR image.

    Two passes. (1) Everything that differs from the background colour: works on plain
    backgrounds, including gradient bubbles. (2) Per bubble colour: pixels of one of the most
    common flat non-background colours. On a patterned wallpaper (WhatsApp doodles) pass 1
    sees the doodles too and fuses all bubbles into one blob; pass 2 keeps only bubble-coloured
    pixels. Boxes found by both passes are kept once."""
    H, W = img.shape[:2]
    bg = background_color(img)
    raw = (np.abs(img.astype(np.int16) - bg.astype(np.int16)).max(axis=2) > BG_TOLERANCE).astype(np.uint8) * 255
    bubbles = _candidates(img, foreground_mask(img, bg), raw, bg)
    # Opening (erode then dilate) erases anything thinner than the kernel: the wallpaper's
    # doodle strokes, which in WhatsApp dark mode are nearly the incoming bubble's grey.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for colour in _fill_colours(img, bg):
        near = (np.abs(img.astype(np.int16) - colour.astype(np.int16)).max(axis=2) <= FILL_TOLERANCE)
        near = near.astype(np.uint8) * 255
        solid = cv2.morphologyEx(near, cv2.MORPH_OPEN, k)  # keeps shapes and the gaps between bubbles
        for b in _candidates(img, _filled(solid), solid, bg):
            if not any(_overlap(b.box, o.box) > 0.5 for o in bubbles):
                bubbles.append(b)

    _assign_sides(bubbles, W)
    bubbles = _drop_off_palette(bubbles)
    bubbles.sort(key=lambda b: (b.box.y, b.box.x))
    return bubbles


def _assign_sides(bubbles: list[Bubble], W: int) -> None:
    """Left messages share a left edge and right messages a right edge, so a bubble belongs
    to the edge it lines up with. "Nearest screen edge" put a long left message
    ("You returning home tomorrow?") on the right."""
    if not bubbles:
        return
    lo = min(b.box.x for b in bubbles)
    hi = max(b.box.right for b in bubbles)
    for b in bubbles:
        dl, dr = b.box.x - lo, hi - b.box.right
        if abs(dl - dr) <= 4:  # spans the whole chat width: fall back to the screen margins
            b.side = "right" if (W - b.box.right) < b.box.x else "left"
        else:
            b.side = "left" if dl < dr else "right"


def _saturated(fill) -> bool:
    return max(fill) - min(fill) > 60


def _drop_off_palette(bubbles: list[Bubble]) -> list[Bubble]:
    """Each side of a chat uses one bubble colour. Anything else (big emoji,
    stickers, link previews, quoted replies) is not a text bubble. A side drawn with a
    gradient has no single colour: there, its colourful bubbles are kept and grey ones
    (quoted-reply previews) dropped."""
    kept = []
    for side in ("left", "right"):
        group = [b for b in bubbles if b.side == side]
        if not group:
            continue
        gradient = [b for b in group if b.fill_map is not None]
        if sum(b.box.w * b.box.h for b in gradient) > sum(b.box.w * b.box.h for b in group) / 2:
            # Short bubbles show little slope; quoted-reply previews are plain grey.
            kept += [b for b in group if b.fill_map is not None or _saturated(b.fill)]
            continue
        weights: dict[tuple, int] = {}
        for b in group:
            weights[b.fill] = weights.get(b.fill, 0) + b.box.w * b.box.h
        palette = max(weights, key=weights.get)
        if _saturated(palette):  # a coloured side (often a gradient: pink at the top, blue at the bottom)
            kept += [b for b in group if _saturated(b.fill)]
            continue
        kept += [b for b in group
                 if max(abs(a - c) for a, c in zip(b.fill, palette)) <= 2 * BG_TOLERANCE]
    return kept
