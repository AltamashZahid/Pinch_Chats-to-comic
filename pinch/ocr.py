"""OCR each detected bubble with Tesseract and turn the screenshot into a dialogue."""
import os
import re
import shutil

import cv2
import numpy as np
import pytesseract

from . import emoji as E
from .bubbles import background_color, detect_bubbles
from .models import Box, Bubble, Message

SPEAKER_BY_SIDE = {"left": "them", "right": "me"}


def emoji_enabled() -> bool:
    """Emoji reading is parked (off by default) while the core comic builder is finished.
    PINCH_EMOJI=on turns it back on."""
    return os.environ.get("PINCH_EMOJI", "off").lower() in ("on", "1", "true", "yes")
MIN_CONFIDENCE = 60

_TIMESTAMP = re.compile(r"\s*\d{1,2}[:.]\d{2}\s*([ap]\.?m\.?)?\s*[✓✔vV/]*\s*$", re.IGNORECASE)
# WhatsApp prints the time inside each bubble and OCR garbles it: "1:05 am ✓✓" came out as
# "4.95 am yw", "1-06 am", "10h am YW". A time token must contain a digit ("Yes I am" stays).
# A time inside the text (two bubbles merged into one region) marks where one message ends.
_TIME_TOKEN = r"(?=\S*\d)[\dOoIl|]{1,2}\s?[:.\-h]?\s?[\dOoIl|]{0,2}\s*[ap]\.?[mn]\b\.?"
_GARBLED_TIME = re.compile(r"\s+" + _TIME_TOKEN + r"\s*[\u2713\u2714/a-zA-Z]{0,3}\s*$", re.IGNORECASE)
_SAID_TIME = re.compile(r"\b(at|by|till|until|around|before|after|from|to|@)\s*$", re.IGNORECASE)
_INNER_TIME = re.compile(r"\s+" + _TIME_TOKEN + r"\s*[\u2713\u2714/]*\s+(?=[A-Z])", re.IGNORECASE)
_WS = re.compile(r"\s+")


def _configure_tesseract() -> None:
    cmd = os.environ.get("PINCH_TESSERACT_CMD")
    if not cmd and not shutil.which("tesseract"):
        for candidate in (r"D:\Tesseract-OCR\tesseract.exe",
                          r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
            if os.path.exists(candidate):
                cmd = candidate
                break
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


def tesseract_available() -> bool:
    _configure_tesseract()
    try:
        pytesseract.get_tesseract_version()
        return True
    except (pytesseract.TesseractNotFoundError, OSError):
        return False


def preprocess(crop: np.ndarray) -> np.ndarray:
    """Grayscale, upscale, binarise, and force dark text on a white background."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (binary == 0).mean() > 0.5:  # the bubble fill is the majority colour
        binary = 255 - binary
    return cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)


def clean_text(raw: str) -> str:
    text = _WS.sub(" ", raw).strip()
    text = " ".join("I" if w == "|" else w for w in text.split())  # Tesseract reads a lone "I" as "|"
    for pattern in (_GARBLED_TIME, _TIMESTAMP):
        m = pattern.search(text)
        if m and not _SAID_TIME.search(text[:m.start()]):  # "see you at 5 pm" keeps its time
            text = text[:m.start()]
    return text.strip()


def split_merged(text: str) -> list[str]:
    """Two stacked bubbles that merged into one region read as "first 1:07 am Second": the
    time printed inside the first bubble marks the boundary."""
    return [t.strip() for t in _INNER_TIME.split(text) if t.strip()]


def bubble_emoji(crop: np.ndarray, fill) -> tuple[np.ndarray, list[str]]:
    """Recognise emoji in a bubble crop and paint them out so Tesseract doesn't read junk."""
    boxes = E.find_in_bubble(crop, fill)
    if not boxes:
        return crop, []
    crops = [E.crop_on_gray(crop, box, fill, pad=2) for box in boxes]
    found = [hit.char for hit in E.recognizer().recognise(crops) if hit]
    clean = crop.copy()
    for x, y, w, h in boxes:
        clean[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2] = fill
    return clean, found


def read_bubble(img: np.ndarray, bubble: Bubble) -> tuple[str, float, list[str]]:
    """OCR one bubble. Returns (text, mean word confidence 0-100, emoji)."""
    b = bubble.box
    inset = max(2, min(b.w, b.h) // 12)  # skip the rounded corners / outline
    crop = img[b.y + inset:b.bottom - inset, b.x + inset:b.right - inset]
    crop, emojis = bubble_emoji(crop, bubble.fill) if emoji_enabled() else (crop, [])
    data = pytesseract.image_to_data(preprocess(crop), config="--psm 6",
                                     output_type=pytesseract.Output.DICT)
    words = [(w, float(c)) for w, c in zip(data["text"], data["conf"]) if w.strip() and float(c) >= 0]
    if not words:
        return "", 0.0, emojis
    text = clean_text(" ".join(w for w, _ in words))
    return text, sum(c for _, c in words) / len(words), emojis


# Standalone emoji have no bubble vouching for them, and a false one invents a message.
# Measured: real big emoji matched the gallery at 0.96; profile pictures at 0.73-0.84.
LOOSE_MIN_SIMILARITY = 0.90


def loose_emoji(img: np.ndarray, bubbles: list[Bubble]) -> list[Message]:
    """Big emoji sent on their own (no bubble). Emoji on one row form one message.

    Only below the first bubble's top and above the last bubble's bottom, with some margin:
    above the chat is the app's header (its profile picture was "recognised" as 😃). With no
    bubbles there is no chat at all: a screenshot the detector failed on became a comic made
    of avatars read as emoji."""
    if not bubbles:
        return []
    bg = background_color(img)
    lefts = [b.box.x for b in bubbles if b.side == "left"]
    top = min(b.box.y for b in bubbles) - 3 * max(b.box.h for b in bubbles)
    bottom = max(b.box.bottom for b in bubbles) + 3 * max(b.box.h for b in bubbles)
    boxes = [b for b in E.find_loose(img, bg, [b.box for b in bubbles], min(lefts) if lefts else 0)
             if top <= b[1] and b[1] + b[3] <= bottom]
    if not boxes:
        return []
    fill = tuple(int(v) for v in bg)
    crops = [E.crop_on_gray(img, box, fill) for box in boxes]
    rows: list[list[tuple]] = []
    for box, hit in zip(boxes, E.recognizer().recognise(crops)):
        if not hit or hit.similarity < LOOSE_MIN_SIMILARITY:
            continue
        x, y, w, h = box
        row = next((r for r in rows if abs(r[0][0][1] - y) < h / 2), None)
        (row.append((box, hit)) if row is not None else rows.append([(box, hit)]))
    W = img.shape[1]
    out = []
    for row in rows:
        x0 = min(b[0] for b, _ in row); y0 = min(b[1] for b, _ in row)
        x1 = max(b[0] + b[2] for b, _ in row); y1 = max(b[1] + b[3] for b, _ in row)
        side = "right" if (W - x1) < x0 else "left"
        out.append(Message(SPEAKER_BY_SIDE[side], "", Box(x0, y0, x1 - x0, y1 - y0),
                           [hit.char for _, hit in sorted(row, key=lambda r: r[0][0])]))
    return out


def extract_dialogue(img: np.ndarray) -> list[Message]:
    """BGR screenshot -> ordered list of messages with speakers."""
    _configure_tesseract()
    messages = []
    bubbles = detect_bubbles(img)
    for bubble in bubbles:
        text, conf, emojis = read_bubble(img, bubble)
        if conf < MIN_CONFIDENCE:  # kaomoji and leftovers OCR as low-confidence junk
            text = ""
        parts = split_merged(text) if text else [""]
        b = bubble.box
        for k, part in enumerate(parts):  # a merged pair becomes two messages, stacked
            box = Box(b.x, b.y + b.h * k // len(parts), b.w, b.h // len(parts))
            if part or emojis:
                messages.append(Message(SPEAKER_BY_SIDE[bubble.side], part, box, emojis if k == len(parts) - 1 else []))
    if emoji_enabled():
        messages += loose_emoji(img, bubbles)
    messages.sort(key=lambda m: (m.box.y, m.box.x))
    return drop_quotes(messages)


def drop_quotes(messages: list[Message]) -> list[Message]:
    """A reply that quotes an earlier message shows the quote inside its bubble ("faiz wellness
    Jab hum bags ... Kab denge"). Keep only what follows the quoted text."""
    for i, m in enumerate(messages):
        for prev in messages[:i]:
            if len(prev.text) >= 12 and prev.text in m.text and m.text != prev.text:
                m.text = m.text.split(prev.text, 1)[1].strip()
    return messages


def load_image(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read image: {path}")
    return img
