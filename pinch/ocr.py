"""OCR each detected bubble with Tesseract and turn the screenshot into a dialogue."""
import os
import re
import shutil

import cv2
import numpy as np
import pytesseract

from .bubbles import detect_bubbles
from .models import Bubble, Message

SPEAKER_BY_SIDE = {"left": "them", "right": "me"}

_TIMESTAMP = re.compile(r"\s*\d{1,2}[:.]\d{2}\s*([ap]\.?m\.?)?\s*[✓✔vV/]*\s*$", re.IGNORECASE)
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


def preprocess(crop: np.ndarray, dark: bool) -> np.ndarray:
    """Grayscale, make text dark-on-light, upscale and binarise for Tesseract."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if dark:
        gray = 255 - gray
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)


def clean_text(raw: str) -> str:
    text = _WS.sub(" ", raw).strip()
    return _TIMESTAMP.sub("", text).strip()


def read_bubble(img: np.ndarray, bubble: Bubble) -> str:
    b = bubble.box
    inset = max(2, min(b.w, b.h) // 12)  # skip the rounded corners / outline
    crop = img[b.y + inset:b.bottom - inset, b.x + inset:b.right - inset]
    raw = pytesseract.image_to_string(preprocess(crop, bubble.dark), config="--psm 6")
    return clean_text(raw)


def extract_dialogue(img: np.ndarray) -> list[Message]:
    """BGR screenshot -> ordered list of messages with speakers."""
    _configure_tesseract()
    messages = []
    for bubble in detect_bubbles(img):
        text = read_bubble(img, bubble)
        if text:
            messages.append(Message(SPEAKER_BY_SIDE[bubble.side], text, bubble.box))
    return messages


def load_image(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read image: {path}")
    return img
