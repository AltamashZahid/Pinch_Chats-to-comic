"""Emoji: find them in screenshots, recognise them, and turn them into a tone.

Tesseract can't read emoji (it turns them into junk like "\\.)"), so emoji are handled
as images:
- find_in_bubble(): inside a text bubble, emoji are the saturated pixels whose hue differs
  from the bubble fill (chat text is white/black, so it's never saturated).
- find_loose(): big emoji sent on their own have no bubble; they're colourful, square
  blobs in the message column.
- EmojiRecognizer: CLIP ViT-L/14 image embeddings of each crop vs a gallery of reference
  emoji in several art styles (Windows Segoe font + Twemoji/OpenMoji PNGs in
  data/emoji_gallery/), nearest neighbour above a similarity threshold. Apps draw emoji
  differently, so the gallery mixes styles; tested leave-one-style-out in
  eval/emoji_recognition.py.

combine() merges an emoji's emotion with the text classifier's: agreement strengthens it,
neutral text takes the emoji's tone, and opposite tones (or a sarcasm emoji like 🙃) mark
the line ironic, which the comic plays for laughs.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# emoji -> emotion. Covers the most-used emoji; anything else falls back to the text.
EMOJI_EMOTION = {
    "joy": "😂🤣😆😄😁😀😃😊😍🥰😘😗😙😚☺❤💕💖💗💓💞💘😎🥳🎉👍🙌👏🤗😋😜😝😛🤪😇✨🔥💯💀😹",
    "sadness": "😭😢😞😔😟☹🙁😿💔🥺😩😫",
    "anger": "😡😠🤬😤👿💢😒🙄",
    "fear": "😨😰😱😧😥",
    "surprise": "😮😯😲😳🤯😵😦👀",
    "disgust": "🤢🤮😖😣🤧",
    "neutral": "😐😑😶🤔",
}
SARCASM = "🙃🙂🥲😅🫠"  # a smile on top of something else: the classic ironic emoji
EMOTION_OF = {e: emo for emo, chars in EMOJI_EMOTION.items() for e in chars}
EMOTION_OF.update({e: "joy" for e in SARCASM})  # what the face shows
VALENCE = {"joy": 1, "sadness": -1, "anger": -1, "fear": -1, "disgust": -1, "surprise": 0, "neutral": 0}

FONT = r"C:\Windows\Fonts\seguiemj.ttf"
GALLERY_DIR = Path(__file__).resolve().parent.parent / "data" / "emoji_gallery"
CLIP_MODEL = "openai/clip-vit-large-patch14"  # ViT-L: 86% vs 71% emotion acc on unseen styles
GRAY = (128, 128, 128)
MIN_DENSITY = 0.5       # share of a candidate box that must be saturated colour
# Detection (colour density + position) decides whether something is an emoji; recognition
# only picks which one. Real in-bubble emoji matched at 0.78-0.98, so keep this low.
MIN_SIMILARITY = 0.70


def emoji_in_text(text: str) -> list[str]:
    """Unicode emoji already in a string (typed input, LLM tests)."""
    return [ch for ch in text if ch in EMOTION_OF]


def strip_emoji(text: str) -> str:
    return " ".join("".join(ch for ch in text if ch not in EMOTION_OF and ch != "\ufe0f").split())


# --- detection ----------------------------------------------------------------------------

def _hue_gap(h: np.ndarray, ref: float) -> np.ndarray:
    d = np.abs(h.astype(np.int16) - int(ref))
    return np.minimum(d, 180 - d)


def emoji_mask(crop: np.ndarray, fill: tuple[int, int, int]) -> np.ndarray:
    """Pixels that belong to colourful glyphs (emoji) rather than the fill or the text."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    fh, fs, _ = cv2.cvtColor(np.uint8([[fill]]), cv2.COLOR_BGR2HSV)[0, 0]
    colourful = (hsv[..., 1] > 90) & (hsv[..., 2] > 80)
    if fs > 60:  # saturated bubble (e.g. Instagram blue): its anti-aliased edges share its hue
        colourful &= _hue_gap(hsv[..., 0], fh) > 18
    return colourful.astype(np.uint8) * 255


def _blobs(mask: np.ndarray, min_side: int) -> list[tuple[int, int, int, int]]:
    k = max(3, min_side // 4) | 1
    merged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(merged)
    out = []
    for x, y, w, h, _ in stats[1:]:
        # Solidly coloured: real emoji measured 0.76 of the box; ClearType colour fringes on
        # plain text (Windows screenshots) measured <= 0.34.
        density = (mask[y:y + h, x:x + w] > 0).mean()
        if min(w, h) >= min_side and 0.55 <= w / h <= 1.8 and density >= MIN_DENSITY:
            out.append((int(x), int(y), int(w), int(h)))
    return sorted(out, key=lambda b: b[0])


def find_in_bubble(crop: np.ndarray, fill: tuple[int, int, int]) -> list[tuple[int, int, int, int]]:
    """Emoji boxes (x, y, w, h) inside a bubble crop."""
    return _blobs(emoji_mask(crop, fill), min_side=max(10, crop.shape[0] // 5))


def find_loose(img: np.ndarray, bg: np.ndarray, taken: list, min_left: int) -> list[tuple[int, int, int, int]]:
    """Big standalone emoji outside bubbles. `taken` are bubble boxes to ignore; blobs that
    end left of `min_left` (the avatar column) are skipped."""
    mask = emoji_mask(img, tuple(int(v) for v in bg))
    for b in taken:
        mask[b.y:b.bottom, b.x:b.right] = 0
    H = img.shape[0]
    return [b for b in _blobs(mask, min_side=max(24, H // 40)) if b[0] + b[2] > min_left]


def crop_on_gray(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Put the emoji on a neutral background so screenshots and the gallery look alike."""
    m = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    filled = np.zeros_like(m)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cs, -1, 255, cv2.FILLED)  # keep eyes/mouth inside the face
    out = np.full_like(crop, GRAY[0])
    out[filled > 0] = crop[filled > 0]
    # Same framing as the gallery (glyph ~68% of a square frame), upscaled smoothly.
    h, w = out.shape[:2]
    side = int(max(h, w) / 0.68)
    framed = np.full((side, side, 3), GRAY[0], np.uint8)
    framed[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = out
    return cv2.resize(framed, (224, 224), interpolation=cv2.INTER_CUBIC)


# --- recognition --------------------------------------------------------------------------

def render_reference(ch: str, size: int = 160):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(FONT, 109)
    img = Image.new("RGB", (size, size), GRAY)
    d = ImageDraw.Draw(img)
    l, t, r, b = d.textbbox((0, 0), ch, font=font, embedded_color=True)
    d.text(((size - (r - l)) / 2 - l, (size - (b - t)) / 2 - t), ch, font=font, embedded_color=True)
    return img


def _load_png(path: Path):
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, GRAY + (255,))
    return Image.alpha_composite(bg, im).convert("RGB").resize((160, 160))


@dataclass
class EmojiHit:
    char: str
    emotion: str
    similarity: float


class EmojiRecognizer:
    def __init__(self, styles: tuple[str, ...] = ("segoe", "*")):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self._torch = torch
        self.model = CLIPModel.from_pretrained(CLIP_MODEL).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
        chars, images = [], []
        if "segoe" in styles:
            for ch in EMOTION_OF:
                chars.append(ch)
                images.append(render_reference(ch))
        if styles:
            for d in sorted(GALLERY_DIR.glob("*")) if GALLERY_DIR.exists() else []:
                if d.is_dir() and ("*" in styles or d.name in styles):
                    for ch in EMOTION_OF:
                        p = d / f"{ord(ch):x}.png"
                        if p.exists():
                            chars.append(ch)
                            images.append(_load_png(p))
        self.chars = chars
        self.gallery = self.embed(images) if images else np.zeros((0, 768), np.float32)

    def embed(self, images) -> np.ndarray:
        with self._torch.no_grad():
            px = self.proc(images=images, return_tensors="pt")["pixel_values"]
            e = self.model.visual_projection(self.model.vision_model(pixel_values=px).pooler_output)
            return self._torch.nn.functional.normalize(e, dim=-1).numpy()

    def recognise(self, images) -> list[EmojiHit | None]:
        """PIL images or BGR arrays -> best match per image, or None if nothing is close."""
        from PIL import Image
        pil = [Image.fromarray(cv2.cvtColor(i, cv2.COLOR_BGR2RGB)) if isinstance(i, np.ndarray) else i
               for i in images]
        if not pil:
            return []
        sims = self.embed(pil) @ self.gallery.T
        out = []
        for row in sims:
            j = int(row.argmax())
            ch = self.chars[j]
            out.append(EmojiHit(ch, EMOTION_OF[ch], float(row[j])) if row[j] >= MIN_SIMILARITY else None)
        return out


_recognizer: EmojiRecognizer | None = None


def recognizer() -> EmojiRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = EmojiRecognizer()
    return _recognizer


# --- tone ---------------------------------------------------------------------------------

@dataclass
class Tone:
    emotion: str            # what the character's face shows
    confidence: float
    ironic: bool = False
    situation: str = ""     # for ironic lines: what's really going on (the text's emotion)


def emoji_emotion(emojis: list[str]) -> str | None:
    if not emojis:
        return None
    votes: dict[str, int] = {}
    for e in emojis:
        votes[EMOTION_OF[e]] = votes.get(EMOTION_OF[e], 0) + 1
    return max(votes, key=votes.get)


def combine(text_emotion: str, text_conf: float, emojis: list[str], has_text: bool = True) -> Tone:
    face = emoji_emotion(emojis)
    if face is None:
        return Tone(text_emotion, text_conf)
    if not has_text:
        return Tone(face, 0.9)
    sarcastic = any(e in SARCASM for e in emojis)
    if sarcastic and VALENCE[text_emotion] < 0:
        return Tone("joy", 0.9, ironic=True, situation=text_emotion)
    if face == text_emotion:
        return Tone(face, max(text_conf, 0.9))
    if text_emotion == "neutral":
        return Tone(face, 0.85)          # the emoji is the tone
    if face == "neutral":
        return Tone(text_emotion, text_conf)
    if VALENCE[face] * VALENCE[text_emotion] < 0:  # e.g. "I failed physics 😂"
        return Tone(face, 0.9, ironic=True, situation=text_emotion)
    return Tone(face, 0.85)              # both negative or surprise: the emoji is the more deliberate signal
