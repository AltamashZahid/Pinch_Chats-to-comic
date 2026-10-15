"""Pre-drawn panel library: instant anime-quality panels instead of ~28 s live generation.

A fixed catalogue of character presets is drawn ahead of time (training/build_library.py)
in every emotion, framing and setting:

    data/library/<preset>/<emotion>_<framing>_<setting>_v<n>.png

At comic time, a character's look (written by the LLM) is snapped to the nearest preset of
the same gender, and each panel is served from the library. Memory then keeps that person on
the same preset in every later comic. Panels the library doesn't cover fall back to live
generation.

Trade-off: library panels show the character's emotion in a generic indoor or outdoor
setting, not the LLM's specific action. PINCH_IMAGE_BACKEND=library turns it on.
"""
import io
import json
import random
import re
from collections import deque
from pathlib import Path

from .style import ART_STYLE, EXPRESSION, STORY_POSE

ROOT = Path(__file__).resolve().parent.parent
LIBRARY_DIR = ROOT / "data" / "library"

PRESETS = {  # id -> look; hair, eyes and outfit differ so characters are easy to tell apart
    "m1": "young man with short black hair, brown eyes, brown skin and a blue hoodie",
    "m2": "young man with messy brown hair, round glasses and a green t-shirt",
    "m3": "young man with spiky blonde hair, blue eyes and a black leather jacket",
    "m4": "young man with a black undercut, dark eyes, brown skin, a white shirt and a red scarf",
    "m5": "young man with curly black hair, light stubble and a grey sweater",
    "m6": "teenage boy with short red hair, freckles, an orange baseball cap and a striped t-shirt",
    "f1": "young woman with a long black ponytail, brown eyes, brown skin and a yellow tunic",
    "f2": "young woman with a short purple bob, grey eyes and a black leather jacket",
    "f3": "young woman with long wavy red hair, green eyes, a cream sweater and a star hairclip",
    "f4": "young woman with brown twin braids, round glasses and a pink cardigan",
    "f5": "young woman with long silver hair, blue eyes and a navy school blazer",
    "f6": "young woman with short black hair, brown skin, hoop earrings and a denim jacket",
}
LOOK_TO_PRESET = {look: pid for pid, look in PRESETS.items()}

FRAMING = {  # the app's camera shots folded into three drawn framings
    "extreme close-up": "close", "close-up": "close",
    "medium shot": "medium", "over-the-shoulder shot": "medium",
    "wide shot": "wide",
}
FRAMING_SHOT = {"close": "close-up", "medium": "medium shot", "wide": "wide shot"}
SETTINGS = {"indoor": "cozy living room, warm light", "outdoor": "city street, daytime"}
OUTDOOR_WORDS = {"street", "park", "outside", "outdoors", "beach", "road", "garden", "bus", "station",
                 "market", "campus", "rooftop", "field", "playground", "city", "sidewalk", "stadium"}


def gender(look: str) -> str:
    words = set(re.findall(r"[a-z]+", look.lower()))
    if words & {"woman", "girl", "she", "her", "lady", "female"}:
        return "f"
    if words & {"man", "boy", "he", "his", "guy", "male"}:
        return "m"
    return ""


def setting_of(location: str) -> str:
    return "outdoor" if set(re.findall(r"[a-z]+", location.lower())) & OUTDOOR_WORDS else "indoor"


def library_prompt(look: str, emotion: str, framing: str, setting: str) -> str:
    """Same structure as the app's panel prompts (style, shot, look, face, pose, place)."""
    return (f"{ART_STYLE}, {FRAMING_SHOT[framing]}, {look}, {EXPRESSION[emotion]}, "
            f"{STORY_POSE[emotion]}, {SETTINGS[setting]}")


def image_path(pid: str, emotion: str, framing: str, setting: str, variant: int = 0) -> Path:
    return LIBRARY_DIR / pid / f"{emotion}_{framing}_{setting}_v{variant}.png"


class Snapper:
    """Maps a free-text look to the nearest preset of the same gender (sentence embeddings)."""

    def __init__(self, embedder=None):
        self._embedder = embedder
        self._preset_vecs = None

    def _embed(self, texts):
        if self._embedder is None:
            from .cache import MiniLMEmbedder
            self._embedder = MiniLMEmbedder()
        return self._embedder(texts)

    def ranked(self, look: str) -> list[str]:
        """Preset ids, best match first; same-gender presets always ahead of the others."""
        if look in LOOK_TO_PRESET:
            return [LOOK_TO_PRESET[look]] + [p for p in PRESETS if p != LOOK_TO_PRESET[look]]
        if self._preset_vecs is None:
            self._preset_vecs = self._embed(list(PRESETS.values()))
        sims = self._preset_vecs @ self._embed([look])[0]
        g = gender(look)
        order = sorted(range(len(PRESETS)), key=lambda i: (list(PRESETS)[i][0] != g if g else 0, -sims[i]))
        return [list(PRESETS)[i] for i in order]

    def snap_cast(self, characters: dict[str, str]) -> dict[str, str]:
        """Snap every role to a preset look, giving two people two different presets."""
        out, taken = {}, set()
        for role, look in characters.items():
            pid = next(p for p in self.ranked(look) if p not in taken)
            taken.add(pid)
            out[role] = PRESETS[pid]
        return out


class LibraryBackend:
    """Serves pre-drawn panels; anything the library lacks is drawn live by `fallback`."""
    name = "library"

    def __init__(self, fallback=None, **_):
        if fallback is None:
            from .imagegen import LocalAnimeBackend
            fallback = LocalAnimeBackend()
        self.fallback = fallback
        self.model = f"library+{fallback.model}"
        self.hits = self.misses = 0
        self.recent: deque[Path] = deque(maxlen=8)  # a comic has at most 8 panels

    def _variants(self, pid: str, emotion: str, framing: str, setting: str) -> list[Path]:
        folder = LIBRARY_DIR / pid
        return sorted(f for f in folder.glob(f"{emotion}_{framing}_{setting}_v*.png")
                      if not f.name.endswith(".rejected.png"))

    def lookup(self, req) -> Path | None:
        """The asked-for framing first; a panel already shown recently (the same emotion twice
        in one comic) is swapped for another variant, then for another framing of the same
        emotion and place."""
        pid = LOOK_TO_PRESET.get(req.character or "")
        framing = FRAMING.get(req.shot or "")
        if pid is None or framing is None or req.emotion not in EXPRESSION:
            return None
        setting = setting_of(req.prompt)
        rng = random.Random(req.seed)
        tried = [framing] + [f for f in FRAMING_SHOT if f != framing]
        candidates = [self._variants(pid, req.emotion, f, setting) for f in tried]
        if not candidates[0]:
            return None
        for group in candidates:
            fresh = [p for p in group if p not in self.recent]
            if fresh:
                choice = rng.choice(fresh)
                break
        else:
            choice = rng.choice(candidates[0])
        self.recent.append(choice)
        return choice

    def generate(self, req) -> bytes:
        path = self.lookup(req)
        if path is None:
            self.misses += 1
            return self.fallback.generate(req)
        self.hits += 1
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if img.size != (req.width, req.height):
            img = img.resize((req.width, req.height), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()


def rejects() -> set[str]:
    """Library images marked bad in review (data/library/rejects.json): a list of entries,
    each a file name for every preset ("disgust_close_indoor_v0.png") or "<preset>/<file>"."""
    f = LIBRARY_DIR / "rejects.json"
    return set(json.loads(f.read_text())) if f.exists() else set()
