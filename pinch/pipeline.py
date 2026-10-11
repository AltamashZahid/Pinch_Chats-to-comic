"""Screenshot -> dialogue -> emotions -> comic script."""
import os
from pathlib import Path

import numpy as np

from .classifier import DEFAULT_MODEL_DIR
from .llm import LLMClient
from .ocr import extract_dialogue
from .scenes import Line, Script, write_script

_classifier = None


def load_env(path: Path = Path(__file__).resolve().parent.parent / ".env") -> None:
    """Read KEY=VALUE lines from .env into os.environ (real env vars win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def classify(texts: list[str]) -> list[tuple[str, float]]:
    global _classifier
    if not (DEFAULT_MODEL_DIR / "config.json").exists():
        print("[pinch] no trained emotion model in models/emotion, everything is neutral "
              "(run training/train_emotion.py)")
        return [("neutral", 1.0)] * len(texts)
    if _classifier is None:
        from .classifier import EmotionClassifier
        _classifier = EmotionClassifier()
    return _classifier.predict(texts)


def screenshot_to_script(img: np.ndarray) -> Script:
    load_env()
    messages = extract_dialogue(img)
    emotions = classify([m.text for m in messages])
    lines = [Line(m.speaker, m.text, e, p) for m, (e, p) in zip(messages, emotions)]
    return write_script(lines, LLMClient.from_env())
