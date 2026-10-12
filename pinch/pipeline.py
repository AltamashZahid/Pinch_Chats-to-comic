"""Screenshot -> dialogue -> emotions -> comic script."""
import os
from pathlib import Path

import numpy as np

from .classifier import DEFAULT_MODEL_DIR
from .emoji import combine
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
    return write_script(lines_from_messages(messages), LLMClient.from_env())


def lines_from_messages(messages) -> list[Line]:
    """Text tone from the classifier, merged with the tone of any emoji (see emoji.combine)."""
    texts = [m.text for m in messages if m.text]
    tones = iter(classify(texts)) if texts else iter(())
    lines = []
    for m in messages:
        text_emotion, conf = next(tones) if m.text else ("neutral", 0.0)
        t = combine(text_emotion, conf, m.emoji, has_text=bool(m.text))
        lines.append(Line(m.speaker, m.text, t.emotion, t.confidence, emoji=list(m.emoji),
                          ironic=t.ironic, situation=t.situation))
    return lines
