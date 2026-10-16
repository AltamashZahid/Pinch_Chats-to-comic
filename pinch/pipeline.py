"""Screenshot or chat text -> dialogue -> emotions -> comic script."""
import os
from pathlib import Path

import numpy as np

from .classifier import DEFAULT_MODEL_DIR
from .emoji import combine
from .llm import LLMClient
from .ocr import extract_dialogue
from .scenes import Line, Script, is_shouting, write_script

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


def classify(texts: list[str], prevs: list[str] | None = None) -> list[tuple[str, float]]:
    global _classifier
    if not (DEFAULT_MODEL_DIR / "config.json").exists():
        print("[pinch] no trained emotion model in models/emotion, everything is neutral "
              "(run training/train_emotion.py)")
        return [("neutral", 1.0)] * len(texts)
    if _classifier is None:
        from .classifier import EmotionClassifier
        _classifier = EmotionClassifier()
    return _classifier.predict(texts, prevs)


SPEAKER_NAME = {"me": "A", "them": "B"}


def classify_dialogue(turns: list[tuple[str, str]]) -> list[tuple[str, float]]:
    global _classifier
    if not (DEFAULT_MODEL_DIR / "config.json").exists():
        return [("neutral", 1.0)] * len(turns)
    if _classifier is None:
        from .classifier import EmotionClassifier
        _classifier = EmotionClassifier()
    return _classifier.predict_dialogue(turns)


_snapper = None


def library_snapper():
    """With the pre-drawn panel library on, characters are snapped to its presets."""
    global _snapper
    if os.environ.get("PINCH_IMAGE_BACKEND", "").lower() != "library":
        return None
    if _snapper is None:
        from .library import Snapper
        _snapper = Snapper()
    return _snapper.snap_cast


def script_with_memory(lines: list[Line], names: dict[str, str] | None = None,
                       people: dict[str, str] | None = None) -> Script:
    """Write the script with what memory knows about these people, then remember this comic.
    `names`: display names shown to the LLM. `people`: memory identities per role
    ("me" is always the app's user, memory.SELF)."""
    from .memory import default_store
    store = default_store()
    people = people or {}
    looks, history = {}, []
    if store is not None and people:
        looks = {role: card.look for role, card in store.cast_for(people).items()}
        history = store.history_for(people)
    script = write_script(lines, LLMClient.from_env(), names=names, looks=looks, history=history,
                          snap=library_snapper())
    if store is not None and people and script.source == "llm":  # template looks are generic, not a person
        store.remember(people, script.characters, script.summary)
    return script


def screenshot_to_script(img: np.ndarray, them: str | None = None) -> Script:
    """`them`: the other person's name (typed by the user, or from a chat export), so memory
    can recognise them across comics. Without it only the user is remembered."""
    from .memory import SELF
    load_env()
    messages = extract_dialogue(img)
    people = {"me": SELF, **({"them": them} if them else {})}
    names = {"them": them} if them else None
    return script_with_memory(lines_from_messages(messages), names, people)


MAX_CHAT_MESSAGES = 12  # 6 panels of 2 lines; longer chats use their most recent messages


def chat_to_script(msgs: list[dict], me: str | None, last: int = MAX_CHAT_MESSAGES) -> Script:
    """Typed chat (WhatsApp export / pasted "Name: text" lines) -> comic script.
    `me` is the user's own name in the chat; everyone else becomes "them"."""
    from .models import Box, Message
    load_env()
    msgs = [m for m in msgs if m["text"]][-last:]
    if me is None:  # assume a two-person chat where the user is whoever didn't speak first
        names = list(dict.fromkeys(m["speaker"] for m in msgs))
        me = names[1] if len(names) > 1 else names[0]
    from .emoji import emoji_in_text, strip_emoji
    # Typed emoji count like emoji read from a screenshot: "I failed 😂" is an ironic laugh.
    messages = [Message("me" if m["speaker"] == me else "them", strip_emoji(m["text"]), Box(0, i, 1, 1),
                        emoji_in_text(m["text"])) for i, m in enumerate(msgs)]
    others = [n for n in dict.fromkeys(m["speaker"] for m in msgs) if n != me]
    names = {"me": me, "them": others[0] if len(others) == 1 else "a friend"}
    from .memory import SELF
    people = {"me": SELF, **({"them": others[0]} if len(others) == 1 else {})}  # group chats: no single "them"
    return script_with_memory(lines_from_messages(messages), names, people)


def lines_from_messages(messages) -> list[Line]:
    """Text tone from the classifier, merged with the tone of any emoji (see emoji.combine)."""
    # The whole chat goes to the classifier at once: conversation models read each message
    # with the ones before and after it. ALL CAPS confuses the classifier, so it reads a
    # lowercased copy; the bubble keeps the original and the shouting itself is drawn.
    turns = [(SPEAKER_NAME.get(m.speaker, m.speaker), m.text.lower() if is_shouting(m.text) else m.text)
             for m in messages if m.text]
    tones = iter(classify_dialogue(turns)) if turns else iter(())
    lines = []
    for m in messages:
        text_emotion, conf = next(tones) if m.text else ("neutral", 0.0)
        t = combine(text_emotion, conf, m.emoji, has_text=bool(m.text))
        lines.append(Line(m.speaker, m.text, t.emotion, t.confidence, emoji=list(m.emoji),
                          ironic=t.ironic, situation=t.situation))
    return lines
