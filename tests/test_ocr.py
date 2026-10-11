import pytest

from pinch.ocr import clean_text, extract_dialogue, tesseract_available
from tests.synth import THEMES, render_chat

needs_tesseract = pytest.mark.skipif(not tesseract_available(), reason="tesseract not installed")

CHAT = [
    ("left", "did you finish the report"),
    ("right", "not yet, almost done"),
    ("left", "the deadline is today"),
]


@pytest.mark.parametrize("raw, expected", [
    ("hello there 10:42 pm", "hello there"),
    ("see you\nsoon 9:05 ✓✓", "see you soon"),
    ("meet at 5:30 then?", "meet at 5:30 then?"),
    ("  lots   of\n\nspace ", "lots of space"),
])
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


@needs_tesseract
@pytest.mark.parametrize("theme", list(THEMES))
def test_extracts_speakers_and_text(theme):
    messages = extract_dialogue(render_chat(CHAT, theme))
    assert [m.speaker for m in messages] == ["them", "me", "them"]
    assert [m.text.lower() for m in messages] == [t for _, t in CHAT]
