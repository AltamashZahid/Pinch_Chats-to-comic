import pytest

from pinch.bubbles import detect_bubbles
from tests.synth import THEMES, render_chat

CHAT = [
    ("left", "hey are you coming tonight?"),
    ("right", "yes!! wouldn't miss it"),
    ("right", "ok"),
    ("left", "great, bring snacks"),
]


@pytest.mark.parametrize("theme", list(THEMES))
def test_finds_every_bubble_on_the_right_side(theme):
    bubbles = detect_bubbles(render_chat(CHAT, theme))
    assert [b.side for b in bubbles] == [side for side, _ in CHAT]


@pytest.mark.parametrize("theme", list(THEMES))
def test_ignores_header_and_date_separator(theme):
    bubbles = detect_bubbles(render_chat(CHAT, theme, header=True))
    assert all(b.box.y > 90 for b in bubbles)
    assert len(bubbles) == len(CHAT)


def test_dark_mode_bubbles_flagged_dark():
    bubbles = detect_bubbles(render_chat(CHAT, "whatsapp_dark"))
    assert all(b.dark for b in bubbles)


def test_tiny_bubble_is_kept():
    bubbles = detect_bubbles(render_chat([("right", "k")], "whatsapp_light"))
    assert len(bubbles) == 1
