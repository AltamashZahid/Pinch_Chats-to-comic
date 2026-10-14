import os

import numpy as np
import pytest

from pinch import emoji as E
from pinch.bubbles import background_color, detect_bubbles
from pinch.models import Box, Message
from pinch.pipeline import lines_from_messages
from pinch.scenes import Line, write_script
from tests.synth import render_chat

slow = pytest.mark.skipif(os.environ.get("PINCH_SKIP_SLOW") == "1", reason="loads CLIP ViT-L")


# --- tone rules ---------------------------------------------------------------------------

def test_no_emoji_keeps_text_tone():
    assert E.combine("anger", 0.7, []).emotion == "anger"


def test_emoji_only_message_takes_emoji_tone():
    t = E.combine("neutral", 0.0, ["😭"], has_text=False)
    assert t.emotion == "sadness" and not t.ironic


def test_neutral_text_takes_emoji_tone():
    t = E.combine("neutral", 0.9, ["😡"])
    assert t.emotion == "anger" and not t.ironic


def test_agreeing_emoji_strengthens():
    t = E.combine("joy", 0.6, ["😂"])
    assert t.emotion == "joy" and t.confidence >= 0.9 and not t.ironic


def test_opposite_emoji_is_ironic():
    t = E.combine("sadness", 0.8, ["😂", "😂"])
    assert t.ironic and t.emotion == "joy" and t.situation == "sadness"


def test_sarcasm_emoji_on_bad_news_is_ironic():
    t = E.combine("anger", 0.8, ["🙂"])
    assert t.ironic and t.situation == "anger"


def test_majority_emoji_wins():
    assert E.emoji_emotion(["😭", "😂", "😂"]) == "joy"


def test_text_helpers():
    assert E.emoji_in_text("lol 😂 ok 😭") == ["😂", "😭"]
    assert E.strip_emoji("lol 😂 ok") == "lol ok"


# --- detection (no CLIP needed) -----------------------------------------------------------

def test_finds_emoji_inside_bubble_but_not_in_plain_text():
    img = render_chat([("left", "I failed physics 😂"), ("right", "no emoji here")], header=False)
    found = []
    for b in detect_bubbles(img):
        crop = img[b.box.y:b.box.bottom, b.box.x:b.box.right]
        found.append(len(E.find_in_bubble(crop, b.fill)))
    assert found == [1, 0]


@pytest.mark.parametrize("theme", ["whatsapp_light", "whatsapp_dark", "imessage"])
def test_saturated_bubble_fill_is_not_an_emoji(theme):
    img = render_chat([("right", "hello there friend")], theme=theme, header=False)
    b = detect_bubbles(img)[0]
    assert E.find_in_bubble(img[b.box.y:b.box.bottom, b.box.x:b.box.right], b.fill) == []


def test_finds_big_loose_emoji_and_their_side():
    img = render_chat([("left", "hi"), ("left-big", "😘😘😘"), ("right", "aww")], header=False)
    bubbles = detect_bubbles(img)
    loose = E.find_loose(img, background_color(img), [b.box for b in bubbles],
                         min(b.box.x for b in bubbles if b.side == "left"))
    assert len(loose) == 3
    assert all(x < img.shape[1] / 2 for x, _, _, _ in loose)


# --- scenes -------------------------------------------------------------------------------

def test_ironic_panel_prompt_shows_the_contrast(monkeypatch):
    monkeypatch.setenv("PINCH_REVEAL", "off")  # single-panel version of the gag
    lines =[Line("me", "I failed physics", "joy", 0.9, emoji=["😂"], ironic=True, situation="sadness"),
             Line("them", "oh no", "sadness", 0.6)]
    p = write_script(lines, mode="story").panels[0]
    assert p.focus == "me"
    assert "happy face" in p.prompt and "falls apart" in p.prompt
    assert lines[0].bubble == "wavy"


def test_laugh_over_bad_news_is_two_beats():
    p = write_script([Line("me", "I failed physics", "joy", 0.9, emoji=["😂"], ironic=True,
                           situation="sadness")], mode="story").panels[0]
    assert p.reveal == "sadness"
    assert "happy face" in p.prompt and "falls apart" not in p.prompt
    assert "crying face" in p.reveal_prompt and "close-up" in p.reveal_prompt


def test_happy_tears_stay_one_panel():
    p = write_script([Line("me", "I got the job", "sadness", 0.9, emoji=["😭"], ironic=True,
                           situation="joy")], mode="story").panels[0]
    assert p.reveal == "" and p.reveal_prompt == ""


def test_reveal_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("PINCH_REVEAL", "off")
    p = write_script([Line("me", "I failed physics", "joy", 0.9, emoji=["😂"], ironic=True,
                           situation="sadness")], mode="story").panels[0]
    assert p.reveal == "" and "falls apart" in p.prompt


def test_two_beats_splits_the_frame():
    from PIL import Image
    from pinch.layout import two_beats
    out = np.asarray(two_beats(Image.new("RGB", (512, 512), "red"), Image.new("RGB", (512, 512), "blue")))
    assert tuple(out[256, 100]) == (255, 0, 0) and tuple(out[256, 400]) == (0, 0, 255)


def test_lines_from_messages_merges_emoji(monkeypatch):
    monkeypatch.setattr("pinch.pipeline.classify_dialogue", lambda turns: [("sadness", 0.8)] * len(turns))
    msgs = [Message("me", "I failed physics", Box(0, 0, 1, 1), ["😂"]),
            Message("them", "", Box(0, 0, 1, 1), ["😭"])]
    a, b = lines_from_messages(msgs)
    assert a.ironic and a.emotion == "joy" and a.situation == "sadness"
    assert b.emotion == "sadness" and not b.ironic


def test_bubble_draws_emoji_in_colour():
    from PIL import Image
    from pinch.layout import draw_bubble
    img = Image.new("RGB", (512, 512), "white")
    draw_bubble(img, "lol", "round", "left", 10, emoji=["😂"])
    a = np.asarray(img).astype(int)
    assert ((a.max(2) - a.min(2)) > 80).sum() > 100  # colourful pixels from the emoji


# --- recognition (loads CLIP ViT-L) -------------------------------------------------------

@slow
def test_end_to_end_reads_text_and_emoji(monkeypatch):
    monkeypatch.setenv("PINCH_EMOJI", "on")
    from pinch.ocr import extract_dialogue, tesseract_available
    if not tesseract_available():
        pytest.skip("tesseract not installed")
    img = render_chat([("left", "I failed physics 😂"), ("right", "omg"), ("left-big", "😭😭")],
                      header=False)
    msgs = extract_dialogue(img)
    assert [m.speaker for m in msgs] == ["them", "me", "them"]
    assert msgs[0].text.lower() == "i failed physics"
    assert E.EMOTION_OF[msgs[0].emoji[0]] == "joy"
    assert msgs[2].text == "" and len(msgs[2].emoji) == 2
    assert all(E.EMOTION_OF[e] == "sadness" for e in msgs[2].emoji)
