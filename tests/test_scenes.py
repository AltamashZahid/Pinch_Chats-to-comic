import json

import httpx

from pinch.llm import LLMClient, LLMError
from pinch.scenes import Line, group_panels, panel_emotion, write_script

LINES = [
    Line("them", "the deadline is today", "fear", 0.8),
    Line("me", "WHAT", "surprise", 0.9),
    Line("me", "i'm so dead", "sadness", 0.6),
    Line("them", "lol good luck", "joy", 0.7),
]

GOOD_REPLY = {
    "setting": "a messy college dorm room late at night",
    "characters": {"me": "tall guy with glasses and a grey hoodie",
                   "them": "girl with a ponytail and a red jacket"},
    "panels": ["girl typing on her phone on a bus", "guy jumps up from his desk, papers flying"],
}


def fake_llm(reply, status=200):
    def handler(request):
        return httpx.Response(status, json={"choices": [{"message": {"content": reply}}]})
    return LLMClient("https://fake/v1", "key", "model", transport=httpx.MockTransport(handler))


def test_group_panels_two_lines_each():
    assert [len(g) for g in group_panels(LINES)] == [2, 2]


def test_group_panels_caps_panel_count():
    lines = [Line("me", str(i)) for i in range(20)]
    assert len(group_panels(lines, max_panels=6)) <= 6


def test_panel_emotion_picks_strongest_non_neutral():
    lines = [Line("me", "a", "neutral", 0.99), Line("me", "b", "sadness", 0.5),
             Line("me", "c", "anger", 0.7)]
    assert panel_emotion(lines, "me") == "anger"
    assert panel_emotion(lines, "them") == "neutral"


def test_llm_script_reuses_character_looks_and_classifier_expressions():
    script = write_script(LINES, fake_llm(json.dumps(GOOD_REPLY)))
    assert script.source == "llm"
    p1, p2 = script.panels
    # Same character description in every panel the character appears in.
    assert "grey hoodie" in p1.prompt and "grey hoodie" in p2.prompt
    # Expression comes from the classifier's emotion, not the LLM.
    assert "shocked expression" in p1.prompt   # me: surprise (0.9) beats nothing else in panel 1
    assert "scared expression" in p1.prompt    # them: fear
    assert "papers flying" in p2.prompt


def test_code_fenced_json_is_accepted():
    script = write_script(LINES, fake_llm("```json\n" + json.dumps(GOOD_REPLY) + "\n```"))
    assert script.source == "llm"


def test_bad_llm_output_falls_back_to_template():
    assert write_script(LINES, fake_llm("sorry, I can't do that")).source == "template"
    wrong_count = dict(GOOD_REPLY, panels=["only one"])
    assert write_script(LINES, fake_llm(json.dumps(wrong_count))).source == "template"


def test_http_error_falls_back_to_template():
    script = write_script(LINES, fake_llm("", status=521))
    assert script.source == "template" and len(script.panels) == 2


def test_no_llm_uses_template():
    script = write_script(LINES)
    assert script.source == "template"
    assert all(p.prompt for p in script.panels)


def test_llm_error_message_includes_status():
    try:
        fake_llm("", status=401).chat([])
    except LLMError as e:
        assert "401" in str(e)
    else:
        raise AssertionError("expected LLMError")


def test_bubble_style_follows_emotion():
    assert Line("me", "!!", "surprise").bubble == "burst"
    assert Line("me", "ugh", "anger").bubble == "jagged"


def test_prompt_fits_clip_window_with_longest_llm_output():
    """CLIP reads 77 tokens; at ~1.3 tokens/word the prompt must stay under ~58 words."""
    long_reply = {"setting": " ".join(["word"] * 8),
                  "characters": {"me": " ".join(["m"] * 10), "them": " ".join(["t"] * 10)},
                  "panels": [" ".join(["scene"] * 14)] * 2}
    script = write_script(LINES, fake_llm(json.dumps(long_reply)))
    for p in script.panels:
        words = p.prompt.replace(",", " ").split()
        assert len(words) <= 64
        assert p.prompt.index("scene") < p.prompt.index("word")  # action before setting
