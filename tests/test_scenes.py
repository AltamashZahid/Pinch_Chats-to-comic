import json

import httpx

from pinch.llm import LLMClient, LLMError
from pinch.scenes import (Line, choose_focus, choose_shot, group_panels, panel_emotion,
                          write_script)

LINES = [
    Line("them", "the deadline is today", "fear", 0.8),
    Line("me", "WHAT", "surprise", 0.9),
    Line("me", "i'm so dead", "sadness", 0.6),
    Line("them", "lol good luck", "joy", 0.7),
]

GOOD_REPLY = {
    "characters": {"me": "tall guy with glasses and a grey hoodie",
                   "them": "girl with a ponytail and a red jacket"},
    "locations": {"me": "messy dorm room at night", "them": "crowded city bus"},
    "panels": [{"focus": "me", "action": "jumps up from his desk, papers flying"},
               {"focus": "them", "action": "laughing at her phone by the window"}],
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


def test_focus_is_the_most_emotional_speaker():
    assert choose_focus(LINES[:2], None) == "me"  # surprise 0.9 beats fear 0.8


def test_focus_alternates_when_nobody_feels_anything():
    calm = [Line("me", "ok"), Line("them", "sure")]
    assert choose_focus(calm, previous="them") == "me"
    assert choose_focus(calm, previous="me") == "them"


def test_shot_follows_emotion_and_never_repeats():
    assert choose_shot("surprise", 1, None) == "extreme close-up"
    assert choose_shot("neutral", 0, None) == "wide shot"
    assert choose_shot("surprise", 2, previous="extreme close-up") != "extreme close-up"


def test_texting_mode_shows_one_character_in_their_own_place():
    script = write_script(LINES, mode="texting")
    assert script.source == "template"
    p1, p2 = script.panels
    assert p1.focus == "me" and p2.focus == "them"
    assert script.location(p1) != script.location(p2)
    assert "extreme close-up" in p1.prompt and "jaw dropped" in p1.prompt  # surprise
    assert "single character" in p1.prompt
    assert script.characters["me"] in p1.prompt and script.characters["them"] not in p1.prompt


def test_consecutive_template_panels_differ():
    lines = [Line("me", "ok"), Line("them", "sure"), Line("me", "fine"), Line("them", "cool")]
    a, b = write_script(lines, mode="texting").panels
    assert a.prompt != b.prompt


def test_llm_script_uses_its_cast_locations_and_actions():
    script = write_script(LINES, fake_llm(json.dumps(GOOD_REPLY)), mode="texting")
    assert script.source == "llm"
    p1, p2 = script.panels
    assert "grey hoodie" in p1.prompt and "messy dorm room" in p1.prompt
    assert "papers flying" in p1.prompt
    assert "red jacket" in p2.prompt and "crowded city bus" in p2.prompt
    # Expression and shot still come from the classifier, not the LLM.
    assert "shocked face" in p1.prompt and "extreme close-up" in p1.prompt


def test_llm_focus_on_silent_character_is_overridden():
    reply = dict(GOOD_REPLY, panels=[{"focus": "nobody", "action": "x"},
                                     {"focus": "them", "action": "y"}])
    script = write_script(LINES, fake_llm(json.dumps(reply)))
    assert script.panels[0].focus == "me"


def test_code_fenced_json_is_accepted():
    script = write_script(LINES, fake_llm("```json\n" + json.dumps(GOOD_REPLY) + "\n```"))
    assert script.source == "llm"


def test_bad_llm_output_falls_back_to_template():
    assert write_script(LINES, fake_llm("sorry, I can't do that")).source == "template"
    wrong_count = dict(GOOD_REPLY, panels=GOOD_REPLY["panels"][:1])
    assert write_script(LINES, fake_llm(json.dumps(wrong_count))).source == "template"


def test_http_error_falls_back_to_template():
    script = write_script(LINES, fake_llm("", status=521))
    assert script.source == "template" and len(script.panels) == 2


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
    long_reply = {"characters": {"me": " ".join(["m"] * 12), "them": " ".join(["t"] * 12)},
                  "locations": {"me": " ".join(["place"] * 8), "them": " ".join(["place"] * 8)},
                  "panels": [{"focus": "me", "action": " ".join(["act"] * 10)},
                             {"focus": "them", "action": " ".join(["act"] * 10)}]}
    script = write_script(LINES, fake_llm(json.dumps(long_reply)))
    for p in script.panels:
        assert len(p.prompt.replace(",", " ").split()) <= 58
        assert p.prompt.index("act") < p.prompt.index("place")  # action before location


def test_wrong_panel_count_is_retried_once():
    replies = iter([json.dumps(dict(GOOD_REPLY, panels=GOOD_REPLY["panels"] * 2)), json.dumps(GOOD_REPLY)])
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": next(replies)}}]})

    llm = LLMClient("https://fake/v1", "key", "model", transport=httpx.MockTransport(handler))
    script = write_script(LINES, llm)
    assert script.source == "llm" and len(calls) == 2
    assert "invalid" in calls[1]["messages"][-1]["content"]


def test_story_mode_is_default_and_acts_it_out_without_phones(monkeypatch):
    monkeypatch.delenv("PINCH_SCENE_MODE", raising=False)
    script = write_script(LINES)
    assert script.mode == "story"
    p1, p2 = script.panels
    assert script.location(p1) == script.location(p2)          # together, one place
    assert "phone" not in p1.prompt and "phone" not in p2.prompt
    assert "single character" not in p1.prompt
    assert "hands on cheeks" in p1.prompt                       # surprise, acted out


def test_story_mode_uses_story_system_prompt():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(GOOD_REPLY)}}]})

    llm = LLMClient("https://fake/v1", "key", "model", transport=httpx.MockTransport(handler))
    write_script(LINES, llm, mode="story")
    assert "TOGETHER in one place" in seen[0]["messages"][0]["content"]


def test_scene_mode_from_env(monkeypatch):
    monkeypatch.setenv("PINCH_SCENE_MODE", "texting")
    assert write_script(LINES).mode == "texting"
