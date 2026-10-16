import json

import httpx

from pinch.llm import LLMClient, LLMError
from pinch.scenes import Line, choose_shot, group_panels, panel_emotion, write_script

LINES = [  # two turns: "me" (two messages), then "them"
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


def test_one_panel_per_turn_in_chat_order():
    lines = [Line("them", "Love uuu"), Line("me", "Awww"), Line("me", "Me 2"),
             Line("them", "Missing u"), Line("me", "Same")]
    groups = group_panels(lines)
    assert [[l.text for l in g] for g in groups] == [["Love uuu"], ["Awww", "Me 2"], ["Missing u"], ["Same"]]
    assert all(len({l.speaker for l in g}) == 1 for g in groups)


def test_long_chats_keep_most_recent_turns():
    lines = [Line("me" if i % 2 else "them", str(i)) for i in range(20)]
    groups = group_panels(lines, max_panels=6)
    assert len(groups) == 6 and groups[-1][0].text == "19"


def test_panel_emotion_picks_strongest_non_neutral():
    lines = [Line("me", "a", "neutral", 0.99), Line("me", "b", "sadness", 0.5),
             Line("me", "c", "anger", 0.7)]
    assert panel_emotion(lines, "me") == "anger"
    assert panel_emotion(lines, "them") == "neutral"


def test_every_panel_shows_whoever_is_speaking():
    script = write_script(LINES)
    for p in script.panels:
        assert {l.speaker for l in p.lines} == {p.focus}


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
    panels = write_script(lines, mode="texting").panels
    assert len(panels) == 4
    assert all(a.prompt != b.prompt for a, b in zip(panels, panels[1:]))


def test_llm_script_uses_its_cast_locations_and_actions():
    script = write_script(LINES, fake_llm(json.dumps(GOOD_REPLY)), mode="texting")
    assert script.source == "llm"
    p1, p2 = script.panels
    assert "grey hoodie" in p1.prompt and "messy dorm room" in p1.prompt
    assert "papers flying" in p1.prompt
    assert "red jacket" in p2.prompt and "crowded city bus" in p2.prompt
    # Expression and shot still come from the classifier, not the LLM.
    assert "shocked face" in p1.prompt and "extreme close-up" in p1.prompt


def test_focus_is_always_the_speaker_whatever_the_llm_says():
    reply = dict(GOOD_REPLY, panels=[{"focus": "them", "action": "x"},
                                     {"focus": "nobody", "action": "y"}])
    script = write_script(LINES, fake_llm(json.dumps(reply)))
    assert [p.focus for p in script.panels] == ["me", "them"]


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


def test_names_are_passed_to_the_llm():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(GOOD_REPLY)}}]})

    llm = LLMClient("https://fake/v1", "key", "model", transport=httpx.MockTransport(handler))
    write_script(LINES, llm, names={"me": "Altamash", "them": "Rahul"})
    assert '"Rahul"' in seen[0]["messages"][1]["content"]


def test_all_caps_is_shouting():
    from pinch.scenes import is_shouting
    assert is_shouting("WHAT") and is_shouting("I HATE YOU!!") and is_shouting("OMG 2 HOURS LATE")
    assert not is_shouting("OK") and not is_shouting("lol WHAT") and not is_shouting("Hello there")


def test_shouting_changes_bubble_prompt_and_shot():
    lines = [Line("me", "WHY WOULD YOU DO THAT", "anger", 0.9), Line("them", "sorry", "sadness", 0.6)]
    p = write_script(lines).panels[0]
    assert p.lines[0].bubble == "jagged"
    assert "shouting, mouth wide open" in p.prompt
    assert "close" in p.shot
    assert p.lines[0].text == "WHY WOULD YOU DO THAT"  # original text kept for the bubble


def test_classifier_sees_lowercase_for_shouting(monkeypatch):
    from pinch import pipeline
    from pinch.models import Box, Message
    seen = []
    monkeypatch.setattr(pipeline, "classify_dialogue", lambda turns: seen.extend(t for _, t in turns) or [("anger", .9)] * len(turns))
    lines = pipeline.lines_from_messages([Message("me", "I HATE THIS", Box(0, 0, 1, 1))])
    assert seen == ["i hate this"] and lines[0].text == "I HATE THIS"


def test_explicit_feeling_beats_later_replies(monkeypatch):
    """"I am sooo happpy" stays joy even though later messages talk about crying, but only when
    the model reading the past alone agrees with the statement."""
    from pinch import pipeline
    turns = [("B", "OMGG I am sooo happpy right now"), ("A", "why?? I thought we were supposed to cry"),
             ("B", "Okayy. Now I am crying"), ("A", "I'm glad you asked")]
    def fake(turns, future=True):
        if future:  # reading the replies too: the first line drifts to sadness
            return [("sadness", 0.9), ("surprise", 0.8), ("sadness", 0.95), ("neutral", 0.7)]
        return [("joy", 0.54), ("surprise", 0.8), ("sadness", 0.95), ("neutral", 0.6)]
    monkeypatch.setattr(pipeline, "classify_dialogue", fake)
    out = pipeline.trust_self_reports(turns, fake(turns))
    assert out[0] == ("joy", 0.54)            # statement + past-only model agree: joy wins
    assert out[2] == ("sadness", 0.95)        # already agreed
    assert out[3] == ("neutral", 0.7)         # "I'm glad you asked": the model disagrees, so no override
