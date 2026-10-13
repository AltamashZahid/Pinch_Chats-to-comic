"""Turn a dialogue into comic panels, each with an image prompt.

Two modes (PINCH_SCENE_MODE):
- story (default): the chat is acted out in person. Both characters share one place and
  each panel dramatises what the messages are about (kind words -> a hug, bad news ->
  comforting), no phones unless the chat is about one.
- texting: literal. Each character is on their phone in their own location.

Each panel centres one character (the "focus": whoever feels the most in that moment).
Camera shot, facial expression and default pose come from the emotion classifier; an
optional LLM call invents the cast, the place(s) and a specific action per panel.

Character descriptions are repeated verbatim in every panel to keep characters
consistent. Without an LLM configured, templates keep the pipeline working.
"""
import json
import os
import re
from dataclasses import dataclass, field

from .llm import LLMClient, LLMError
from .style import (ART_STYLE, BUBBLE, DEFAULT_CHARACTERS, EXPRESSION, IRONY, POSE, SHOT,
                    SHOT_ALTERNATIVES, STORY_LOCATION, STORY_POSE, TEXTING_LOCATIONS)

MAX_PANELS = 8
MODES = ("story", "texting")


def scene_mode(mode: str | None = None) -> str:
    mode = (mode or os.environ.get("PINCH_SCENE_MODE") or "story").lower()
    if mode not in MODES:
        raise ValueError(f"PINCH_SCENE_MODE must be one of {MODES}, got {mode!r}")
    return mode


@dataclass
class Line:
    speaker: str      # "me" / "them"
    text: str
    emotion: str = "neutral"     # what the face shows (text tone merged with emoji tone)
    confidence: float = 1.0
    emoji: list[str] = field(default_factory=list)
    ironic: bool = False         # emoji and words disagree, e.g. "I failed physics 😂"
    situation: str = ""          # for ironic lines: the words' emotion (what's really going on)

    @property
    def bubble(self) -> str:
        return "wavy" if self.ironic else BUBBLE[self.emotion]

    @property
    def display(self) -> str:
        """Text plus its emoji, for logs and the LLM."""
        return " ".join(p for p in (self.text, "".join(self.emoji)) if p)


@dataclass
class Panel:
    lines: list[Line]
    focus: str = "me"      # which character the camera shows
    action: str = ""       # what they are doing
    shot: str = "medium shot"
    prompt: str = ""

    @property
    def emotion(self) -> str:
        return panel_emotion(self.lines, self.focus)

    @property
    def irony(self) -> str:
        """The focus character's ironic situation, if their strongest line is ironic."""
        felt = [l for l in self.lines if l.speaker == self.focus and l.emotion != "neutral"]
        top = max(felt, key=lambda l: l.confidence) if felt else None
        return IRONY.get(top.situation, "") if top is not None and top.ironic else ""


@dataclass
class Script:
    characters: dict[str, str]
    locations: dict[str, str]
    panels: list[Panel] = field(default_factory=list)
    source: str = "template"   # "llm" or "template"
    mode: str = "story"

    def location(self, panel: Panel) -> str:
        return self.locations.get(panel.focus, "")


def group_panels(lines: list[Line], max_panels: int = MAX_PANELS) -> list[list[Line]]:
    """One panel per turn, in chat order.

    A turn is consecutive messages from the same person ("Awww" + "Me 2" is one reply), so
    every panel shows exactly one speaker and only their bubbles: the face on screen is
    always the one talking. Long chats keep their most recent `max_panels` turns."""
    turns: list[list[Line]] = []
    for line in lines:
        if turns and turns[-1][0].speaker == line.speaker:
            turns[-1].append(line)
        else:
            turns.append([line])
    return turns[-max_panels:]


def panel_emotion(lines: list[Line], speaker: str) -> str:
    """The speaker's strongest non-neutral emotion in the panel, else neutral."""
    felt = [l for l in lines if l.speaker == speaker and l.emotion != "neutral"]
    return max(felt, key=lambda l: l.confidence).emotion if felt else "neutral"


def choose_shot(emotion: str, index: int, previous: str | None) -> str:
    shot = "wide shot" if index == 0 and emotion == "neutral" else SHOT[emotion]
    if shot == previous:  # never the same framing twice in a row
        shot = next(s for s in SHOT_ALTERNATIVES if s != previous)
    return shot


def build_prompt(script: Script, panel: Panel) -> str:
    """Most important first: SD's CLIP text encoder ignores everything past 77 tokens,
    so shot, character and expression come before the action and the background."""
    look = script.characters.get(panel.focus, "a young person")
    who = f"single character, {look}" if script.mode == "texting" else look
    parts = [ART_STYLE, panel.shot, who, EXPRESSION[panel.emotion]]
    if panel.irony:  # the joke goes before the action so it survives the 77-token cut
        parts.append(panel.irony)
    return ", ".join(parts + [panel.action, script.location(panel)])


SYSTEM_STORY = """You turn text-message conversations into short in-person comics.
Act the conversation out: the two people are TOGETHER in one place, talking face to face.
Each panel shows the person speaking in that turn, doing something that shows what the messages are about
(a kind message -> a hug or handing over a gift; bad news -> slumping while the other
comforts; a joke -> doubling over laughing). No phones unless the messages are about one.
Reply with JSON only, no prose, in exactly this shape:
{"characters": {"me": "<fixed visual look: age, gender, hair, clothes, 6-12 words>", "them": "<same>"},
 "locations": {"me": "<the shared place, 3-8 words>", "them": "<the SAME place>"},
 "panels": [{"focus": "me" or "them", "action": "<what they visibly do, 4-10 words>"}, ...]}
Rules: one entry in "panels" per panel given, in order. Each panel is ONE person's turn
in the conversation; its focus is the person speaking (given for each panel). The action must fit the emotion shown in brackets.
Describe only what a camera sees: no dialogue, no text, no speech bubbles. Messages may
be in Hinglish or slang; infer the meaning. Make the two characters look clearly different.
Lines marked IRONIC have an emoji that contradicts the words (e.g. "I failed 😂"): make
that panel funny, with the person reacting like the emoji while the situation matches the words."""

SYSTEM_TEXTING = """You write scripts for short comics made from text-message conversations.
The two people are texting, so they are in different places. Each panel shows the person
speaking in that turn.
Reply with JSON only, no prose, in exactly this shape:
{"characters": {"me": "<fixed visual look: age, gender, hair, clothes, 6-12 words>", "them": "<same>"},
 "locations": {"me": "<where 'me' is while texting, 3-8 words>", "them": "<same>"},
 "panels": [{"focus": "me" or "them", "action": "<what they visibly do, 4-10 words>"}, ...]}
Rules: one entry in "panels" per panel given, in order. Each panel is ONE person's turn
in the conversation; its focus is the person speaking (given for each panel). The action must fit the emotion shown in brackets
and the location, and should involve their phone or surroundings. Describe only what a
camera sees: no dialogue, no text, no speech bubbles. Messages may be in Hinglish or
slang; infer the meaning. Make the two characters look clearly different.
Lines marked IRONIC have an emoji that contradicts the words (e.g. "I failed 😂"): make
that panel funny, with the person reacting like the emoji while the situation matches the words."""


def _user_prompt(groups: list[list[Line]]) -> str:
    out = []
    for i, group in enumerate(groups, 1):
        out.append(f"Panel {i} ({group[0].speaker} speaks, focus must be {group[0].speaker}):")
        for l in group:
            tag = f"IRONIC: acts {l.emotion}, really {l.situation}" if l.ironic else l.emotion
            out.append(f'  {l.speaker} ({tag}): "{l.display}"')
    return "\n".join(out)


def _parse(raw: str, n_panels: int) -> tuple[dict, dict, list[tuple[str, str]]]:
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    data = json.loads(raw)
    panels = [(str(p["focus"]), str(p["action"])) for p in data["panels"]]
    if len(panels) != n_panels:
        raise ValueError(f"expected {n_panels} panels, got {len(panels)}")
    chars, locs = data["characters"], data["locations"]
    return ({"me": str(chars["me"]), "them": str(chars["them"])},
            {"me": str(locs["me"]), "them": str(locs["them"])}, panels)


def write_script(lines: list[Line], llm: LLMClient | None = None, mode: str | None = None) -> Script:
    mode = scene_mode(mode)
    groups = group_panels(lines)
    locations = dict(TEXTING_LOCATIONS) if mode == "texting" else         {"me": STORY_LOCATION, "them": STORY_LOCATION}
    script = Script(dict(DEFAULT_CHARACTERS), locations, mode=mode)
    poses = POSE if mode == "texting" else STORY_POSE
    system = SYSTEM_TEXTING if mode == "texting" else SYSTEM_STORY
    directions: list[tuple[str, str]] = []
    if llm is not None and groups:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": _user_prompt(groups) +
                     f'\n\nReturn exactly {len(groups)} entries in "panels".'}]
        for attempt in (1, 2):
            try:
                raw = llm.chat(messages)
                script.characters, script.locations, directions = _parse(raw, len(groups))
                script.source = "llm"
                break
            except LLMError as e:  # network/HTTP: retrying the same request rarely helps
                print(f"[pinch] LLM scene writing failed, using template: {e}")
                break
            except (ValueError, KeyError, TypeError) as e:
                if attempt == 2:
                    # Comic still renders with template prompts; surface why.
                    print(f"[pinch] LLM scene writing failed, using template: {e}")
                    break
                # Malformed reply: show the model its mistake once.
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": f"That reply was invalid ({e}). "
                              f"Reply again with JSON only and exactly {len(groups)} panels."}]

    prev_focus = prev_shot = None
    for i, group in enumerate(groups):
        focus = group[0].speaker  # each panel is one person's turn
        _, action = directions[i] if directions else (None, "")
        panel = Panel(group, focus)
        panel.action = action or poses[panel.emotion]
        panel.shot = choose_shot(panel.emotion, i, prev_shot)
        panel.prompt = build_prompt(script, panel)
        script.panels.append(panel)
        prev_focus, prev_shot = focus, panel.shot
    return script
