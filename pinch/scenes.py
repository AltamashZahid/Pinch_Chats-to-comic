"""Turn a dialogue into comic panels, each with an image prompt.

People texting each other are in different places, so each panel shows one character
(the "focus": whoever feels the most in that moment) in their own location. The camera
shot, facial expression and default pose come from the emotion classifier; an optional
LLM call invents the cast, their locations and a specific action per panel.

Character descriptions are repeated verbatim in every panel to keep characters
consistent. Without an LLM configured, templates keep the pipeline working.
"""
import json
import math
import re
from dataclasses import dataclass, field

from .llm import LLMClient, LLMError
from .style import (ART_STYLE, BUBBLE, DEFAULT_CHARACTERS, DEFAULT_LOCATIONS, EXPRESSION,
                    POSE, SHOT, SHOT_ALTERNATIVES)

MAX_PANELS = 6


@dataclass
class Line:
    speaker: str      # "me" / "them"
    text: str
    emotion: str = "neutral"
    confidence: float = 1.0

    @property
    def bubble(self) -> str:
        return BUBBLE[self.emotion]


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


@dataclass
class Script:
    characters: dict[str, str]
    locations: dict[str, str]
    panels: list[Panel] = field(default_factory=list)
    source: str = "template"   # "llm" or "template"

    def location(self, panel: Panel) -> str:
        return self.locations.get(panel.focus, "")


def group_panels(lines: list[Line], max_panels: int = MAX_PANELS) -> list[list[Line]]:
    """Consecutive lines, at most 2 per panel unless the chat is long."""
    if not lines:
        return []
    per_panel = max(2, math.ceil(len(lines) / max_panels))
    return [lines[i:i + per_panel] for i in range(0, len(lines), per_panel)]


def panel_emotion(lines: list[Line], speaker: str) -> str:
    """The speaker's strongest non-neutral emotion in the panel, else neutral."""
    felt = [l for l in lines if l.speaker == speaker and l.emotion != "neutral"]
    return max(felt, key=lambda l: l.confidence).emotion if felt else "neutral"


def choose_focus(lines: list[Line], previous: str | None) -> str:
    """Show whoever feels the most; if nobody does, alternate so both characters appear."""
    felt = [l for l in lines if l.emotion != "neutral"]
    if felt:
        return max(felt, key=lambda l: l.confidence).speaker
    speakers = [l.speaker for l in lines]
    others = [s for s in speakers if s != previous]
    return others[-1] if others else speakers[-1]


def choose_shot(emotion: str, index: int, previous: str | None) -> str:
    shot = "wide shot" if index == 0 and emotion == "neutral" else SHOT[emotion]
    if shot == previous:  # never the same framing twice in a row
        shot = next(s for s in SHOT_ALTERNATIVES if s != previous)
    return shot


def build_prompt(script: Script, panel: Panel) -> str:
    """Most important first: SD's CLIP text encoder ignores everything past 77 tokens,
    so shot, character and expression come before the action and the background."""
    look = script.characters.get(panel.focus, "a young person")
    return ", ".join([ART_STYLE, panel.shot, f"single character, {look}",
                      EXPRESSION[panel.emotion], panel.action, script.location(panel)])


SYSTEM = """You write scripts for short comics made from text-message conversations.
The two people are texting, so they are in different places. Each panel shows ONE of them.
Reply with JSON only, no prose, in exactly this shape:
{"characters": {"me": "<fixed visual look: age, gender, hair, clothes, 6-12 words>", "them": "<same>"},
 "locations": {"me": "<where 'me' is while texting, 3-8 words>", "them": "<same>"},
 "panels": [{"focus": "me" or "them", "action": "<what they visibly do, 4-10 words>"}, ...]}
Rules: one entry in "panels" per panel given, in order. Pick as focus the person whose
reaction matters most in that panel. The action must fit the emotion shown in brackets
and the location, and should involve their phone or surroundings. Describe only what a
camera sees: no dialogue, no text, no speech bubbles. Messages may be in Hinglish or
slang; infer the meaning. Make the two characters look clearly different."""


def _user_prompt(groups: list[list[Line]]) -> str:
    out = []
    for i, group in enumerate(groups, 1):
        out.append(f"Panel {i}:")
        out += [f'  {l.speaker} ({l.emotion}): "{l.text}"' for l in group]
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


def write_script(lines: list[Line], llm: LLMClient | None = None) -> Script:
    groups = group_panels(lines)
    script = Script(dict(DEFAULT_CHARACTERS), dict(DEFAULT_LOCATIONS))
    directions: list[tuple[str, str]] = []
    if llm is not None and groups:
        messages = [{"role": "system", "content": SYSTEM},
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
        speakers = {l.speaker for l in group}
        focus, action = directions[i] if directions else (None, "")
        if focus not in speakers:  # LLM picked someone silent in this panel, or no LLM
            focus = choose_focus(group, prev_focus)
        panel = Panel(group, focus)
        panel.action = action or POSE[panel.emotion]
        panel.shot = choose_shot(panel.emotion, i, prev_shot)
        panel.prompt = build_prompt(script, panel)
        script.panels.append(panel)
        prev_focus, prev_shot = focus, panel.shot
    return script
