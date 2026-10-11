"""Turn a dialogue into comic panels, each with an image prompt.

One LLM call per comic invents the setting and a fixed look for each character, plus a
short scene description per panel. The final image prompt is then assembled in code:
character descriptions are repeated verbatim in every panel (keeps characters
consistent across images) and facial expressions come from our emotion classifier, not
the LLM. Without an LLM configured, a template fallback keeps the pipeline working.
"""
import json
import math
import re
from dataclasses import dataclass, field

from .llm import LLMClient, LLMError
from .style import ART_STYLE, BUBBLE, EXPRESSION

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
    scene: str
    prompt: str = ""


@dataclass
class Script:
    setting: str
    characters: dict[str, str]
    panels: list[Panel] = field(default_factory=list)
    source: str = "template"   # "llm" or "template"


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


def build_prompt(script: Script, panel: Panel) -> str:
    people = []
    for speaker in sorted({l.speaker for l in panel.lines}):
        look = script.characters.get(speaker, "a young person")
        people.append(f"{look}, {EXPRESSION[panel_emotion(panel.lines, speaker)]}")
    return ", ".join([ART_STYLE, script.setting, *people, panel.scene])


SYSTEM = """You write scripts for short comics made from text-message conversations.
Reply with JSON only, no prose, in exactly this shape:
{"setting": "<one place where the story happens, 5-12 words>",
 "characters": {"me": "<fixed visual description, 8-15 words>", "them": "<same>"},
 "panels": ["<what is visibly happening in panel 1, 8-20 words>", ...]}
Rules: one entry in "panels" per panel given, in order. Describe only what a camera
would see: poses, actions, props, framing. No dialogue, no text, no speech bubbles in
the descriptions. Messages may be in Hinglish or slang; infer the meaning."""


def _user_prompt(groups: list[list[Line]]) -> str:
    out = []
    for i, group in enumerate(groups, 1):
        out.append(f"Panel {i}:")
        out += [f'  {l.speaker} ({l.emotion}): "{l.text}"' for l in group]
    return "\n".join(out)


def _parse(raw: str, n_panels: int) -> tuple[str, dict, list[str]]:
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    data = json.loads(raw)
    scenes = [str(s) for s in data["panels"]]
    if len(scenes) != n_panels:
        raise ValueError(f"expected {n_panels} panels, got {len(scenes)}")
    chars = data["characters"]
    return str(data["setting"]), {"me": str(chars["me"]), "them": str(chars["them"])}, scenes


def _template(groups: list[list[Line]]) -> tuple[str, dict, list[str]]:
    scenes = []
    for group in groups:
        speakers = {l.speaker for l in group}
        who = "two friends texting each other" if len(speakers) > 1 else \
              f"{'the main character' if 'me' in speakers else 'their friend'} looking at a phone"
        scenes.append(f"{who}, medium shot")
    chars = {"me": "young person with short dark hair, blue hoodie",
             "them": "young person with curly hair, yellow t-shirt"}
    return "a cozy bedroom at night lit by a phone screen", chars, scenes


def write_script(lines: list[Line], llm: LLMClient | None = None) -> Script:
    groups = group_panels(lines)
    source = "template"
    setting, chars, scenes = _template(groups)
    if llm is not None and groups:
        try:
            raw = llm.chat([{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": _user_prompt(groups)}])
            setting, chars, scenes = _parse(raw, len(groups))
            source = "llm"
        except (LLMError, ValueError, KeyError, TypeError) as e:
            # Comic still renders with template prompts; surface why.
            print(f"[pinch] LLM scene writing failed, using template: {e}")

    script = Script(setting, chars, source=source)
    for group, scene in zip(groups, scenes):
        panel = Panel(group, scene)
        panel.prompt = build_prompt(script, panel)
        script.panels.append(panel)
    return script
