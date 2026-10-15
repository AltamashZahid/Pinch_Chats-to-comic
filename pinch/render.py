"""Generate (or fetch from cache) one image per comic panel.

Best-of-N: for backends with a trained expression judge, each panel is drawn several times
with different seeds and the judge keeps the one that most clearly shows the panel's emotion.
Simulated on 700 blind-labelled panels it went 55% -> 74% (best of 4); end to end, blind, the
SD-Turbo pipeline reached 68%. Backends without their own judge draw one image.

Same face across panels: backends that take a reference image (anime + IP-Adapter) get each
character's portrait, drawn once per look and cached.
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .cache import ImageCache, sha
from .imagegen import ImageRequest
from .judge import head_for, judge
from .scenes import REVEAL_ACTION, Panel, Script
from .style import ART_STYLE, EXPRESSION


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class RenderedPanel:
    panel: Panel
    image: Path
    source: str        # "generated", "exact", or "semantic"
    seconds: float
    candidates: int = 1          # images drawn for this panel
    judge_prob: float | None = None  # judge's confidence that the kept image shows the emotion
    reveal_image: Path | None = None  # second beat of a two-beat panel


def cast_seed(script: Script) -> int:
    """Same cast -> same seed, which nudges the model toward consistent faces."""
    return int(sha(json.dumps(script.characters, sort_keys=True))[:8], 16) % 2**31


def _beat(panel: Panel, reveal: bool) -> tuple[str, str, str]:
    """(prompt, emotion, shot) of a panel's first beat, or of its reveal beat."""
    if reveal:
        return panel.reveal_prompt, panel.reveal, "close-up"
    return panel.prompt, panel.emotion, panel.shot


def panel_key(cache: ImageCache, backend, script: Script, panel: Panel,
              seed: int, width: int, height: int, reveal: bool = False, reference: bytes | None = None) -> str:
    """Everything that must match exactly for a cached image to be reused."""
    _, emotion, shot = _beat(panel, reveal)
    return cache.exact_key(
        backend=backend.name, model=backend.model, width=width, height=height, seed=seed,
        style=ART_STYLE, shot=shot, candidates=_tries(backend, emotion),
        character=script.characters.get(panel.focus, ""), expression=emotion,
        reference=sha_bytes(reference) if reference else "",
    )


def portrait_prompt(look: str) -> str:
    """A neutral, front-facing portrait on a plain background: the character's reference image."""
    return f"{ART_STYLE}, close-up, portrait, {look}, {EXPRESSION['neutral']}, looking at viewer, plain white background"


def uses_reference(backend) -> bool:
    return bool(getattr(backend, "ip_scale", 0))


def portrait(backend, cache: ImageCache, look: str, seed: int) -> bytes:
    """The character's reference portrait: drawn once per look, then reused from the cache.
    Memory keeps a person's look fixed across comics, so their portrait is fixed too."""
    key = cache.exact_key(kind="portrait", backend=backend.name, model=backend.model, style=ART_STYLE,
                          character=look, seed=seed)
    hit = cache.get("portrait", key)
    if hit:
        return hit.path.read_bytes()
    start = time.perf_counter()
    image = backend.generate(ImageRequest(portrait_prompt(look), seed, emotion="neutral", shot="close-up"))
    cache.put("portrait", key, image, gen_seconds=time.perf_counter() - start)
    return image


def scene_text(script: Script, panel: Panel, reveal: bool = False) -> str:
    """The free-text part compared by embedding similarity: what happens, and where."""
    action = REVEAL_ACTION[panel.reveal] if reveal else panel.action
    return f"{action}, {script.location(panel)}"


EASY = {"joy", "neutral"}  # already ~98% right within 2 tries


def _candidates(emotion: str | None = None) -> int:
    """Tries per panel: PINCH_CANDIDATES (default 6) for hard emotions, PINCH_CANDIDATES_EASY
    (default 2) for joy/neutral. Measured on 420 blind-labelled SD1.5+LoRA panels: 53% right
    with 1 try, 76% best of 4, 81% best of 6."""
    n = int(os.environ.get("PINCH_CANDIDATES", "6"))
    if emotion in EASY:
        n = min(n, int(os.environ.get("PINCH_CANDIDATES_EASY", "2")))
    return max(1, n)


def _tries(backend, emotion: str | None) -> int:
    """Candidates actually drawn: best-of-N needs a judge trained on this backend's images."""
    return _candidates(emotion) if head_for(getattr(backend, "name", "local")) else 1


def draw_best(backend, panel: Panel, seed: int, width: int, height: int, reveal: bool = False,
              reference: bytes | None = None, character: str | None = None) -> tuple[bytes, int, float | None]:
    """Draw up to N candidates and keep the one the judge rates best for the panel's emotion."""
    prompt, emotion, shot = _beat(panel, reveal)
    n = _tries(backend, emotion)
    req = lambda s: ImageRequest(prompt, s, width, height, emotion=emotion, shot=shot,
                                 reference=reference, character=character)  # noqa: E731
    first = backend.generate(req(seed))
    j = judge(getattr(backend, "name", "local")) if n > 1 else None  # don't load CLIP when there's nothing to choose between
    if j is None:
        return first, 1, None
    images = [first] + [backend.generate(req(seed + 7919 * k)) for k in range(1, n)]
    scores = [p.get(emotion, 0.0) for p in j.probs(images)]
    i = max(range(len(images)), key=scores.__getitem__)
    return images[i], len(images), scores[i]


def render_panels(script: Script, backend, cache: ImageCache,
                  width: int = 512, height: int = 512) -> list[RenderedPanel]:
    seed = cast_seed(script)
    # Same face in every panel: each character's portrait steers their panels (IP-Adapter).
    refs = {role: portrait(backend, cache, script.characters[role], seed)
            for role in dict.fromkeys(p.focus for p in script.panels)} if uses_reference(backend) else {}

    def one(panel: Panel, reveal: bool) -> RenderedPanel:
        text = scene_text(script, panel, reveal)
        ref = refs.get(panel.focus)
        key = panel_key(cache, backend, script, panel, seed, width, height, reveal, ref)
        start = time.perf_counter()
        hit = cache.get(text, key)
        if hit:
            return RenderedPanel(panel, hit.path, hit.kind, time.perf_counter() - start)
        image, tried, prob = draw_best(backend, panel, seed, width, height, reveal, ref,
                                         script.characters.get(panel.focus))
        took = time.perf_counter() - start
        return RenderedPanel(panel, cache.put(text, key, image, gen_seconds=took), "generated", took, tried, prob)

    out = []
    for panel in script.panels:
        rp = one(panel, False)
        if panel.reveal:  # two-beat panel: second image for the cracked smile
            rp.reveal_image = one(panel, True).image
        out.append(rp)
    return out
