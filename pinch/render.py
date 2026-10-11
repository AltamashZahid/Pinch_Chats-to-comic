"""Generate (or fetch from cache) one image per comic panel."""
import json
import time
from dataclasses import dataclass
from pathlib import Path

from .cache import ImageCache, sha
from .imagegen import ImageRequest
from .scenes import Panel, Script
from .style import ART_STYLE


@dataclass
class RenderedPanel:
    panel: Panel
    image: Path
    source: str        # "generated", "exact", or "semantic"
    seconds: float


def cast_seed(script: Script) -> int:
    """Same cast -> same seed, which nudges the model toward consistent faces."""
    return int(sha(json.dumps(script.characters, sort_keys=True))[:8], 16) % 2**31


def panel_key(cache: ImageCache, backend, script: Script, panel: Panel,
              seed: int, width: int, height: int) -> str:
    """Everything that must match exactly for a cached image to be reused."""
    return cache.exact_key(
        backend=backend.name, model=backend.model, width=width, height=height, seed=seed,
        style=ART_STYLE, shot=panel.shot,
        character=script.characters.get(panel.focus, ""), expression=panel.emotion,
    )


def scene_text(script: Script, panel: Panel) -> str:
    """The free-text part compared by embedding similarity: what happens, and where."""
    return f"{panel.action}, {script.location(panel)}"


def render_panels(script: Script, backend, cache: ImageCache,
                  width: int = 512, height: int = 512) -> list[RenderedPanel]:
    seed = cast_seed(script)
    out = []
    for panel in script.panels:
        text = scene_text(script, panel)
        key = panel_key(cache, backend, script, panel, seed, width, height)
        start = time.perf_counter()
        hit = cache.get(text, key)
        if hit:
            out.append(RenderedPanel(panel, hit.path, hit.kind, time.perf_counter() - start))
            continue
        image = backend.generate(ImageRequest(panel.prompt, seed, width, height))
        took = time.perf_counter() - start
        path = cache.put(text, key, image, gen_seconds=took)
        out.append(RenderedPanel(panel, path, "generated", took))
    return out
