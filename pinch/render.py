"""Generate (or fetch from cache) one image per comic panel."""
import json
import time
from dataclasses import dataclass
from pathlib import Path

from .cache import ImageCache, sha
from .imagegen import ImageRequest
from .scenes import Panel, Script, panel_emotion
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
    speakers = sorted({l.speaker for l in panel.lines})
    return cache.exact_key(
        backend=backend.name, model=backend.model, width=width, height=height, seed=seed,
        style=ART_STYLE,
        characters={s: script.characters.get(s, "") for s in speakers},
        expressions={s: panel_emotion(panel.lines, s) for s in speakers},
    )


def render_panels(script: Script, backend, cache: ImageCache,
                  width: int = 512, height: int = 512) -> list[RenderedPanel]:
    seed = cast_seed(script)
    out = []
    for panel in script.panels:
        scene_text = f"{script.setting}, {panel.scene}"
        key = panel_key(cache, backend, script, panel, seed, width, height)
        start = time.perf_counter()
        hit = cache.get(scene_text, key)
        if hit:
            out.append(RenderedPanel(panel, hit.path, hit.kind, time.perf_counter() - start))
            continue
        image = backend.generate(ImageRequest(panel.prompt, seed, width, height))
        took = time.perf_counter() - start
        path = cache.put(scene_text, key, image, gen_seconds=took)
        out.append(RenderedPanel(panel, path, "generated", took))
    return out
