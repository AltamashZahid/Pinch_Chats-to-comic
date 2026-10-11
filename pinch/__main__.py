"""python -m pinch screenshot.png [--images]

Without --images: prints the comic script (dialogue, emotions, prompts) as JSON.
With --images: also generates one image per panel (cached) and prints where they are.
"""
import argparse
import json
from dataclasses import asdict

from .ocr import load_image
from .pipeline import screenshot_to_script


def main() -> None:
    parser = argparse.ArgumentParser(prog="pinch")
    parser.add_argument("screenshot")
    parser.add_argument("--images", action="store_true", help="generate panel images")
    args = parser.parse_args()
    script = screenshot_to_script(load_image(args.screenshot))
    if not args.images:
        print(json.dumps(asdict(script), indent=2, ensure_ascii=False))
        return

    from .cache import ImageCache
    from .imagegen import backend_from_env
    from .render import render_panels

    cache = ImageCache()
    for i, r in enumerate(render_panels(script, backend_from_env(), cache), 1):
        said = " / ".join(f"{l.speaker}: {l.text}" for l in r.panel.lines)
        print(f"panel {i} [{r.source:9} {r.seconds:5.1f}s] {r.image}\n    {said}")
    s = cache.stats
    print(f"cache: {s['exact']} exact, {s['semantic']} semantic, {s['miss']} miss, "
          f"{s['seconds_saved']:.1f}s of generation saved")


if __name__ == "__main__":
    main()
