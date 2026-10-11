"""python -m pinch screenshot.png  ->  prints the comic script (dialogue, emotions, prompts)."""
import argparse
import json
from dataclasses import asdict

from .ocr import load_image
from .pipeline import screenshot_to_script


def main() -> None:
    parser = argparse.ArgumentParser(prog="pinch")
    parser.add_argument("screenshot")
    args = parser.parse_args()
    script = screenshot_to_script(load_image(args.screenshot))
    print(json.dumps(asdict(script), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
