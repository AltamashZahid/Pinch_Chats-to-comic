"""python -m pinch screenshot.png  ->  prints the extracted dialogue as JSON."""
import argparse
import json

from .ocr import extract_dialogue, load_image


def main() -> None:
    parser = argparse.ArgumentParser(prog="pinch")
    parser.add_argument("screenshot")
    args = parser.parse_args()
    messages = extract_dialogue(load_image(args.screenshot))
    print(json.dumps([m.to_dict() for m in messages], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
