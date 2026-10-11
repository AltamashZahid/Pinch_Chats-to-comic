"""Accuracy on real screenshots in samples/ (kept out of git for privacy).

samples/expected.json maps file name -> [[speaker, text], ...]. Speakers must match
exactly; text is scored with character-level similarity so one OCR slip doesn't fail it.
"""
import json
from difflib import SequenceMatcher
from pathlib import Path

import pytest

from pinch.ocr import extract_dialogue, load_image, tesseract_available

SAMPLES = Path(__file__).parent.parent / "samples"
EXPECTED = SAMPLES / "expected.json"
cases = json.loads(EXPECTED.read_text(encoding="utf-8")) if EXPECTED.exists() else {}


@pytest.mark.skipif(not tesseract_available(), reason="tesseract not installed")
@pytest.mark.parametrize("name", sorted(cases))
def test_real_screenshot(name):
    got = extract_dialogue(load_image(str(SAMPLES / name)))
    want = cases[name]
    assert [m.speaker for m in got] == [s for s, _ in want]
    for m, (_, text) in zip(got, want):
        assert SequenceMatcher(None, m.text.lower(), text.lower()).ratio() >= 0.9, (m.text, text)
