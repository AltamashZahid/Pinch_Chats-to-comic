"""Generate extra training dialogues for rare emotions (fear, disgust) with an LLM.

    python training/augment_rare.py --per-emotion 400

Why: in conversation datasets fear and disgust are ~2% of turns and every published model is
weak on them; minority-class augmentation is one of the few things reported to help (CBERL).
Each request asks for a short, realistic casual text chat (2-6 turns, two people) in which
exactly one marked turn clearly expresses the target emotion and the other turns are labelled
too. Scenarios are varied by topic seeds to avoid near-duplicates.

Output: data/text/augment/<emotion>.jsonl, one dialogue per line:
    {"turns": [["A", "text", "label"], ...], "target": "fear", "topic": "..."}
training/text_datasets.dialogues("augment", "train") reads them. Never used for testing.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pinch.pipeline import load_env  # noqa: E402

load_env()

import httpx  # noqa: E402

from pinch.emotions import LABELS  # noqa: E402

OUT = ROOT / "data" / "text" / "augment"
MODEL = os.environ.get("PINCH_AUGMENT_MODEL", "openai/gpt-oss-120b")
TOPICS = {
    "fear": ["walking home alone at night", "a strange noise in the house", "medical test results", "a turbulent flight",
             "being followed", "an exam they might fail", "a car almost hitting them", "a job interview",
             "a spider or snake", "a phone call from an unknown number", "a horror movie", "getting lost",
             "a fire alarm", "parents finding out a secret", "a dentist appointment", "an earthquake",
             "a dark basement", "a storm and power cut", "a dog chasing them", "speaking on stage",
             "being alone in a big empty house", "money running out", "a friend not answering for hours"],
    "disgust": ["spoiled food", "a dirty public toilet", "someone chewing loudly", "a cockroach in the kitchen",
                "a roommate's dirty dishes", "a hair in their food", "a smelly bus ride", "someone picking their nose",
                "rotten milk", "a gross video online", "a sweaty gym", "a moldy fridge", "vomit on the street",
                "a cheating partner's behavior", "a rude creepy message", "an overflowing trash can",
                "a dead rat", "slimy food texture", "someone not washing hands", "a hotel room with stains"],
}
STYLE_HINTS = ["very short texts, lowercase, slang", "normal casual texting", "a few typos, abbreviations like 'u' and 'rn'",
               "dramatic, lots of punctuation", "dry and understated", "mixes in some emoji words like 'lol' and 'omg'"]

PROMPT = """Write a short, realistic casual text-message chat between two friends, A and B.
Scenario: {topic}. Style: {style}.
Rules:
- 2 to 6 messages in total, alternating or not as people really text.
- Exactly one message must clearly express {emotion} ({desc}); mark it as the target.
- Label EVERY message with one of: anger, disgust, fear, joy, sadness, surprise, neutral.
- Do not name the emotion word itself in the target message (no "I'm scared", "disgusting"
  only half the time at most) - show it the way real people text.
Reply with JSON only:
{{"turns": [{{"speaker": "A", "text": "...", "label": "...", "target": false}}, ...]}}"""

DESC = {"fear": "scared, anxious, nervous, terrified", "disgust": "grossed out, revolted, repulsed, sickened"}


def generate(emotion: str, rng: random.Random) -> dict | None:
    topic, style = rng.choice(TOPICS[emotion]), rng.choice(STYLE_HINTS)
    body = {"model": MODEL, "temperature": 1.0, "max_tokens": 1200, "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": PROMPT.format(topic=topic, style=style, emotion=emotion,
                                                                   desc=DESC[emotion])}]}
    for attempt in range(5):
        try:
            r = httpx.post("https://api.groq.com/openai/v1/chat/completions", json=body, timeout=60,
                           headers={"Authorization": f"Bearer {os.environ['PINCH_LLM_API_KEY']}"})
            if r.status_code == 429:
                time.sleep(float(r.headers.get("retry-after", 20)))
                continue
            r.raise_for_status()
            data = json.loads(re.search(r"\{.*\}", r.json()["choices"][0]["message"]["content"], re.S).group(0))
            turns = [(str(t["speaker"]), str(t["text"]).strip(), str(t["label"]).lower().strip(), bool(t.get("target")))
                     for t in data["turns"]]
            if not 2 <= len(turns) <= 6 or any(l not in LABELS or not txt for _, txt, l, _ in turns):
                raise ValueError("bad shape or label")
            targets = [t for t in turns if t[3]]
            if len(targets) != 1 or targets[0][2] != emotion:
                raise ValueError("target turn missing or mislabelled")
            return {"turns": [[s, txt, l] for s, txt, l, _ in turns], "target": emotion, "topic": topic, "style": style}
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError):
            time.sleep(2)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-emotion", type=int, default=400)
    ap.add_argument("--emotions", nargs="*", default=["fear", "disgust"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    for emotion in args.emotions:
        path = OUT / f"{emotion}.jsonl"
        have = sum(1 for _ in open(path, encoding="utf-8")) if path.exists() else 0
        seen = set()
        with open(path, "a", encoding="utf-8") as f:
            while have < args.per_emotion:
                d = generate(emotion, rng)
                if d is None:
                    continue
                key = " ".join(t[1].lower() for t in d["turns"])
                if key in seen:
                    continue
                seen.add(key)
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                f.flush()
                have += 1
                if have % 25 == 0:
                    print(f"  {emotion}: {have}/{args.per_emotion}", flush=True)


if __name__ == "__main__":
    main()
