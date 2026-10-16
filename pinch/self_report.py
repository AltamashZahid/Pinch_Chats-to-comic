"""Explicit self-reported feelings: "I am sooo happpy", "I'm crying", "I feel scared".

The conversation model reads the messages before and after each line. That helps on average
(past-only context: MELD weighted-F1 0.653 -> 0.628), but a later "now I'm crying" turned
"I am sooo happy" into sadness (0.90). When the speaker states their own feeling outright,
that statement wins. Negated statements ("I'm not happy") are left to the model.
"""
import re

STRETCH = re.compile(r"(\w)\1{2,}")

FEELINGS = {
    "joy": ["happy", "glad", "excited", "thrilled", "overjoyed", "ecstatic", "delighted", "so proud", "proud",
            "relieved", "grateful", "on cloud nine", "laughing", "lmao", "dying of laughter"],
    "sadness": ["sad", "upset", "depressed", "heartbroken", "crying", "cryin", "devastated", "miserable", "lonely",
                "hurt", "gutted", "in tears"],
    "anger": ["angry", "mad", "furious", "pissed", "annoyed", "irritated", "fuming", "livid", "so done"],
    "fear": ["scared", "afraid", "terrified", "frightened", "nervous", "anxious", "worried", "panicking", "freaking out"],
    "disgust": ["disgusted", "grossed out", "sick of", "repulsed", "nauseous"],
    "surprise": ["shocked", "surprised", "stunned", "speechless", "amazed", "astonished"],
}
_WORD = {w: e for e, words in FEELINGS.items() for w in words}
_INTENSIFIER = r"(?:(?:so+|very|really|super|soo+|literally|actually|just|kinda|lowkey|honestly|totally|now)\s+)*"
_PATTERN = re.compile(
    r"\b(?:i\s*am|i'?m|im|i\s+feel|i'?m\s+feeling|feeling)\s+" + _INTENSIFIER +
    r"(?P<feeling>" + "|".join(sorted(map(re.escape, _WORD), key=len, reverse=True)) + r")\b",
    re.IGNORECASE)
_NEGATION = re.compile(r"\b(?:not|never|no\s+longer|n't)\s+" + _INTENSIFIER + r"$", re.IGNORECASE)


def unstretch(text: str) -> str:
    """"happpy" -> "happy", "sooo" -> "soo" (3+ repeats -> 2; "cool" stays)."""
    return STRETCH.sub(r"\1\1", text)


def self_report(text: str) -> str | None:
    """The emotion the speaker explicitly says they feel, or None. The last statement wins
    ("I was sad, now I'm happy" -> joy)."""
    t = unstretch(text)
    found = None
    for m in _PATTERN.finditer(t):
        before = t[:m.start("feeling")]
        if _NEGATION.search(before[before.lower().rfind(" i") if " i" in before.lower() else 0:]):
            continue
        found = _WORD[m.group("feeling").lower()]
    return found
