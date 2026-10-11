"""How each emotion looks on the page: face, body, camera and speech bubble."""

EXPRESSION = {
    "anger": "angry expression, furrowed brows, gritted teeth",
    "disgust": "disgusted expression, wrinkled nose, grimace",
    "fear": "scared expression, wide eyes, sweat drop",
    "joy": "happy expression, big smile, bright eyes",
    "sadness": "sad expression, teary eyes, downturned mouth",
    "surprise": "shocked expression, wide eyes, open mouth",
    "neutral": "calm relaxed expression",
}

# What the character does with the phone (texting is the one constant of a chat comic).
POSE = {
    "anger": "glaring at phone, gripping it tightly",
    "disgust": "holding phone at arm's length, recoiling",
    "fear": "clutching phone with both hands, trembling",
    "joy": "grinning at phone, laughing",
    "sadness": "slumped, staring at phone",
    "surprise": "staring at phone in shock, jaw dropped",
    "neutral": "casually texting on phone",
}

# Stronger emotions get tighter framing, the way comic artists cut in for reactions.
SHOT = {
    "anger": "close-up",
    "disgust": "close-up",
    "fear": "extreme close-up",
    "joy": "medium shot",
    "sadness": "medium shot",
    "surprise": "extreme close-up",
    "neutral": "wide shot",
}
SHOT_ALTERNATIVES = ["medium shot", "close-up", "over-the-shoulder shot", "wide shot"]

BUBBLE = {
    "anger": "jagged",     # spiky outline
    "disgust": "wavy",
    "fear": "wobbly",      # shaky outline
    "joy": "round",
    "sadness": "droopy",   # sagging bottom edge
    "surprise": "burst",   # starburst
    "neutral": "round",
}

ART_STYLE = "comic panel, clean ink lines, flat colors"

DEFAULT_CHARACTERS = {
    "me": "young man with short black hair and a blue hoodie",
    "them": "young woman with curly red hair and a yellow sweater",
}
DEFAULT_LOCATIONS = {
    "me": "cozy bedroom at night, warm lamp light",
    "them": "busy coffee shop, daytime",
}
