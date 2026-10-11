"""How each emotion looks on the page: the character's face and the speech bubble shape."""

EXPRESSION = {
    "anger": "angry expression, furrowed brows, gritted teeth",
    "disgust": "disgusted expression, wrinkled nose, grimace",
    "fear": "scared expression, wide eyes, sweat drop",
    "joy": "happy expression, big smile, bright eyes",
    "sadness": "sad expression, teary eyes, downturned mouth",
    "surprise": "shocked expression, wide eyes, open mouth",
    "neutral": "calm relaxed expression",
}

BUBBLE = {
    "anger": "jagged",     # spiky outline
    "disgust": "wavy",
    "fear": "wobbly",      # shaky outline
    "joy": "round",
    "sadness": "droopy",   # sagging bottom edge
    "surprise": "burst",   # starburst
    "neutral": "round",
}

ART_STYLE = "comic book panel, clean ink lines, flat colors, expressive cartoon characters"
