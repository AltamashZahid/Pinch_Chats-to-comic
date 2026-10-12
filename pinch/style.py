"""How each emotion looks on the page: face, body, camera and speech bubble."""

# Comic-artist visual shorthand, not emotion words: SD-Turbo drew "disgusted expression"
# correctly 0/10 times but "grossed out face, tongue sticking out..." about 5/10 (fear 2 -> ~7,
# anger 2 -> ~5, blind visual check of 10 images each).
EXPRESSION = {
    "anger": "furious face, eyebrows slanted down hard, teeth bared, vein popping on forehead",
    "disgust": "grossed out face, tongue sticking out, nose scrunched, eyes squinted",
    "fear": "terrified face, wide eyes with tiny pupils, screaming, sweat drops",
    "joy": "happy face, big smile, bright eyes",
    "sadness": "crying face, tears streaming down cheeks, trembling lip",
    "surprise": "shocked face, eyes popping wide, jaw dropped",
    "neutral": "calm relaxed face",
}

# Texting mode: what the character does with the phone.
POSE = {
    "anger": "glaring at phone, gripping it tightly",
    "disgust": "holding phone at arm's length, recoiling",
    "fear": "clutching phone with both hands, trembling",
    "joy": "grinning at phone, laughing",
    "sadness": "slumped, staring at phone",
    "surprise": "staring at phone in shock, jaw dropped",
    "neutral": "casually texting on phone",
}

# Story mode: the conversation acted out in person, no phones.
STORY_POSE = {
    "anger": "pointing a finger, shouting",
    "disgust": "recoiling, covering nose with hand",
    "fear": "cowering, hands raised defensively",
    "joy": "laughing, arms thrown open",
    "sadness": "head down, wiping away tears",
    "surprise": "jaw dropped, hands on cheeks",
    "neutral": "talking, gesturing casually",
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
TEXTING_LOCATIONS = {
    "me": "cozy bedroom at night, warm lamp light",
    "them": "busy coffee shop, daytime",
}
STORY_LOCATION = "cozy living room, warm evening light"
