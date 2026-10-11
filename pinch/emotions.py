"""Emotion labels shared by training and inference.

GoEmotions has 27 fine-grained emotions + neutral. A comic only needs a handful of
facial expressions, so we collapse them to Ekman's 6 basic emotions + neutral using
the mapping published with the dataset.
"""
LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

EKMAN = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism",
            "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"],
}

FINE_TO_COARSE = {fine: coarse for coarse, fines in EKMAN.items() for fine in fines}
