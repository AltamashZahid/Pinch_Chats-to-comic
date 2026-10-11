import pytest

from pinch.classifier import DEFAULT_MODEL_DIR

pytestmark = pytest.mark.skipif(not (DEFAULT_MODEL_DIR / "config.json").exists(),
                                reason="emotion model not trained")


@pytest.fixture(scope="module")
def clf():
    from pinch.classifier import EmotionClassifier
    return EmotionClassifier()


@pytest.mark.parametrize("text, emotion", [
    ("I'm so happy for you, congrats!!", "joy"),
    ("this is so annoying, stop doing that", "anger"),
    ("wait what?? how did that happen", "surprise"),
    ("I miss her so much, I can't stop crying", "sadness"),
    ("I'm terrified of the exam tomorrow", "fear"),
])
def test_obvious_emotions(clf, text, emotion):
    assert clf.predict([text])[0][0] == emotion


def test_returns_probability(clf):
    (label, p), = clf.predict(["ok"])
    assert 0.0 < p <= 1.0
