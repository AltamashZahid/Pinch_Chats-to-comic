from pinch import render
from pinch.scenes import Line, Panel


class CountingBackend:
    name, model = "fake", "fake"

    def __init__(self):
        self.seeds = []

    def generate(self, req):
        self.seeds.append(req.seed)
        return str(req.seed).encode()


class FakeJudge:
    def __init__(self, scores):
        self.scores = scores  # image bytes -> probability of the target emotion

    def probs(self, images):
        return [{"sadness": self.scores.get(b, 0.0)} for b in images]


def _panel():
    return Panel([Line("me", "missing u", "sadness", 0.9)], "me", prompt="p")


def test_keeps_the_candidate_the_judge_likes_best(monkeypatch):
    b = CountingBackend()
    monkeypatch.setenv("PINCH_CANDIDATES", "4")
    monkeypatch.setattr(render, "judge", lambda name="local": FakeJudge({b"100": 0.1, str(100 + 7919).encode(): 0.4,
                                                          str(100 + 2 * 7919).encode(): 0.3}))
    img, tried, prob = render.draw_best(b, _panel(), 100, 512, 512)
    assert tried == 4 and len(set(b.seeds)) == 4
    assert img == str(100 + 7919).encode() and prob == 0.4


def test_single_candidate_skips_judge(monkeypatch):
    b = CountingBackend()
    monkeypatch.setenv("PINCH_CANDIDATES", "1")
    monkeypatch.setattr(render, "judge", lambda name="local": (_ for _ in ()).throw(AssertionError("judge used")))
    _, tried, prob = render.draw_best(b, _panel(), 100, 512, 512)
    assert tried == 1 and prob is None
