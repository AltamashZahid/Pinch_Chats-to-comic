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
    monkeypatch.setattr(render, "head_for", lambda name: "head.npz")  # pretend a judge was trained for it
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


def test_no_judge_for_this_backend_means_one_try(monkeypatch):
    """A judge trained on another model's images must not pick (it did worse than chance-level CLIP)."""
    b = CountingBackend()
    monkeypatch.setenv("PINCH_CANDIDATES", "4")
    monkeypatch.setattr(render, "judge", lambda name="local": (_ for _ in ()).throw(AssertionError("judge used")))
    _, tried, prob = render.draw_best(b, _panel(), 100, 512, 512)
    assert tried == 1 and prob is None and b.seeds == [100]


def test_head_for_has_no_cross_backend_fallback(monkeypatch):
    from pinch.judge import head_for
    monkeypatch.delenv("PINCH_JUDGE_HEAD", raising=False)
    assert head_for("anime") is None


class ReferenceBackend(CountingBackend):
    """Fake backend that takes a reference portrait, like anime + IP-Adapter."""
    name, model, ip_scale = "fakeip", "fakeip", 0.5

    def __init__(self):
        super().__init__()
        self.requests = []

    def generate(self, req):
        self.requests.append(req)
        return bytes([0x89]) + b"PNG" + req.prompt.encode() + str(req.seed).encode()


def test_each_character_gets_one_portrait_used_as_reference(tmp_path):
    from pinch.cache import ImageCache
    from pinch.scenes import Script
    from tests.test_cache import BagOfWords
    script = Script({"me": "boy with glasses", "them": "girl with red hair"}, {"me": "cafe", "them": "cafe"})
    script.panels = [Panel([Line("me", "hi", "joy", 0.9)], "me", prompt="me joy"),
                     Panel([Line("them", "yo", "anger", 0.9)], "them", prompt="them anger"),
                     Panel([Line("me", "ok", "sadness", 0.9)], "me", prompt="me sad")]
    b = ReferenceBackend()
    cache = ImageCache(tmp_path, embedder=BagOfWords())
    render.render_panels(script, b, cache)
    portraits = [r for r in b.requests if "portrait" in r.prompt]
    panels = [r for r in b.requests if "portrait" not in r.prompt]
    assert len(portraits) == 2                       # one per character, not per panel
    me_ref = [r.reference for r in panels if r.prompt.startswith("me")]
    assert me_ref[0] == me_ref[1] is not None        # both of "me"'s panels share the portrait
    assert panels[1].reference != me_ref[0]
    b.requests.clear()
    render.render_panels(script, b, cache)           # second comic: portraits and panels cached
    assert b.requests == []
