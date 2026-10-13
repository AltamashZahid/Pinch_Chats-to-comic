import os

import httpx
import numpy as np
import pytest

from pinch.cache import ImageCache, normalize
from pinch.imagegen import ImageGenError, ImageRequest, PlaceholderBackend, PollinationsBackend
from pinch.render import render_panels
from pinch.scenes import Line, write_script


class BagOfWords:
    """Deterministic stand-in for MiniLM: hashed word counts, L2-normalised."""
    def __call__(self, texts):
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.split():
                out[i, hash(w) % 256] += 1
        return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.fixture
def cache(tmp_path):
    return ImageCache(tmp_path, embedder=BagOfWords(), threshold=0.85)


def test_exact_hit_ignores_case_and_spacing(cache):
    key = cache.exact_key(seed=1)
    cache.put("Two friends on a couch", key, b"\x89PNGfake")
    hit = cache.get("two  friends on a COUCH", key)
    assert hit.kind == "exact" and hit.path.read_bytes() == b"\x89PNGfake"


def test_semantic_hit_on_reworded_scene(cache):
    key = cache.exact_key(seed=1)
    cache.put("two friends texting on a couch at night", key, b"img")
    hit = cache.get("two friends texting on a couch late at night", key)
    assert hit is not None and hit.kind == "semantic"


def test_different_scene_misses(cache):
    key = cache.exact_key(seed=1)
    cache.put("two friends texting on a couch", key, b"img")
    assert cache.get("a dragon flying over a castle", key) is None


def test_same_scene_different_exact_key_misses(cache):
    cache.put("girl on a bus", cache.exact_key(expressions={"them": "joy"}), b"img")
    assert cache.get("girl on a bus", cache.exact_key(expressions={"them": "sadness"})) is None


def test_stats_and_hit_rate(cache):
    key = cache.exact_key(seed=1)
    cache.put("a b c", key, b"img", gen_seconds=5.0)
    cache.get("a b c", key)
    cache.get("x y z", key)
    assert cache.stats["exact"] == 1 and cache.stats["miss"] == 1
    assert cache.stats["seconds_saved"] == 5.0 and cache.hit_rate() == 0.5


def test_persists_across_instances(tmp_path):
    key = ImageCache.exact_key(seed=1)
    ImageCache(tmp_path, embedder=BagOfWords()).put("scene", key, b"img")
    assert ImageCache(tmp_path, embedder=BagOfWords()).get("scene", key).kind == "exact"


def test_render_generates_once_then_hits_cache(cache):
    lines = [Line("them", "hi", "joy", .9), Line("me", "hey", "joy", .9)]
    script = write_script(lines)
    first = render_panels(script, PlaceholderBackend(), cache, 256, 256)
    second = render_panels(script, PlaceholderBackend(), cache, 256, 256)
    assert [r.source for r in first] == ["generated", "generated"]  # one panel per speaker's turn
    assert [r.source for r in second] == ["exact", "exact"]
    assert first[0].image == second[0].image


def test_pollinations_sends_prompt_and_seed():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, content=b"\xff\xd8jpeg", headers={"content-type": "image/jpeg"})

    data = PollinationsBackend(transport=httpx.MockTransport(handler)).generate(
        ImageRequest("a cat, comic", seed=42, width=512, height=512))
    assert data == b"\xff\xd8jpeg"
    assert "a%20cat%2C%20comic" in seen["url"] and "seed=42" in seen["url"]


def test_pollinations_error_is_raised():
    backend = PollinationsBackend(transport=httpx.MockTransport(
        lambda r: httpx.Response(502, text="bad gateway")))
    with pytest.raises(ImageGenError, match="502"):
        backend.generate(ImageRequest("x"))


# Calibration of the real threshold with the real MiniLM encoder.
SAME = [("a messy dorm room at night, two friends texting on a couch, medium shot",
         "a messy dorm room at night, two friends on a sofa texting each other, medium shot"),
        ("girl with ponytail typing on her phone on a crowded bus",
         "girl with a ponytail on a busy bus typing on her phone"),
        ("guy jumps up from his desk, papers flying everywhere",
         "guy leaps up from his desk sending papers flying"),
        ("boy laughing at his phone in bed, room lit by the screen",
         "boy in bed laughing at his phone, the room lit by the phone screen")]
DIFF = [("girl with ponytail typing on her phone on a crowded bus",
         "girl with ponytail typing on her phone in a quiet library"),
        ("guy jumps up from his desk, papers flying everywhere",
         "guy asleep at his desk, papers stacked neatly"),
        ("two friends texting on a couch, medium shot",
         "two friends arguing face to face in a kitchen, close-up"),
        ("boy laughing at his phone in bed", "boy crying into his pillow in bed")]


@pytest.mark.skipif(os.environ.get("PINCH_SKIP_SLOW") == "1", reason="needs MiniLM download")
def test_minilm_threshold_separates_paraphrases_from_new_scenes():
    from pinch.cache import DEFAULT_THRESHOLD, MiniLMEmbedder
    embed = MiniLMEmbedder()

    def sim(a, b):
        va, vb = embed([normalize(a), normalize(b)])
        return float(va @ vb)

    assert min(sim(a, b) for a, b in SAME) >= DEFAULT_THRESHOLD
    assert max(sim(a, b) for a, b in DIFF) < DEFAULT_THRESHOLD
