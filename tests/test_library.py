import io

import pytest
from PIL import Image

from pinch import library as L
from pinch.imagegen import ImageRequest
from tests.test_cache import BagOfWords


@pytest.fixture
def snapper():
    return L.Snapper(embedder=BagOfWords())


def test_gender_and_setting_helpers():
    assert L.gender("young woman with red hair") == "f"
    assert L.gender("mid twenties male, hoodie") == "m"
    assert L.gender("a person") == ""
    assert L.setting_of("standing on a busy city street") == "outdoor"
    assert L.setting_of("cozy living room") == "indoor"


def test_snap_keeps_gender_and_nearest_look(snapper):
    cast = snapper.snap_cast({"me": "young man with spiky blonde hair and a black leather jacket",
                              "them": "young woman with long wavy red hair and a cream sweater"})
    assert cast["me"] == L.PRESETS["m3"]
    assert cast["them"] == L.PRESETS["f3"]


def test_two_people_never_share_a_preset(snapper):
    cast = snapper.snap_cast({"me": "young man with short black hair and a blue hoodie",
                              "them": "young man with short black hair and a blue hoodie"})
    assert cast["me"] != cast["them"]
    assert all(L.gender(look) == "m" for look in cast.values())


def test_preset_looks_snap_to_themselves(snapper):
    assert snapper.snap_cast({"me": L.PRESETS["f6"]}) == {"me": L.PRESETS["f6"]}


class FakeLive:
    name, model = "live", "live"

    def __init__(self):
        self.calls = 0

    def generate(self, req):
        self.calls += 1
        return b"live"


def _png(color) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "LIBRARY_DIR", tmp_path)
    p = L.image_path("m1", "anger", "close", "outdoor", 0)
    p.parent.mkdir(parents=True)
    p.write_bytes(_png("red"))
    p.with_name("anger_close_outdoor_v0.0.rejected.png").write_bytes(_png("black"))
    return L.LibraryBackend(fallback=FakeLive())


def test_library_hit_serves_predrawn_panel(lib):
    req = ImageRequest("... angry, city street", 5, 128, 128, emotion="anger", shot="extreme close-up",
                       character=L.PRESETS["m1"])
    img = Image.open(io.BytesIO(lib.generate(req)))
    assert img.size == (128, 128) and img.getpixel((10, 10)) == (255, 0, 0)  # never the rejected one
    assert lib.hits == 1 and lib.fallback.calls == 0


def test_same_emotion_twice_in_a_comic_is_not_the_same_image(lib):
    p = L.image_path("m1", "anger", "medium", "outdoor", 0)
    p.write_bytes(_png("blue"))
    req = ImageRequest("street", 5, 64, 64, emotion="anger", shot="close-up", character=L.PRESETS["m1"])
    first, second = (Image.open(io.BytesIO(lib.generate(req))).getpixel((1, 1)) for _ in range(2))
    assert {first, second} == {(255, 0, 0), (0, 0, 255)}   # second falls back to the other framing
    assert lib.fallback.calls == 0


def test_library_miss_falls_back_to_live(lib):
    for req in [ImageRequest("street", 1, emotion="joy", shot="close-up", character=L.PRESETS["m1"]),   # not drawn
                ImageRequest("street", 1, emotion="anger", shot="close-up", character="someone else"),  # not a preset
                ImageRequest("street", 1, emotion="anger", shot="close-up", character=L.PRESETS["m1"].upper())]:
        assert lib.generate(req) == b"live"
    assert lib.misses == 3 and lib.fallback.calls == 3


def test_write_script_snaps_cast_before_prompts(snapper):
    from pinch.scenes import Line, write_script
    script = write_script([Line("me", "hi", "joy", 0.9), Line("them", "yo", "anger", 0.9)], mode="story",
                          looks={"them": "young woman with short purple hair and a leather jacket"},
                          snap=snapper.snap_cast)
    assert set(script.characters.values()) <= set(L.PRESETS.values())
    assert script.characters["them"] == L.PRESETS["f2"]
    assert L.PRESETS["f2"] in next(p.prompt for p in script.panels if p.focus == "them")
