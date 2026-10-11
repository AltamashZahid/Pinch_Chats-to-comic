import os

import cv2
import pytest

os.environ["PINCH_IMAGE_BACKEND"] = "placeholder"  # before the app builds its backend

from fastapi.testclient import TestClient  # noqa: E402

from pinch import server  # noqa: E402
from pinch.cache import ImageCache  # noqa: E402
from pinch.imagegen import PlaceholderBackend  # noqa: E402
from pinch.ocr import tesseract_available  # noqa: E402
from tests.synth import render_chat  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "OUTPUTS", tmp_path)
    monkeypatch.setitem(server._state, "backend", PlaceholderBackend())
    monkeypatch.setitem(server._state, "cache", ImageCache(tmp_path / "cache"))
    monkeypatch.delenv("PINCH_LLM_API_KEY", raising=False)
    return TestClient(server.app)


def _png(chat):
    ok, buf = cv2.imencode(".png", render_chat(chat))
    return buf.tobytes()


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Pinch" in r.text


def test_rejects_non_image(client):
    r = client.post("/api/comic", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_no_bubbles_is_422(client):
    blank = cv2.imencode(".png", render_chat([], header=False))[1].tobytes()
    r = client.post("/api/comic", files={"file": ("blank.png", blank, "image/png")})
    assert r.status_code == 422


@pytest.mark.skipif(not tesseract_available(), reason="tesseract not installed")
def test_end_to_end_then_cached(client, tmp_path):
    data = _png([("left", "are you coming tonight"), ("right", "yes see you there")])
    first = client.post("/api/comic", files={"file": ("chat.png", data, "image/png")}).json()
    assert [l["speaker"] for p in first["panels"] for l in p["lines"]] == ["them", "me"]
    assert (tmp_path / first["comic_url"].split("/")[-1]).exists()
    assert first["cache"]["miss"] == 1

    again = client.post("/api/comic", files={"file": ("chat.png", data, "image/png")}).json()
    assert again["panels"][0]["source"] == "exact"
