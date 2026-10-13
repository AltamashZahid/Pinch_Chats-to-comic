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
    monkeypatch.setenv("PINCH_LLM_API_KEY", "")  # empty, not deleted: load_env() would re-add .env's key
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
    assert first["cache"]["miss"] == 2  # one panel per turn

    again = client.post("/api/comic", files={"file": ("chat.png", data, "image/png")}).json()
    assert again["panels"][0]["source"] == "exact"


CHAT = "Rahul: bhai result aaya?\nAltamash: fail ho gaya yaar\nRahul: kya bakwas\nAltamash: haan yaar"


def test_chat_speakers(client):
    r = client.post("/api/chat/speakers", data={"text": CHAT}).json()
    assert r["messages"] == 4 and r["speakers"] == {"Rahul": 2, "Altamash": 2}


def test_comic_from_pasted_chat(client, tmp_path):
    r = client.post("/api/comic", data={"text": CHAT, "me": "Altamash", "last": "12"})
    assert r.status_code == 200, r.text
    data = r.json()
    speakers = [l["speaker"] for p in data["panels"] for l in p["lines"]]
    assert speakers == ["them", "me", "them", "me"]
    assert (tmp_path / data["comic_url"].split("/")[-1]).exists()


def test_comic_from_whatsapp_export_file_uses_last_n(client):
    export = "\n".join(f"12/10/26, 9:{i:02d} pm - {'A' if i % 2 else 'B'}: message {i}" for i in range(30))
    r = client.post("/api/comic", files={"file": ("chat.txt", export.encode(), "text/plain")},
                    data={"me": "A", "last": "4"})
    assert r.status_code == 200, r.text
    texts = [l["text"] for p in r.json()["panels"] for l in p["lines"]]
    assert texts == ["message 26", "message 27", "message 28", "message 29"]


def test_empty_chat_is_422(client):
    assert client.post("/api/comic", data={"text": "no colon lines here"}).status_code == 422
