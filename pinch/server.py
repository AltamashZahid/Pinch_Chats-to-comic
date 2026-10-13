"""Web app: upload a chat screenshot, get a comic page back.

    .venv\\Scripts\\python -m uvicorn pinch.server:app --port 8000
"""
import hashlib
import threading
from pathlib import Path

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .cache import ImageCache
from .imagegen import ImageGenError, backend_from_env
from . import labeling
from .layout import compose_page
from .pipeline import MAX_CHAT_MESSAGES, chat_to_script, load_env, screenshot_to_script
from .render import render_panels

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 10 * 1024 * 1024

load_env()
OUTPUTS.mkdir(exist_ok=True)
app = FastAPI(title="Pinch")
app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")

_state: dict = {}
_gpu_lock = threading.Lock()  # one generation at a time: a 4 GB GPU can't hold two pipelines


def _resources():
    if not _state:
        _state["backend"] = backend_from_env()
        _state["cache"] = ImageCache()
    return _state["backend"], _state["cache"]


def make_comic(data: bytes) -> dict:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "That file isn't an image we can read.")
    script = screenshot_to_script(img)
    if not script.panels:
        raise HTTPException(422, "No chat bubbles with readable text were found in that screenshot.")
    return render_comic(script, data)


def make_comic_from_chat(raw: str, me: str | None, last: int) -> dict:
    msgs = labeling.parse_chat_text(raw)
    if not msgs:
        raise HTTPException(422, "No messages found. Paste lines like 'Name: message' or a WhatsApp export.")
    script = chat_to_script(msgs, me or None, last)
    return render_comic(script, f"{me}|{last}|{raw}".encode())


def render_comic(script, key: bytes) -> dict:
    """Draw the panels (cached) and lay out the page. `key` names the output file."""
    backend, cache = _resources()
    before = dict(cache.stats)
    with _gpu_lock:
        rendered = render_panels(script, backend, cache)
    name = hashlib.sha256(key).hexdigest()[:16] + "_comic.png"
    compose_page(rendered, OUTPUTS / name)
    run = {k: round(cache.stats[k] - before[k], 1) for k in before}
    return {
        "comic_url": f"/outputs/{name}",
        "script_source": script.source,
        "locations": script.locations,
        "panels": [{
            "source": r.source,
            "focus": r.panel.focus,
            "shot": r.panel.shot,
            "seconds": round(r.seconds, 2),
            "prompt": r.panel.prompt,
            "lines": [{"speaker": l.speaker, "text": l.text, "emoji": l.emoji, "emotion": l.emotion,
                       "confidence": l.confidence, "bubble": l.bubble, "ironic": l.ironic,
                       "situation": l.situation} for l in r.panel.lines],
        } for r in rendered],
        "cache": run,
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/label")
def label_page():
    return FileResponse(STATIC / "label.html")


def _parse_for_labelling(data: bytes | None, filename: str, text: str) -> dict:
    from .ocr import extract_dialogue
    from .pipeline import classify
    if data and not filename.lower().endswith(".txt"):
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Upload a WhatsApp export (.txt) or a screenshot.")
        msgs = [{"speaker": m.speaker, "text": m.text} for m in extract_dialogue(img) if m.text]
    else:
        raw = data.decode("utf-8", errors="replace") if data else text
        msgs = labeling.parse_chat_text(raw)
    if not msgs:
        raise HTTPException(422, "No messages found. Paste lines like 'Name: message' or a WhatsApp export.")
    msgs = labeling.with_context(msgs)
    preds = classify([m["text"] for m in msgs], [m["prev"] for m in msgs])
    for m, (label, p) in zip(msgs, preds):
        m["model_pred"], m["confidence"] = label, p
    return {"conversation": labeling.conversation_id(msgs), "messages": msgs}


@app.post("/api/label/parse")
async def label_parse(file: UploadFile | None = File(None), text: str = Form("")):
    data = await file.read(MAX_UPLOAD + 1) if file else None
    if data and len(data) > MAX_UPLOAD:
        raise HTTPException(413, "File is larger than 10 MB.")
    return await run_in_threadpool(_parse_for_labelling, data, file.filename if file else "", text)


@app.post("/api/label/save")
def label_save(payload: dict = Body(...)):
    try:
        n = labeling.save_conversation(payload["conversation"], payload["items"])
    except (KeyError, ValueError) as e:
        raise HTTPException(400, f"Bad labels: {e}") from e
    return {"saved": n, **labeling.stats()}


@app.get("/api/label/stats")
def label_stats():
    return labeling.stats()


@app.post("/api/chat/speakers")
async def chat_speakers(file: UploadFile | None = File(None), text: str = Form("")):
    """Who's in a typed chat, so the page can ask which one is the user."""
    raw = (await file.read(MAX_UPLOAD)).decode("utf-8", errors="replace") if file else text
    msgs = labeling.parse_chat_text(raw)
    counts: dict[str, int] = {}
    for m in msgs:
        counts[m["speaker"]] = counts.get(m["speaker"], 0) + 1
    return {"messages": len(msgs), "speakers": counts}


@app.post("/api/comic")
async def comic(file: UploadFile | None = File(None), text: str = Form(""), me: str = Form(""),
                last: int = Form(MAX_CHAT_MESSAGES)):
    """A screenshot (image file), or a typed chat: a WhatsApp export (.txt) or pasted lines."""
    data = await file.read(MAX_UPLOAD + 1) if file else None
    if data and len(data) > MAX_UPLOAD:
        raise HTTPException(413, "File is larger than 10 MB.")
    is_text = file is None or (file.filename or "").lower().endswith(".txt")
    try:
        if is_text:
            raw = data.decode("utf-8", errors="replace") if data else text
            return await run_in_threadpool(make_comic_from_chat, raw, me, max(2, min(last, 24)))
        return await run_in_threadpool(make_comic, data)
    except ImageGenError as e:
        raise HTTPException(502, f"Image generation failed: {e}") from e
