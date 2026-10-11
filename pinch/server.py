"""Web app: upload a chat screenshot, get a comic page back.

    .venv\\Scripts\\python -m uvicorn pinch.server:app --port 8000
"""
import hashlib
import threading
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .cache import ImageCache
from .imagegen import ImageGenError, backend_from_env
from .layout import compose_page
from .pipeline import load_env, screenshot_to_script
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

    backend, cache = _resources()
    before = dict(cache.stats)
    with _gpu_lock:
        rendered = render_panels(script, backend, cache)
    name = hashlib.sha256(data).hexdigest()[:16] + "_comic.png"
    compose_page(rendered, OUTPUTS / name)
    run = {k: round(cache.stats[k] - before[k], 1) for k in before}
    return {
        "comic_url": f"/outputs/{name}",
        "script_source": script.source,
        "setting": script.setting,
        "panels": [{
            "source": r.source,
            "seconds": round(r.seconds, 2),
            "prompt": r.panel.prompt,
            "lines": [{"speaker": l.speaker, "text": l.text, "emotion": l.emotion,
                       "confidence": l.confidence, "bubble": l.bubble} for l in r.panel.lines],
        } for r in rendered],
        "cache": run,
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/comic")
async def comic(file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Screenshot is larger than 10 MB.")
    try:
        return await run_in_threadpool(make_comic, data)
    except ImageGenError as e:
        raise HTTPException(502, f"Image generation failed: {e}") from e
