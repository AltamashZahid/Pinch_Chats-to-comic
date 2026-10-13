"""Facial-expression control: draw a face-landmark sketch for the wanted emotion and let a
ControlNet (CrucibleAI ControlNetMediaPipeFace, SD 1.5) draw a face with exactly that shape.

Prompts alone leave SD guessing what "terrified" looks like, and on our 700-panel test it
got fear/disgust/surprise right only ~26% of the time. A landmark sketch pins down the
eyebrows, eyes and mouth instead.

Expression templates come from our own blind-labelled panels (data/expr_pool): MediaPipe
reads the face in each panel whose drawn emotion is known, and the landmarks are stored per
emotion in data/face_templates/<emotion>.npy (normalised to the face's bounding box).
The sketch is drawn exactly like the ControlNet's training data (laion_face_common.py):
contours on black, BGR colours from that file, pupils as small squares.
"""
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "data" / "face_templates"
LANDMARKER = Path("D:/tmp/face_landmarker.task")

# (color as drawn in RGB after the reference code's BGR->RGB flip, thickness 2)
_SPEC = {
    "FACE_LANDMARKS_FACE_OVAL": (10, 200, 10),
    "FACE_LANDMARKS_LEFT_EYE": (10, 200, 180),
    "FACE_LANDMARKS_LEFT_EYEBROW": (10, 220, 180),
    "FACE_LANDMARKS_RIGHT_EYE": (180, 200, 10),
    "FACE_LANDMARKS_RIGHT_EYEBROW": (180, 220, 10),
    "FACE_LANDMARKS_LIPS": (10, 180, 10),
}
_PUPILS = {468: (250, 200, 10), 473: (10, 200, 250)}


def _connections():
    from mediapipe.tasks.python import vision
    C = vision.FaceLandmarksConnections
    return {name: [(c.start, c.end) for c in getattr(C, name)] for name in _SPEC}


def landmarker():
    from mediapipe.tasks.python import BaseOptions, vision
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(LANDMARKER)), num_faces=1))


def normalise(points: np.ndarray) -> np.ndarray:
    """Landmarks (N, 2) in image fractions -> face-box coordinates in [0, 1]."""
    lo, hi = points.min(0), points.max(0)
    return (points - lo) / np.maximum(hi - lo, 1e-6)


def build_templates() -> dict[str, int]:
    """Read the face in every blind-labelled pool panel; save landmarks per (true) emotion."""
    import mediapipe as mp
    labels = json.loads((ROOT / "data" / "expr_pool" / "labels.json").read_text())["labels"]
    lm = landmarker()
    per: dict[str, list] = {}
    for pid, label in labels.items():
        if label == "unclear":
            continue
        res = lm.detect(mp.Image.create_from_file(str(ROOT / "data" / "expr_pool" / "images" / f"{pid}.png")))
        if res.face_landmarks:
            pts = np.array([(p.x, p.y) for p in res.face_landmarks[0]], dtype=np.float32)
            per.setdefault(label, []).append(normalise(pts))
    TEMPLATES.mkdir(parents=True, exist_ok=True)
    for label, arr in per.items():
        np.save(TEMPLATES / f"{label}.npy", np.stack(arr))
    return {k: len(v) for k, v in per.items()}


def load_template(emotion: str, rng: random.Random) -> np.ndarray | None:
    path = TEMPLATES / f"{emotion}.npy"
    if not path.exists():
        return None
    arr = np.load(path)
    return arr[rng.randrange(len(arr))]


# Face size (fraction of the panel) per camera shot.
FACE_SIZE = {"extreme close-up": 0.78, "close-up": 0.6, "close-up portrait": 0.6, "medium close-up": 0.45,
             "medium shot": 0.34, "over-the-shoulder shot": 0.3, "wide shot": 0.22}


def sketch(template: np.ndarray, size: int = 512, shot: str = "close-up", center=(0.5, 0.42)):
    """Draw the template as a ControlNet conditioning image (PIL, RGB)."""
    import cv2
    from PIL import Image
    face = FACE_SIZE.get(shot, 0.5) * size
    aspect = 1.25  # faces are taller than wide
    w, h = face / aspect, face
    cx, cy = center[0] * size, center[1] * size
    pts = np.stack([cx - w / 2 + template[:, 0] * w, cy - h / 2 + template[:, 1] * h], 1).astype(int)
    img = np.zeros((size, size, 3), np.uint8)
    for name, edges in _connections().items():
        for a, b in edges:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), _SPEC[name], 2)
    for idx, color in _PUPILS.items():
        if idx < len(pts):
            x, y = pts[idx]
            img[max(0, y - 2):y + 2, max(0, x - 2):x + 2] = color
    return Image.fromarray(img)


if __name__ == "__main__":
    print(build_templates())
