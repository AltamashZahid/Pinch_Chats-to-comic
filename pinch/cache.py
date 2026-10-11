"""Prompt -> image cache with exact and semantic (embedding similarity) lookup.

Exact hashing alone almost never hits: an LLM rewords the same scene ("two friends
texting on a couch" vs "two friends on a sofa texting"). So each prompt is also embedded
with a small sentence encoder, and a lookup returns a stored image when a previous prompt
is close enough in meaning.

Only the scene text (action + location) is embedded. Everything that must match exactly
goes in a separate "exact key": art style, character looks, expressions, backend, model,
size and seed. Embedding the full prompt doesn't work: the shared style/character
boilerplate pushes unrelated scenes to cosine 0.87-0.93, overlapping with paraphrases.
On scene text alone, paraphrases measured 0.92-0.98 and different scenes 0.39-0.75
(see tests/test_cache.py), so the threshold sits at 0.88.
"""
import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "cache"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.88


def normalize(prompt: str) -> str:
    return " ".join(prompt.lower().replace(",", " , ").split())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MiniLMEmbedder:
    """Mean-pooled, L2-normalised MiniLM sentence embeddings (384-d)."""

    def __init__(self, model_name: str = EMBED_MODEL):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()

    def __call__(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with self._torch.no_grad():
            hidden = self.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vecs = (hidden * mask).sum(1) / mask.sum(1)
        vecs = self._torch.nn.functional.normalize(vecs, dim=-1)
        return vecs.numpy().astype(np.float32)


@dataclass
class Hit:
    path: Path
    kind: str           # "exact" or "semantic"
    similarity: float
    matched_prompt: str


class ImageCache:
    def __init__(self, root: Path = DEFAULT_DIR, embedder=None, threshold: float = DEFAULT_THRESHOLD):
        self.root = Path(root)
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        # The web server calls in from worker threads; the lock serialises DB access.
        self.db = sqlite3.connect(self.root / "index.sqlite", check_same_thread=False)
        self._lock = threading.Lock()
        self.db.execute("""CREATE TABLE IF NOT EXISTS entries (
            prompt_hash TEXT NOT NULL, exact_key TEXT NOT NULL, prompt TEXT NOT NULL,
            embedding BLOB NOT NULL, file TEXT NOT NULL, gen_seconds REAL, created REAL,
            PRIMARY KEY (prompt_hash, exact_key))""")
        self.db.execute("CREATE INDEX IF NOT EXISTS by_key ON entries(exact_key)")
        self._embedder = embedder
        self.threshold = threshold
        self.stats = {"exact": 0, "semantic": 0, "miss": 0, "seconds_saved": 0.0}

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = MiniLMEmbedder()
        return self._embedder

    @staticmethod
    def exact_key(**params) -> str:
        return sha(json.dumps(params, sort_keys=True))

    def get(self, prompt: str, exact_key: str) -> Hit | None:
        with self._lock:
            return self._get(prompt, exact_key)

    def _get(self, prompt: str, exact_key: str) -> Hit | None:
        norm = normalize(prompt)
        row = self.db.execute(
            "SELECT file, prompt, gen_seconds FROM entries WHERE prompt_hash=? AND exact_key=?",
            (sha(norm), exact_key)).fetchone()
        if row:
            return self._hit(row[0], "exact", 1.0, row[1], row[2])

        rows = self.db.execute("SELECT file, prompt, embedding, gen_seconds FROM entries WHERE exact_key=?",
                               (exact_key,)).fetchall()
        if rows:
            query = self.embedder([norm])[0]
            matrix = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
            sims = matrix @ query
            best = int(sims.argmax())
            if sims[best] >= self.threshold:
                f, p, _, secs = rows[best]
                return self._hit(f, "semantic", float(sims[best]), p, secs)
        self.stats["miss"] += 1
        return None

    def put(self, prompt: str, exact_key: str, image: bytes, gen_seconds: float = 0.0) -> Path:
        norm = normalize(prompt)
        name = sha(norm + exact_key)[:24] + (".png" if image[:4] == b"\x89PNG" else ".jpg")
        path = self.root / "images" / name
        path.write_bytes(image)
        emb = self.embedder([norm])[0]
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?,?)",
                            (sha(norm), exact_key, prompt, emb.tobytes(), name, gen_seconds, time.time()))
            self.db.commit()
        return path

    def _hit(self, file, kind, sim, matched, secs) -> Hit:
        self.stats[kind] += 1
        self.stats["seconds_saved"] += secs or 0.0
        return Hit(self.root / "images" / file, kind, sim, matched)

    def hit_rate(self) -> float:
        total = self.stats["exact"] + self.stats["semantic"] + self.stats["miss"]
        return (self.stats["exact"] + self.stats["semantic"]) / total if total else 0.0
