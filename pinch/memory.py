"""Who's who across comics: each person's fixed look, and what happened in past comics.

Laid out like an OS memory hierarchy:

    LRU cache (RAM)  ->  PersistKV (disk)  ->  allocate a new card
       hit: free         page fault: one         first time this person
                         B+Tree lookup           appears in any comic

The backing store is PersistKV (github.com/AltamashZahid/persist_KV), a crash-safe C++
key-value engine (B+Tree + write-ahead log), used through its pybind11 module. Without it
installed, a plain in-memory store with the same interface keeps everything working
(nothing survives a restart then).

Keys. B+Tree keys are sorted, so one prefix scan returns a person's whole history in order:

    card:<person>                   -> {"name", "look", "comics", "created", "updated"}
    story:<person>:<timestamp>      -> {"summary", "with"}

<person> is the lower-cased name; the app's user is always "self". A comic with two people
writes its story under both of them in one atomic WriteBatch (one fsync).

PINCH_MEMORY=off disables memory; PINCH_MEMORY_DIR sets where it lives (default data/memory).
"""
import json
import os
import re
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SELF = "self"
MAX_NAME_BYTES = 96  # keys are capped at 128 bytes by PersistKV; leave room for the prefix + timestamp


def person_key(name: str) -> str:
    """Stable id for a display name: "  Rahul  K " and "rahul k" are the same person."""
    key = re.sub(r"\s+", " ", name.strip().lower())
    return key.encode()[:MAX_NAME_BYTES].decode(errors="ignore") or "unknown"


def now() -> str:
    return datetime.now().isoformat(timespec="microseconds")


@dataclass
class Card:
    name: str
    look: str               # fixed visual description, repeated in every panel
    comics: int = 0         # how many comics this person has been in
    created: str = field(default_factory=now)
    updated: str = field(default_factory=now)


class LRUCache:
    """Fixed number of entries; the least recently used one is evicted first.
    An OrderedDict keeps entries in use order: get/put move a key to the end, eviction pops
    the front. All O(1). Thread-safe (the web server handles requests on several threads)."""

    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = self.evictions = 0

    def get(self, key):
        with self._lock:
            if key not in self._d:
                self.misses += 1
                return None
            self.hits += 1
            self._d.move_to_end(key)
            return self._d[key]

    def put(self, key, value) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)
                self.evictions += 1

    def __len__(self) -> int:
        return len(self._d)

    def __contains__(self, key) -> bool:
        return key in self._d


class MemoryKV:
    """Same interface as the PersistKV adapter, in a dict. Nothing survives a restart."""
    durable = False

    def __init__(self):
        self._d: dict[bytes, bytes] = {}
        self._lock = threading.Lock()

    def get(self, key: bytes) -> bytes | None:
        return self._d.get(key)

    def write(self, puts: dict[bytes, bytes]) -> None:
        with self._lock:
            self._d.update(puts)

    def scan_prefix(self, prefix: bytes) -> list[tuple[bytes, bytes]]:
        with self._lock:
            return sorted((k, v) for k, v in self._d.items() if k.startswith(prefix))

    def close(self) -> None:
        pass


class PersistKVStore:
    """The PersistKV engine. Every write() is one atomic WriteBatch: one WAL record, one fsync."""
    durable = True

    def __init__(self, path: str | Path):
        import persistkv
        Path(path).mkdir(parents=True, exist_ok=True)
        self._pkv = persistkv
        self._db = persistkv.DB(str(path))

    def get(self, key: bytes) -> bytes | None:
        return self._db.get(key)

    def write(self, puts: dict[bytes, bytes]) -> None:
        batch = self._pkv.WriteBatch()
        for k, v in puts.items():
            batch.put(k, v)
        self._db.write(batch)

    def scan_prefix(self, prefix: bytes) -> list[tuple[bytes, bytes]]:
        return self._db.scan_prefix(prefix)

    def close(self) -> None:
        self._db.close()


def open_kv(path: str | Path):
    """PersistKV if its Python module is installed, else the in-memory store."""
    try:
        return PersistKVStore(path)
    except ImportError:
        print("[pinch] persistkv not installed: character memory lasts until restart "
              "(pip install git+https://github.com/AltamashZahid/persist_KV.git)")
        return MemoryKV()


class CastStore:
    def __init__(self, kv, capacity: int = 64):
        self.kv = kv
        self.cache = LRUCache(capacity)
        self.page_faults = 0  # cache misses that went to disk

    # --- cards ---------------------------------------------------------------------------

    def card(self, name: str) -> Card | None:
        """RAM first; on a miss, load from disk (a page fault). None if never seen."""
        pid = person_key(name)
        hit = self.cache.get(pid)
        if hit is not None:
            return hit
        self.page_faults += 1
        raw = self.kv.get(f"card:{pid}".encode())
        if raw is None:
            return None
        card = Card(**json.loads(raw))
        self.cache.put(pid, card)
        return card

    def save(self, *cards: Card) -> None:
        """Write-through: disk first (atomically, all cards in one batch), then the cache."""
        if not cards:
            return
        self.kv.write({f"card:{person_key(c.name)}".encode(): json.dumps(asdict(c)).encode() for c in cards})
        for c in cards:
            self.cache.put(person_key(c.name), c)

    # --- stories -------------------------------------------------------------------------

    def add_story(self, names: list[str], summary: str, when: str | None = None) -> None:
        """One comic's summary, stored under every person in it, in one atomic batch."""
        names = [n for n in dict.fromkeys(names) if n]
        if not summary or not names:
            return
        ts = when or now()
        self.kv.write({f"story:{person_key(n)}:{ts}".encode():
                       json.dumps({"summary": summary, "with": [m for m in names if m != n]}).encode()
                       for n in names})

    def stories(self, name: str, n: int = 3) -> list[str]:
        """The person's last n story summaries, oldest first."""
        rows = self.kv.scan_prefix(f"story:{person_key(name)}:".encode())
        return [json.loads(v)["summary"] for _, v in rows[-n:]]

    # --- one comic -----------------------------------------------------------------------

    def cast_for(self, names: dict[str, str]) -> dict[str, Card]:
        """Known cards for the roles in a comic ({"me": "self", "them": "Rahul"})."""
        out = {}
        for role, name in names.items():
            card = self.card(name)
            if card is not None:
                out[role] = card
        return out

    def history_for(self, names: dict[str, str], n: int = 3) -> list[str]:
        """Recent stories with the other person. None without one: the user's own history
        spans every chat, so it would bring up unrelated events."""
        others = [v for v in names.values() if person_key(v) != SELF]
        seen, out = set(), []
        for name in others:
            for s in self.stories(name, n):
                if s not in seen:
                    seen.add(s)
                    out.append(s)
        return out[-n:]

    def remember(self, names: dict[str, str], looks: dict[str, str], summary: str = "") -> None:
        """After a comic: create cards for new people, count the comic, save its story."""
        cards = []
        for role, name in names.items():
            card = self.card(name)
            if card is None:
                if not looks.get(role):
                    continue
                card = Card(name=name, look=looks[role])
            card.comics += 1
            card.updated = now()
            cards.append(card)
        self.save(*cards)
        self.add_story(list(names.values()), summary)

    def stats(self) -> dict:
        return {"durable": self.kv.durable, "cached": len(self.cache), "hits": self.cache.hits,
                "page_faults": self.page_faults, "evictions": self.cache.evictions}


_store: CastStore | None = None
_store_lock = threading.Lock()


def default_store() -> CastStore | None:
    """The app's shared store, opened on first use; None when PINCH_MEMORY=off."""
    global _store
    if os.environ.get("PINCH_MEMORY", "on").lower() in ("0", "off", "false", "no"):
        return None
    with _store_lock:
        if _store is None:
            _store = CastStore(open_kv(os.environ.get("PINCH_MEMORY_DIR") or ROOT / "data" / "memory"))
        return _store
