import json

import pytest

from pinch import memory as M
from pinch.scenes import Line, write_script
from tests.test_scenes import GOOD_REPLY, LINES, fake_llm


@pytest.fixture
def store():
    return M.CastStore(M.MemoryKV(), capacity=2)


# --- LRU cache --------------------------------------------------------------------------

def test_lru_evicts_least_recently_used():
    c = M.LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1      # "a" is now the most recent
    c.put("c", 3)               # so "b" goes
    assert "b" not in c and "a" in c and "c" in c
    assert c.evictions == 1 and c.hits == 1


def test_lru_counts_misses():
    c = M.LRUCache(2)
    assert c.get("x") is None
    assert c.misses == 1


# --- cards: cache -> disk -> allocate ---------------------------------------------------

def test_names_are_normalised():
    assert M.person_key("  Rahul   K ") == M.person_key("rahul k") == "rahul k"
    assert len(M.person_key("x" * 500).encode()) <= M.MAX_NAME_BYTES


def test_unknown_person_has_no_card(store):
    assert store.card("Zoya") is None


def test_card_round_trip_and_page_fault_after_eviction(store):
    store.save(M.Card("Rahul", "boy with glasses"), M.Card("Zoya", "girl with curly hair"))
    store.save(M.Card("Aman", "tall guy, red cap"))          # capacity 2: Rahul is evicted
    assert "rahul" not in store.cache
    faults = store.page_faults
    assert store.card("rahul").look == "boy with glasses"     # reloaded from the backing store
    assert store.page_faults == faults + 1
    assert store.card("rahul").look == "boy with glasses"     # now a cache hit
    assert store.page_faults == faults + 1


def test_stories_are_ordered_shared_and_limited(store):
    store.add_story(["self", "Rahul"], "Rahul failed physics.", when="2026-10-01")
    store.add_story(["self", "Rahul"], "Rahul got a dog.", when="2026-10-09")
    store.add_story(["self", "Zoya"], "Zoya moved to Pune.", when="2026-10-05")
    assert store.stories("Rahul") == ["Rahul failed physics.", "Rahul got a dog."]
    assert store.stories("Rahul", n=1) == ["Rahul got a dog."]
    assert store.stories("self") == ["Rahul failed physics.", "Zoya moved to Pune.", "Rahul got a dog."]
    assert store.history_for({"me": "self", "them": "Zoya"}) == ["Zoya moved to Pune."]
    assert store.history_for({"me": "self"}) == []  # no other person: no unrelated history


def test_remember_creates_then_reuses_cards(store):
    people = {"me": "self", "them": "Rahul"}
    store.remember(people, {"me": "guy in a blue hoodie", "them": "boy with glasses"}, "Rahul failed.")
    store.remember(people, {"me": "SOMEONE ELSE", "them": "SOMEONE ELSE"}, "Rahul passed.")
    assert store.card("Rahul").look == "boy with glasses"      # first look is kept
    assert store.card("Rahul").comics == 2
    assert store.cast_for(people)["me"].look == "guy in a blue hoodie"


# --- PersistKV backing store ------------------------------------------------------------

def test_persistkv_survives_restart(tmp_path):
    pytest.importorskip("persistkv")
    kv = M.PersistKVStore(tmp_path / "mem")
    s = M.CastStore(kv)
    s.remember({"me": "self", "them": "Rahul"}, {"me": "blue hoodie", "them": "glasses"}, "Rahul failed.")
    kv.close()
    kv = M.PersistKVStore(tmp_path / "mem")                    # a fresh process would see the same
    s = M.CastStore(kv)
    assert s.card("Rahul").look == "glasses"
    assert s.history_for({"me": "self", "them": "rahul"}) == ["Rahul failed."]
    assert kv.durable
    kv.close()


# --- scene writing with memory ----------------------------------------------------------

def test_fixed_look_overrides_llm_and_summary_is_parsed():
    reply = dict(GOOD_REPLY, summary="Rahul panics about an exam.")
    script = write_script(LINES, fake_llm(json.dumps(reply)), mode="story",
                          looks={"them": "boy with round glasses"}, history=["Rahul failed physics."])
    assert script.characters["them"] == "boy with round glasses"
    assert script.characters["me"] == GOOD_REPLY["characters"]["me"]
    assert script.summary == "Rahul panics about an exam."
    assert any("boy with round glasses" in p.prompt for p in script.panels if p.focus == "them")


def test_prompt_carries_fixed_looks_and_history():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        import httpx
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(GOOD_REPLY)}}]})
    import httpx
    from pinch.llm import LLMClient
    llm = LLMClient("https://fake/v1", "key", "model", transport=httpx.MockTransport(handler))
    write_script(LINES, llm, looks={"them": "boy with round glasses"}, history=["Rahul failed physics."])
    user = seen["body"]["messages"][1]["content"]
    assert 'FIXED look for them: "boy with round glasses"' in user
    assert "Earlier: Rahul failed physics." in user


def test_chat_to_script_remembers_people_across_comics(monkeypatch):
    from pinch import pipeline
    monkeypatch.setenv("PINCH_MEMORY", "on")
    monkeypatch.setattr(M, "_store", M.CastStore(M.MemoryKV()))
    monkeypatch.setattr(pipeline, "classify_dialogue", lambda turns: [("neutral", 0.9)] * len(turns))
    first = dict(GOOD_REPLY, summary="Rahul is stressed about exams.")
    monkeypatch.setattr(pipeline.LLMClient, "from_env", classmethod(lambda cls: fake_llm(json.dumps(first))))
    chat = [{"speaker": "Rahul", "text": "exam tomorrow"}, {"speaker": "Altamash", "text": "you got this"}]
    pipeline.chat_to_script(chat, "Altamash")

    other = dict(GOOD_REPLY, characters={"me": "different", "them": "different"}, summary="Rahul passed.")
    monkeypatch.setattr(pipeline.LLMClient, "from_env", classmethod(lambda cls: fake_llm(json.dumps(other))))
    script = pipeline.chat_to_script(chat, "Altamash")
    assert script.characters == GOOD_REPLY["characters"]      # same people, same looks
    assert M._store.stories("Rahul") == ["Rahul is stressed about exams.", "Rahul passed."]


def test_screenshot_name_is_remembered(monkeypatch):
    from pinch import pipeline
    from pinch.models import Box, Message
    monkeypatch.setenv("PINCH_MEMORY", "on")
    monkeypatch.setattr(M, "_store", M.CastStore(M.MemoryKV()))
    monkeypatch.setattr(pipeline, "classify_dialogue", lambda turns: [("neutral", 0.9)] * len(turns))
    monkeypatch.setattr(pipeline, "extract_dialogue", lambda img: [
        Message("me", "exam tomorrow", Box(0, 0, 1, 1)), Message("them", "lol good luck", Box(0, 1, 1, 1))])
    monkeypatch.setattr(pipeline.LLMClient, "from_env", classmethod(lambda cls: fake_llm(json.dumps(GOOD_REPLY))))
    pipeline.screenshot_to_script(None, them="Aadi Singhal")
    assert M._store.card("aadi singhal").look == GOOD_REPLY["characters"]["them"]


def test_template_scripts_are_not_remembered(monkeypatch):
    from pinch import pipeline
    monkeypatch.setenv("PINCH_MEMORY", "on")
    monkeypatch.setattr(M, "_store", M.CastStore(M.MemoryKV()))
    monkeypatch.setattr(pipeline, "classify_dialogue", lambda turns: [("neutral", 0.9)] * len(turns))
    monkeypatch.setattr(pipeline.LLMClient, "from_env", classmethod(lambda cls: None))
    pipeline.chat_to_script([{"speaker": "Rahul", "text": "hi"}, {"speaker": "Me", "text": "yo"}], "Me")
    assert M._store.card("Rahul") is None


def test_memory_off_returns_no_store(monkeypatch):
    monkeypatch.setenv("PINCH_MEMORY", "off")
    assert M.default_store() is None
