import json

import pytest

from pinch import labeling

ANDROID = """12/10/26, 9:41 pm - Messages and calls are end-to-end encrypted. No one outside of this chat can read them.
12/10/26, 9:41 pm - Rahul: bhai kal ka plan?
12/10/26, 9:42 pm - Altamash: pata nahi yaar
thak gaya hu
12/10/26, 9:43 pm - Rahul: <Media omitted>
12/10/26, 9:44 pm - Rahul: chal theek hai"""

IOS = """[12/10/26, 9:41:05 PM] Rahul: bhai kal ka plan?
[12/10/26, 9:42:10 PM] Altamash: pata nahi yaar"""


def test_parses_android_export_with_multiline_and_skips_system_and_media():
    msgs = labeling.parse_chat_text(ANDROID)
    assert [(m["speaker"], m["text"]) for m in msgs] == [
        ("Rahul", "bhai kal ka plan?"), ("Altamash", "pata nahi yaar thak gaya hu"), ("Rahul", "chal theek hai")]


def test_parses_ios_export():
    assert [m["speaker"] for m in labeling.parse_chat_text(IOS)] == ["Rahul", "Altamash"]


def test_parses_plain_name_colon_lines():
    msgs = labeling.parse_chat_text("Rahul: hi\nMe: kya hua\n\nRahul: kuch nahi")
    assert [m["text"] for m in msgs] == ["hi", "kya hua", "kuch nahi"]


def test_context_is_previous_message():
    msgs = labeling.with_context([{"speaker": "a", "text": "x"}, {"speaker": "b", "text": "y"}])
    assert [m["prev"] for m in msgs] == ["", "x"]


def test_save_replaces_conversation_and_counts(tmp_path):
    store = tmp_path / "labels.jsonl"
    items = [{"speaker": "a", "prev": "", "text": "yay", "label": "joy", "model_pred": "joy"},
             {"speaker": "b", "prev": "yay", "text": "ugh", "label": "anger", "model_pred": "neutral"}]
    labeling.save_conversation("c1", items, store)
    labeling.save_conversation("c1", items[:1], store)  # relabel: replaces, doesn't duplicate
    labeling.save_conversation("c2", items, store)
    s = labeling.stats(store)
    assert s["messages"] == 3 and s["conversations"] == 2
    assert s["per_label"]["joy"] == 2 and s["model_agreement"] == round(2 / 3, 3)


def test_rejects_unknown_label(tmp_path):
    with pytest.raises(ValueError):
        labeling.save_conversation("c", [{"text": "x", "label": "happy"}], tmp_path / "l.jsonl")


def test_user_split_never_puts_one_conversation_in_two_splits(tmp_path, monkeypatch):
    from training import text_datasets
    store = tmp_path / "user" / "labels.jsonl"
    store.parent.mkdir()
    rows = [{"conversation": f"c{i}", "prev": "", "text": f"t{i}-{j}", "label": "joy"} for i in range(20) for j in range(3)]
    store.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(text_datasets, "TEXT", tmp_path)
    split = {s: {t.split("-")[0] for _, t, _ in text_datasets.user(s)} for s in ("train", "validation", "test")}
    assert split["test"] and split["validation"] and split["train"]
    assert not (split["train"] & split["test"]) and not (split["train"] & split["validation"])
    assert sum(len(v) for v in split.values()) == 20
