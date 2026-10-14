"""EmoBERTa-style input building, shared by training and the app.

Each turn is "Speaker: text"; the current turn sits between past and future turns, separated
by double </s> tokens, and context grows outward until the token budget is used.
"""


def turn_text(speaker: str, text: str) -> str:
    return f"{speaker}: {text}" if speaker else text


def build_input(tok, dialogue, i: int, max_len: int, future: bool = True) -> list[int]:
    """Token ids for turn i with as much past (and future) context as fits (EmoBERTa Alg. 1)."""
    sep = [tok.sep_token_id]
    cur = tok.encode(turn_text(*dialogue[i][:2]), add_special_tokens=False)[: max_len - 8]
    past, fut = [], []
    p, f = i - 1, i + 1
    budget = max_len - 2 - len(cur) - 4  # <s> </s> and two double separators
    while budget > 0 and (p >= 0 or (future and f < len(dialogue))):
        grew = False
        if p >= 0:
            t = tok.encode(turn_text(*dialogue[p][:2]), add_special_tokens=False)
            if len(t) + 1 <= budget:
                past = t + [tok.sep_token_id] + past if past else t
                budget -= len(t) + 1
                grew = True
            p -= 1
        if future and f < len(dialogue):
            t = tok.encode(turn_text(*dialogue[f][:2]), add_special_tokens=False)
            if len(t) + 1 <= budget:
                fut = fut + [tok.sep_token_id] + t if fut else t
                budget -= len(t) + 1
                grew = True
            f += 1
        if not grew and (p < 0 or budget <= 0) and (not future or f >= len(dialogue) or budget <= 0):
            break
    ids = [tok.cls_token_id]
    if past:
        ids += past + sep + sep
    ids += cur
    if fut:
        ids += sep + sep + fut
    return ids + sep
