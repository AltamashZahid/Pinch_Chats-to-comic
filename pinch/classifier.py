"""Run the fine-tuned emotion model on chat messages."""
import os
from pathlib import Path

_MODELS = Path(__file__).resolve().parent.parent / "models"
# v2 (HingRoBERTa fine-tuned on GoEmotions + DailyDialog + MELD + Hinglish + tweets) when it
# exists: mean macro-F1 0.546 across five test domains vs 0.372 for v1 (reports/*_eval.json).
DEFAULT_MODEL_DIR = Path(os.environ.get("PINCH_EMOTION_MODEL") or
                         (_MODELS / "emotion_v2_hing" if (_MODELS / "emotion_v2_hing" / "config.json").exists()
                          else _MODELS / "emotion"))


class EmotionClassifier:
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR):
        import torch  # heavy imports kept out of module import time
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        # CPU by default: a chat is a few dozen lines (<1 s), and the 4 GB GPU is needed for
        # image generation + the expression judge. PINCH_CLASSIFIER_DEVICE=cuda to override.
        self.device = torch.device(os.environ.get("PINCH_CLASSIFIER_DEVICE", "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        # Models trained with --context read "message </s> previous message".
        self.context = bool(getattr(self.model.config, "pinch_context", False))
        # Per-emotion logit biases tuned with the Grey Wolf Optimizer (training/tune_text_biases.py).
        import json
        bias_file = Path(model_dir) / "pinch_biases.json"
        b = json.loads(bias_file.read_text()) if bias_file.exists() and os.environ.get("PINCH_TEXT_BIASES", "on") != "off" else {}
        self.bias = torch.tensor([b.get(l, 0.0) for l in self.labels])

    def predict(self, texts: list[str], prevs: list[str] | None = None) -> list[tuple[str, float]]:
        """Returns (emotion, probability) for each text. `prevs` are the previous messages in
        the conversation (used only by context models)."""
        if not texts:
            return []
        if self.context:
            prevs = prevs or [""] * len(texts)
            enc = self.tokenizer(texts, [p or "" for p in prevs], truncation="longest_first", max_length=96,
                                 padding=True, return_tensors="pt").to(self.device)
        else:
            enc = self.tokenizer(texts, truncation=True, max_length=64, padding=True,
                                 return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            probs = (self.model(**enc).logits.float().cpu() + self.bias).softmax(-1)
        best = probs.max(-1)
        return [(self.labels[i], round(float(p), 3)) for p, i in zip(best.values, best.indices)]


if __name__ == "__main__":
    # python -m pinch.classifier "text one" "text two"   or no args for an interactive prompt
    import sys
    clf = EmotionClassifier()
    texts = sys.argv[1:]
    if texts:
        for text, (label, p) in zip(texts, clf.predict(texts)):
            print(f"{label:9} {p:.2f}  {text}")
    else:
        print("Type a message (empty line to quit)")
        while text := input("> ").strip():
            label, p = clf.predict([text])[0]
            print(f"  {label} ({p:.0%})")
