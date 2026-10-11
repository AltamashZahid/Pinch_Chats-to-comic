"""Run the fine-tuned emotion model on chat messages."""
from pathlib import Path

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "emotion"


class EmotionClassifier:
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR):
        import torch  # heavy imports kept out of module import time
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def predict(self, texts: list[str]) -> list[tuple[str, float]]:
        """Returns (emotion, probability) for each text."""
        if not texts:
            return []
        enc = self.tokenizer(texts, truncation=True, max_length=64, padding=True,
                             return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            probs = self.model(**enc).logits.softmax(-1).cpu()
        best = probs.max(-1)
        return [(self.labels[i], round(float(p), 3)) for p, i in zip(best.values, best.indices)]
