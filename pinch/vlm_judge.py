"""Ask a vision-language model which emotion a comic character's face shows.

CLIP zero-shot ("a comic drawing of a person with a sad face") turned out to be a poor
judge of drawn expressions: it labelled clearly sad and terrified faces as neutral. A VLM
that actually looks at the face does much better (see eval/judge_agreement.py).
"""
import base64
import json
import os
import re
import time
from pathlib import Path

import httpx

from .emotions import LABELS

VLM_MODEL = "qwen/qwen3.8-27b"
QUESTION = (
    "This is a comic panel. Look at the main character's FACE and body language only "
    "(ignore any text). Which one emotion does it most clearly show? Choose exactly one of: "
    + ", ".join(LABELS) + ". If no face is visible, answer neutral. "
    'Reply with JSON only: {"emotion": "<one label>"}'
)


class VLMJudge:
    def __init__(self, model: str = VLM_MODEL, api_key: str | None = None,
                 base_url: str = "https://api.groq.com/openai/v1", timeout: float = 60.0,
                 transport: httpx.BaseTransport | None = None):
        self.model, self.base_url, self.timeout, self.transport = model, base_url, timeout, transport
        self.api_key = api_key or os.environ.get("PINCH_LLM_API_KEY")
        if not self.api_key:
            raise RuntimeError("VLM judge needs PINCH_LLM_API_KEY (a Groq key)")

    def emotion(self, image: Path | bytes) -> str:
        data = image if isinstance(image, bytes) else Path(image).read_bytes()
        mime = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"
        url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
        body = {"model": self.model, "temperature": 0, "max_tokens": 2000,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": QUESTION},
                    {"type": "image_url", "image_url": {"url": url}}]}]}
        with httpx.Client(timeout=self.timeout, transport=self.transport) as http:
            for attempt in range(6):  # rate limits (429) and transient 5xx: back off and retry
                try:
                    r = http.post(f"{self.base_url}/chat/completions", json=body,
                                  headers={"Authorization": f"Bearer {self.api_key}"})
                except httpx.TransportError:  # dropped connection / DNS blip
                    if attempt == 5:
                        raise
                    time.sleep(2 ** (attempt + 1))
                    continue
                if r.status_code != 429 and r.status_code < 500:
                    break
                time.sleep(float(r.headers.get("retry-after", 2 ** (attempt + 1))))
        r.raise_for_status()
        return parse_emotion(r.json()["choices"][0]["message"]["content"])


def parse_emotion(text: str) -> str:
    """Pull a label out of the reply: JSON first, then the last label word mentioned."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{[^{}]*\}", text)
    if m:
        try:
            label = str(json.loads(m.group(0)).get("emotion", "")).lower().strip()
            if label in LABELS:
                return label
        except ValueError:
            pass
    found = [(text.lower().rfind(l), l) for l in LABELS if l in text.lower()]
    return max(found)[1] if found else "neutral"
