# Pinch

Turn chat screenshots into AI-generated comic panels.

## Pipeline

1. **Dialogue extraction** (done): OpenCV finds chat bubbles (background-colour masking,
   fill-ratio and per-side colour filtering), speakers come from bubble alignment, and each
   bubble is OCR'd with Tesseract. Low-confidence reads (emoji, kaomoji) are dropped.
2. **Emotion classifier** (done): DistilBERT fine-tuned on GoEmotions, collapsed to 7 Ekman
   emotions, compared against a TF-IDF + logistic regression baseline. See
   `reports/emotion_metrics.json`.
3. **Scene prompts** (done): one LLM call writes the setting, a fixed look per character and a
   scene per panel. Image prompts are assembled in code, so character descriptions repeat
   verbatim across panels and expressions come from the classifier. Falls back to templates
   when no LLM is configured or the reply is malformed.
4. **Image generation and caching** (done): SD-Turbo runs locally in fp16 (~1 s/panel, 3 GB VRAM
   on an RTX 2050); Hugging Face and Pollinations backends are optional. Results are cached by
   exact hash and by MiniLM embedding similarity of the scene text, with the threshold (0.88)
   calibrated on paraphrase vs. different-scene pairs. Style, characters, expressions, seed and
   model must match exactly, so a "sad" panel never reuses a "happy" image.
5. **Comic layout** (done): panels on a grid with speech bubbles whose outline follows the
   message's emotion (starburst for surprise, spikes for anger, shaky for fear, sagging for sadness).
6. **Web app** (done): FastAPI backend and a single-page upload UI that shows the comic plus the
   per-message emotions, bubble styles and cache hits.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\pip install transformers datasets scikit-learn accelerate diffusers
```

Install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki). If it isn't on PATH or in
`D:\Tesseract-OCR`, set `PINCH_TESSERACT_CMD`.

Copy `.env.example` to `.env` and add an LLM key (optional; without one, prompts come from templates).

## Usage

```
.venv\Scripts\python training\train_emotion.py      # train the emotion model on the GPU
.venv\Scripts\python -m pinch path\to\screenshot.png            # script as JSON
.venv\Scripts\python -m pinch path\to\screenshot.png --images   # comic in outputs/
.venv\Scripts\python -m uvicorn pinch.server:app --port 8000     # web app at localhost:8000
.venv\Scripts\python -m pytest
```

Real screenshots for regression tests go in `samples/` with a `samples/expected.json`
(git-ignored, since they contain real conversations).
