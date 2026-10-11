# Pinch

Turn chat screenshots into AI-generated comic panels.

## Pipeline

1. **Dialogue extraction** (done): detect chat bubbles with OpenCV, assign speakers by bubble
   alignment, OCR each bubble with Tesseract.
2. **Emotion classifier**: fine-tuned DistilBERT on GoEmotions; drives expressions and bubble style.
3. **Scene prompts**: an LLM turns dialogue and emotions into panel descriptions.
4. **Image generation and caching**: diffusion backend with an embedding-similarity cache.
5. **Comic layout**: compose panels and speech bubbles.
6. **Web app**: FastAPI backend and a simple frontend.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
```

Install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki). If it's not on PATH,
set `PINCH_TESSERACT_CMD` to the full path of `tesseract.exe`.

## Usage

```
.venv\Scripts\python -m pinch path\to\screenshot.png
.venv\Scripts\python -m pytest
```
