# Pinterest-style Content Recommendation System

Content-based recommendation system that mimics Pinterest "pins" (image + title + description + tags). Uses LangChain to embed pins with a local `sentence-transformers` model, stores vectors in a local Chroma index, and serves recommendations via FastAPI.

This is a standalone project with a mock dataset — no Pinterest API, no OpenAI billing, no credentials required. Embeddings run fully on-device via HuggingFace `sentence-transformers/all-MiniLM-L6-v2`.

## Live demo

Not deployed publicly yet. Run it locally (see **Setup** below) — once running:

- Interactive API docs: http://localhost:8000/docs
- `curl "http://localhost:8000/recommend/pin/p001?k=5"`
- `curl "http://localhost:8000/recommend/query?q=cozy%20living%20room&k=5"`

Public deployment (Render/Fly.io/Railway) TBD.

## How it works

- Each pin (`data/pins.json`) is converted into a LangChain `Document`:
  - `page_content`: `"{title}. {description}. Tags: {tags}"`
  - `metadata`: `{id, title, board, image_url, tags}`
- Documents are embedded with `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")` and stored in a persisted Chroma collection (`chroma_db/`). Model weights download once from HuggingFace on first run (~90MB), then run locally — no API key needed.
- Pin images are real photos self-hosted in `data/images/` (sourced from Lorem Picsum, seeded per pin id for reproducibility), served by the API itself via a static file mount at `/images`.
- Two recommendation modes:
  - **By pin**: look up a stored pin and find similar pins (excluding itself).
  - **By query**: embed free-text and find the most similar pins.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

No `.env` setup required — `.env.example` kept as a placeholder in case you swap in a hosted embeddings provider later.

## Build the index

```bash
python scripts/ingest.py
```

This reads `data/pins.json`, embeds each pin, and persists the index to `chroma_db/`. Prints the number of pins indexed.

## Run the API

```bash
uvicorn app.main:app --reload
```

On startup, the app loads the persisted `chroma_db/` index if it exists, or builds it automatically if missing.

## Example requests

Recommend pins similar to a specific pin:

```bash
curl "http://localhost:8000/recommend/pin/p001?k=5"
```

Recommend pins matching a free-text query:

```bash
curl "http://localhost:8000/recommend/query?q=cozy%20living%20room&k=5"
```

Response shape:

```json
{
  "results": [
    {
      "pin_id": "p003",
      "title": "Small Space Reading Nook",
      "board": "Home Decor",
      "image_url": "/images/p003.jpg",
      "score": 0.42
    }
  ]
}
```

## Tests

Tests use a fake deterministic embedding function, so no API key or network access is required:

```bash
pytest tests/
```

## Project structure

```
├── data/
│   ├── pins.json           # mock pin dataset
│   └── images/             # real self-hosted photos, one per pin (served at /images)
├── app/
│   ├── models.py           # Pydantic schemas
│   ├── vectorstore.py      # build/load Chroma index
│   ├── recommender.py      # similarity search logic
│   └── main.py             # FastAPI app
├── scripts/ingest.py        # CLI to (re)build the index
└── tests/test_recommender.py
```
