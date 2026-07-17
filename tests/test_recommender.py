import hashlib
import json

import pytest
from langchain_chroma import Chroma

from app.recommender import PinNotFoundError, PinRecommender
from app.vectorstore import pin_to_document

SAMPLE_PINS = [
    {
        "id": "t001",
        "title": "Cozy Living Room Ideas",
        "description": "Warm neutral tones, layered rugs, and soft lighting for a cozy living room.",
        "tags": ["living room", "cozy", "interior design"],
        "board": "Home Decor",
        "image_url": "https://example.com/t001.jpg",
    },
    {
        "id": "t002",
        "title": "Minimalist Bedroom Design",
        "description": "Clean lines and simple furniture for a calm minimalist bedroom.",
        "tags": ["bedroom", "minimalist", "interior design"],
        "board": "Home Decor",
        "image_url": "https://example.com/t002.jpg",
    },
    {
        "id": "t003",
        "title": "One Pot Pasta Recipe",
        "description": "Quick and easy creamy garlic pasta made in a single pot.",
        "tags": ["pasta", "dinner", "easy recipe"],
        "board": "Recipes",
        "image_url": "https://example.com/t003.jpg",
    },
    {
        "id": "t004",
        "title": "Backpacking in Patagonia",
        "description": "A hiking itinerary through Torres del Paine national park.",
        "tags": ["travel", "hiking", "adventure"],
        "board": "Travel",
        "image_url": "https://example.com/t004.jpg",
    },
]


class FakeEmbeddings:
    """Deterministic fake embeddings so tests run without network access or an API key."""

    def _vector(self, text: str):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:16]]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def store():
    documents = [pin_to_document(pin) for pin in SAMPLE_PINS]
    ids = [pin["id"] for pin in SAMPLE_PINS]
    chroma_store = Chroma(
        collection_name="test_pins",
        embedding_function=FakeEmbeddings(),
    )
    chroma_store.add_documents(documents=documents, ids=ids)
    return chroma_store


@pytest.fixture
def recommender(store):
    return PinRecommender(store)


def test_pins_json_loads():
    from app.vectorstore import DATA_PATH

    with open(DATA_PATH) as f:
        pins = json.load(f)
    assert len(pins) >= 30
    assert {"id", "title", "description", "tags", "board", "image_url"} <= set(pins[0].keys())


def test_similarity_search_by_query_returns_expected_shape(recommender):
    results = recommender.similarity_search_by_query("cozy home decor ideas", k=2)
    assert len(results) <= 2
    for r in results:
        assert r.pin_id
        assert r.title
        assert r.board
        assert isinstance(r.score, float)


def test_similarity_search_by_pin_id_excludes_source_pin(recommender):
    results = recommender.similarity_search_by_pin_id("t001", k=3)
    assert all(r.pin_id != "t001" for r in results)
    assert len(results) <= 3


def test_similarity_search_by_pin_id_missing_raises(recommender):
    with pytest.raises(PinNotFoundError):
        recommender.similarity_search_by_pin_id("does-not-exist", k=3)
