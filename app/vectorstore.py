import json
import os
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "pins.json"
PERSIST_DIR = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "pins"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_pins(data_path: Path = DATA_PATH) -> List[dict]:
    with open(data_path, "r") as f:
        return json.load(f)


def pin_to_document(pin: dict) -> Document:
    content = f"{pin['title']}. {pin['description']}. Tags: {', '.join(pin['tags'])}"
    metadata = {
        "id": pin["id"],
        "title": pin["title"],
        "board": pin["board"],
        "image_url": pin["image_url"],
        "tags": ", ".join(pin["tags"]),
    }
    return Document(page_content=content, metadata=metadata)


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore(
    data_path: Path = DATA_PATH,
    persist_directory: str = PERSIST_DIR,
) -> Chroma:
    """Build a fresh Chroma index from pins.json and persist it to disk."""
    pins = load_pins(data_path)
    documents = [pin_to_document(pin) for pin in pins]
    ids = [pin["id"] for pin in pins]

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=persist_directory,
    )
    # Reset any existing collection so re-ingesting is idempotent.
    existing = store.get()
    if existing["ids"]:
        store.delete(ids=existing["ids"])

    store.add_documents(documents=documents, ids=ids)
    return store


def load_vectorstore(persist_directory: str = PERSIST_DIR) -> Chroma:
    """Load an existing persisted Chroma index, building it if missing."""
    if not os.path.isdir(persist_directory) or not os.listdir(persist_directory):
        return build_vectorstore(persist_directory=persist_directory)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=persist_directory,
    )
