#!/usr/bin/env python3
"""Rebuild the Chroma vector index from data/pins.json."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from app.vectorstore import DATA_PATH, PERSIST_DIR, build_vectorstore, load_pins


def main():
    load_dotenv()
    pins = load_pins()
    print(f"Loaded {len(pins)} pins from {DATA_PATH}")

    print(f"Building Chroma index at {PERSIST_DIR} ...")
    build_vectorstore()
    print(f"Indexed {len(pins)} pins into Chroma collection 'pins'.")


if __name__ == "__main__":
    main()
