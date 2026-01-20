#!/usr/bin/env python3
"""
Build text + image indexes for product RAG.

Writes:
- data/vector/text_embedder.json
- data/vector/products_text.lance
"""

import argparse

from src.retriever import build_indexes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="Reset existing vector collections")
    ap.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument(
        "--image-model",
        default="",
        help="Image embedding model (leave empty to skip image embeddings)",
    )
    ap.add_argument(
        "--text-backend",
        default="hashing",
        choices=["hashing", "sentence_transformers", "fastembed"],
        help="Embedding backend for text vectors",
    )
    args = ap.parse_args()

    build_indexes(
        text_model_name=args.text_model,
        image_model_name=args.image_model,
        reset=args.reset,
        text_backend=args.text_backend,
    )


if __name__ == "__main__":
    main()
