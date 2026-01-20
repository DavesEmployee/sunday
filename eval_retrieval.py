#!/usr/bin/env python3
"""
Simple retrieval eval harness for Sunday product RAG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from src.retriever import HybridRetriever


def load_eval_cases(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Eval cases must be a list of objects.")
    return data


def compute_metrics(results: List[Dict[str, object]], expected_slug: str) -> Dict[str, int]:
    slugs = [r.get("slug") for r in results if r.get("slug")]
    return {
        "hit_at_1": int(expected_slug in slugs[:1]),
        "hit_at_3": int(expected_slug in slugs[:3]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-file", default="data/eval/retrieval_cases.json")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    ap.add_argument("--compare-rerank", action="store_true", help="Compare base vs reranked results")
    args = ap.parse_args()

    eval_path = Path(args.eval_file)
    cases = load_eval_cases(eval_path)
    retriever = HybridRetriever()

    total = 0
    hit_at_1 = 0
    hit_at_3 = 0
    base_hit_at_1 = 0
    base_hit_at_3 = 0

    for case in cases:
        query = case.get("query")
        expected = case.get("expected_slug")
        if not isinstance(query, str) or not isinstance(expected, str):
            continue
        if args.compare_rerank:
            base_results = retriever.query(query, top_k=args.top_k, rerank=False)
            base_metrics = compute_metrics(base_results, expected)
            base_hit_at_1 += base_metrics["hit_at_1"]
            base_hit_at_3 += base_metrics["hit_at_3"]
        results = retriever.query(query, top_k=args.top_k, rerank=not args.no_rerank)
        metrics = compute_metrics(results, expected)
        hit_at_1 += metrics["hit_at_1"]
        hit_at_3 += metrics["hit_at_3"]
        total += 1
        top_slugs = [r.get("slug") for r in results]
        print(f"Q: {query}")
        print(f"Expected: {expected}")
        print(f"Top: {top_slugs}\n")

    if total == 0:
        print("No valid eval cases found.")
        return

    print("Results")
    print(f"Total: {total}")
    if args.compare_rerank:
        print(f"Base Hit@1: {base_hit_at_1}/{total} = {base_hit_at_1 / total:.2f}")
        print(f"Base Hit@3: {base_hit_at_3}/{total} = {base_hit_at_3 / total:.2f}")
    print(f"Hit@1: {hit_at_1}/{total} = {hit_at_1 / total:.2f}")
    print(f"Hit@3: {hit_at_3}/{total} = {hit_at_3 / total:.2f}")


if __name__ == "__main__":
    main()
