#!/usr/bin/env python3
"""
Single-agent CLI for the RAG chatbot.
"""

import sys
from pathlib import Path


def _bootstrap_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main() -> None:
    _bootstrap_repo_root()
    from src.chat_runtime import run_cli

    run_cli()


if __name__ == "__main__":
    main()
