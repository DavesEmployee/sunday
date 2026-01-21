#!/usr/bin/env python3
"""
Multi-agent orchestration demo.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.multiagent import run_multiagent_session


if __name__ == "__main__":
    run_multiagent_session()
