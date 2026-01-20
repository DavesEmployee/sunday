#!/bin/sh
set -e

if [ ! -d "data/json" ]; then
  echo "data/json not found. Skipping index build."
  exec "$@"
fi

if [ ! -d "data/vector/products_text.lance" ]; then
  TEXT_BACKEND="${TEXT_BACKEND:-hashing}"
  echo "Building vector index (backend: ${TEXT_BACKEND})..."
  uv run -- python build_product_index.py --reset --text-backend "${TEXT_BACKEND}"
else
  echo "Vector index already present."
fi

exec "$@"
