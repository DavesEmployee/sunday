from __future__ import annotations

import json
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import lancedb
from rank_bm25 import BM25Okapi


DATA_DIR = Path("data")
JSON_DIR = DATA_DIR / "json"
VECTOR_DIR = DATA_DIR / "vector"
EMBEDDER_META_PATH = VECTOR_DIR / "text_embedder.json"


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _doc_text(doc: Dict[str, object]) -> str:
    parts: List[str] = []
    for key in ("product_name", "description"):
        val = doc.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    sections = doc.get("sections")
    if isinstance(sections, dict):
        for k, v in sections.items():
            if isinstance(v, str) and v.strip():
                parts.append(f"{k.replace('_', ' ')}: {v.strip()}")
    return "\n\n".join(parts)


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = sum(v * v for v in vec) ** 0.5
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _clean_metadata(meta: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in meta.items() if v is not None}


def _token_overlap_score(query_tokens: List[str], text: str) -> float:
    if not text:
        return 0.0
    tokens = set(_tokenize(text))
    if not tokens:
        return 0.0
    overlap = sum(1 for t in query_tokens if t in tokens)
    return overlap / max(len(query_tokens), 1)


def _preload_torch_dlls() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        from importlib.util import find_spec

        spec = find_spec("torch")
        if not spec or not spec.origin:
            return
        dll_path = Path(spec.origin).parent / "lib" / "c10.dll"
        if dll_path.exists():
            ctypes.CDLL(str(dll_path))
    except Exception:
        pass


def _get_lancedb(vector_dir: Path) -> lancedb.DBConnection:
    return lancedb.connect(str(vector_dir))


class TextEmbedder:
    def __init__(
        self,
        model_name: str,
        fallback_dim: int = 2048,
        force_backend: Optional[str] = None,
    ) -> None:
        self.backend = force_backend or "hashing"
        self.model_name = model_name
        self.model = None
        self.dim = fallback_dim
        if self.backend in {"sentence_transformers", "fastembed"}:
            self._init_backend(prefer=self.backend)

    def _init_backend(self, prefer: str) -> None:
        if prefer == "sentence_transformers":
            try:
                _preload_torch_dlls()
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(self.model_name)
                _ = model.encode(["test"], normalize_embeddings=True)
                self.model = model
                self.backend = "sentence_transformers"
                return
            except Exception:
                self.model = None
                self.backend = "hashing"
                return
        if prefer == "fastembed":
            try:
                from fastembed import TextEmbedding

                model = TextEmbedding(self.model_name)
                _ = list(model.embed(["test"]))
                self.model = model
                self.backend = "fastembed"
                return
            except Exception:
                self.model = None
                self.backend = "hashing"
                return

    def _hash_embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for token in _tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest, "big") % self.dim
            vec[idx] += 1.0
        return _l2_normalize(vec)

    def encode(self, texts: List[str]) -> List[List[float]]:
        if self.backend == "sentence_transformers":
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            return [e.tolist() for e in embeddings]
        if self.backend == "fastembed":
            try:
                embeddings = list(self.model.embed(texts))
                return [_l2_normalize(e.tolist()) for e in embeddings]
            except Exception:
                self.backend = "hashing"
        return [self._hash_embed(text) for text in texts]


class ImageEmbedder:
    def __init__(self, model_name: Optional[str]) -> None:
        self.available = False
        self.model = None
        if not model_name:
            return
        try:
            _preload_torch_dlls()
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.available = True
        except Exception:
            self.available = False
            self.model = None

    def encode_path(self, path: str) -> Optional[List[float]]:
        if not self.available:
            return None
        emb = self.model.encode(path, normalize_embeddings=True)
        return emb.tolist()


def build_indexes(
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    image_model_name: Optional[str] = None,
    reset: bool = False,
    text_backend: str = "hashing",
) -> None:
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(JSON_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {JSON_DIR}")

    docs: List[Dict[str, object]] = []

    for path in json_files:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        if not isinstance(data, dict):
            continue
        slug = data.get("product_slug") or path.stem
        if not isinstance(slug, str):
            slug = path.stem
        product_url = data.get("product_url") or data.get("url")
        doc = {
            "slug": slug,
            "url": data.get("url"),
            "product_url": product_url,
            "product_name": data.get("product_name"),
            "description": data.get("description"),
            "sections": data.get("sections") or {},
            "image_url": data.get("image_url"),
            "image_path": data.get("image_path"),
        }
        text = _doc_text(doc)
        doc["text"] = text
        docs.append(doc)

    if reset:
        db = _get_lancedb(VECTOR_DIR)
        for name in ("products_text", "products_image"):
            try:
                db.drop_table(name)
            except Exception:
                pass

    text_model = TextEmbedder(text_model_name, force_backend=text_backend)
    image_model = ImageEmbedder(image_model_name)
    embedder_meta = {
        "backend": text_model.backend,
        "model_name": text_model.model_name,
        "dim": text_model.dim,
    }
    EMBEDDER_META_PATH.write_text(json.dumps(embedder_meta, indent=2), encoding="utf-8")

    db = _get_lancedb(VECTOR_DIR)

    embeddings = text_model.encode([d["text"] for d in docs])
    text_rows = []
    for doc, emb in zip(docs, embeddings):
        text_rows.append(
            _clean_metadata(
                {
                    "slug": doc["slug"],
                    "product_name": doc.get("product_name"),
                    "image_path": doc.get("image_path"),
                    "image_url": doc.get("image_url"),
                    "url": doc.get("url"),
                    "product_url": doc.get("product_url"),
                    "text": doc.get("text"),
                    "vector": emb,
                }
            )
        )
    db.create_table("products_text", text_rows, mode="overwrite")

    image_ids: List[str] = []
    image_embeddings: List[List[float]] = []
    image_metadatas: List[Dict[str, object]] = []
    for d in docs:
        image_path = d.get("image_path")
        if not isinstance(image_path, str):
            continue
        img_file = Path(image_path)
        if not img_file.exists():
            continue
        emb = image_model.encode_path(str(img_file))
        if emb is None:
            continue
        image_ids.append(d["slug"])
        image_embeddings.append(emb)
        image_metadatas.append(
            _clean_metadata(
                {
                    "slug": d["slug"],
                    "image_path": image_path,
                    "image_url": d.get("image_url"),
                    "url": d.get("url"),
                }
            )
        )

    if image_ids:
        image_rows = []
        for slug, emb, meta in zip(image_ids, image_embeddings, image_metadatas):
            row = dict(meta)
            row["slug"] = slug
            row["vector"] = emb
            image_rows.append(row)
        db.create_table("products_image", image_rows, mode="overwrite")


class HybridRetriever:
    def __init__(
        self,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_dir: Path = VECTOR_DIR,
    ) -> None:
        json_files = sorted(JSON_DIR.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {JSON_DIR}")

        docs: List[Dict[str, object]] = []
        tokenized: List[List[str]] = []
        for path in json_files:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if not isinstance(data, dict):
                continue
            slug = data.get("product_slug") or path.stem
            if not isinstance(slug, str):
                slug = path.stem
            product_url = data.get("product_url") or data.get("url")
            doc = {
                "slug": slug,
                "url": data.get("url"),
                "product_url": product_url,
                "product_name": data.get("product_name"),
                "description": data.get("description"),
                "sections": data.get("sections") or {},
                "image_url": data.get("image_url"),
                "image_path": data.get("image_path"),
            }
            text = _doc_text(doc)
            doc["text"] = text
            docs.append(doc)
            tokenized.append(_tokenize(text))

        self.docs = docs
        self.doc_map = {d["slug"]: d for d in self.docs}
        self.bm25 = BM25Okapi(tokenized)
        meta = None
        if EMBEDDER_META_PATH.exists():
            try:
                meta = json.loads(EMBEDDER_META_PATH.read_text(encoding="utf-8"))
            except Exception:
                meta = None
        fallback_dim = 2048
        model_name = text_model_name
        force_backend = "hashing"
        if isinstance(meta, dict):
            fallback_dim = int(meta.get("dim") or fallback_dim)
            model_name = str(meta.get("model_name") or model_name)
            force_backend = meta.get("backend")
        self.text_model = TextEmbedder(
            model_name,
            fallback_dim=fallback_dim,
            force_backend=force_backend,
        )
        self.db = _get_lancedb(vector_dir)
        self.text_table = self.db.open_table("products_text")

    def query(self, text: str, top_k: int = 3, rerank: bool = True) -> List[Dict[str, object]]:
        tokens = _tokenize(text)
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_rank = sorted(
            enumerate(bm25_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        text_emb = self.text_model.encode([text])
        vec_results = (
            self.text_table.search(text_emb[0])
            .limit(min(25, len(self.docs)))
            .to_list()
        )
        vec_ids = [r.get("slug") for r in vec_results if r.get("slug")]

        rrf_scores: Dict[str, float] = {}
        for rank, (idx, _) in enumerate(bm25_rank, start=1):
            slug = self.docs[idx]["slug"]
            rrf_scores[slug] = rrf_scores.get(slug, 0.0) + 1.0 / (60 + rank)

        for rank, slug in enumerate(vec_ids, start=1):
            rrf_scores[slug] = rrf_scores.get(slug, 0.0) + 1.0 / (60 + rank)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        if not rerank:
            results: List[Dict[str, object]] = []
            for slug, _score in ranked[:top_k]:
                doc = self.doc_map.get(slug)
                if not doc:
                    continue
                results.append(doc)
            return results

        candidates = ranked[: min(25, len(ranked))]
        rescored = []
        for slug, base_score in candidates:
            doc = self.doc_map.get(slug)
            if not doc:
                continue
            name_score = _token_overlap_score(tokens, str(doc.get("product_name") or ""))
            desc_score = _token_overlap_score(tokens, str(doc.get("description") or ""))
            sections = doc.get("sections") or {}
            section_text = " ".join(v for v in sections.values() if isinstance(v, str))
            section_score = _token_overlap_score(tokens, section_text)
            score = base_score + (0.35 * name_score) + (0.15 * desc_score) + (0.2 * section_score)
            rescored.append((score, doc))

        rescored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _score, doc in rescored[:top_k]]
