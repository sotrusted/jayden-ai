#!/usr/bin/env python3
"""
Generate few-shot stub candidates by retrieving relevant items from spite_dump.json.

Inputs:
  - questions file (one question per line)
  - spite_dump.json (array of objects; tries common fields: title/description/text/display_name/...)

Retrieval modes:
  - TF-IDF (default): fast lexical search
  - Embeddings/FAISS (optional): --mode embed with prebuilt or on-the-fly embeddings

Outputs:
  - A .txt file containing, for each question, top-k results with title, summary, and display name
    ready for manual curation into few-shots.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import List, Dict, Any, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


COMMON_TITLE_KEYS = ["title",  "name",]
COMMON_TEXT_KEYS = ["content", ]
COMMON_DISPLAY_KEYS = ["display_name"]


def load_dump(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("spite_dump.json must be a JSON array of objects")
    return [d for d in data if isinstance(d, dict)]


def get_first(d: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        if k in d and isinstance(d[k], str) and d[k].strip():
            return d[k]
    return ""


def build_records(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build unified records from spite_dump rows.

    Expected shapes:
      - blog.post:  {"model": "blog.post", "fields": {"title", "content", "display_name", ...}}
      - blog.comment: {"model": "blog.comment", "fields": {"name", "content", ...}}
    Fallback: try common top-level keys if shape differs.
    """
    recs: List[Dict[str, str]] = []
    for r in rows:
        model = r.get("model", "")
        fields = r.get("fields", {}) if isinstance(r.get("fields"), dict) else {}

        title = ""
        text = ""
        display = ""

        if model == "blog.post" and fields:
            title = str(fields.get("title", "") or "").strip()
            text = str(fields.get("content", "") or "").strip()
            display = str(fields.get("display_name", "") or "").strip()
        elif model == "blog.comment" and fields:
            # Use commenter name as title; content as body
            name = str(fields.get("name", "") or "").strip()
            title = name or "Comment"
            text = str(fields.get("content", "") or "").strip()
            display = name
        else:
            # Fallback to common top-level keys if any
            title = get_first(r, COMMON_TITLE_KEYS) or ""
            text = get_first(r, COMMON_TEXT_KEYS) or ""
            display = get_first(r, COMMON_DISPLAY_KEYS) or ""
            if not title and "id" in r:
                title = f"Post {r['id']}"
            if not text and fields:
                # Try within fields if present but model not matched
                text = str(fields.get("content", "") or "")
                if not title:
                    title = str(fields.get("title", "") or fields.get("name", ""))

        # Skip empty rows with no useful text
        if not (title or text or display):
            continue

        # Summarize text lightly (first line ~ 200 chars)
        first_line = re.split(r"[\n]", text.strip())[0] if text else ""
        summary = first_line[:200]

        search_text = " ".join([title, text, display]).strip()
        recs.append({
            "title": title or "(no title)",
            "content": text or "",
            "display": display or "",
            "search_text": search_text,
        })
    return recs


def retrieve_tfidf(records: List[Dict[str, str]], questions: List[str], k: int) -> Dict[str, List[Tuple[int, float]]]:
    corpus = [r["search_text"] for r in records]
    vec = TfidfVectorizer(max_features=50000, stop_words="english")
    X = vec.fit_transform(corpus)
    Q = vec.transform(questions)
    sims = cosine_similarity(Q, X)
    out: Dict[str, List[Tuple[int, float]]] = {}
    for qi, q in enumerate(questions):
        scores = sims[qi]
        idxs = np.argsort(-scores)[:k]
        out[q] = [(int(i), float(scores[i])) for i in idxs]
    return out


def build_faiss(records: List[Dict[str, str]], embeddings_path: str) -> Tuple[Any, np.ndarray]:
    if SentenceTransformer is None or faiss is None:
        raise RuntimeError("Embeddings mode requires sentence-transformers and faiss installed")
    model = SentenceTransformer("all-mpnet-base-v2")
    texts = [r["search_text"] for r in records]
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    if embeddings_path:
        np.save(embeddings_path, emb)
    return index, emb


def load_faiss(records: List[Dict[str, str]], embeddings_path: str) -> Tuple[Any, np.ndarray]:
    if faiss is None:
        raise RuntimeError("FAISS not available")
    emb = np.load(embeddings_path)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    return index, emb


def retrieve_embed(records: List[Dict[str, str]], questions: List[str], k: int, index: Any) -> Dict[str, List[Tuple[int, float]]]:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not available")
    model = SentenceTransformer("all-mpnet-base-v2")
    Q = model.encode(questions, convert_to_numpy=True, show_progress_bar=False)
    out: Dict[str, List[Tuple[int, float]]] = {}
    for qi, q in enumerate(questions):
        D, I = index.search(Q[qi:qi+1], k)
        scores = [1.0/(1.0+float(d)) for d in D[0]]
        out[q] = [(int(i), float(s)) for i, s in zip(I[0], scores)]
    return out


def write_stubs(path: str, records: List[Dict[str, str]], results: Dict[str, List[Tuple[int, float]]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for q, hits in results.items():
            f.write(f"Question: {q}\n")
            for rank, (idx, score) in enumerate(hits, start=1):
                r = records[idx]
                f.write(f"[{rank}] title: {r['title']}\n")
                f.write(f"    display: {r['display']}\n")
                f.write(f"    score: {score:.3f}\n")
                f.write(f"    content: {r['content']}\n")
            f.write("\n")


def read_questions(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines


def main():
    ap = argparse.ArgumentParser(description="Generate few-shot stubs from spite_dump.json")
    ap.add_argument("--dump", default="spite_dump.json")
    ap.add_argument("--questions", required=True, help="path to text file with one question per line")
    ap.add_argument("--out", default="fewshot_stubs.txt")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--mode", choices=["tfidf", "embed"], default="tfidf")
    ap.add_argument("--embeddings-path", default="spite_dump_embeddings.npy")
    ap.add_argument("--build-embeddings", action="store_true")
    args = ap.parse_args()

    print("Loading dump...")
    rows = load_dump(args.dump)
    print(f"Loaded {len(rows)} rows")
    records = build_records(rows)

    print("Loading questions...")
    questions = read_questions(args.questions)
    print(f"Loaded {len(questions)} questions")

    if args.mode == "tfidf":
        print("Retrieving with TF-IDF...")
        results = retrieve_tfidf(records, questions, k=args.k)
    else:
        if os.path.exists(args.embeddings_path) and not args.build_embeddings:
            print("Loading existing embeddings...")
            index, _ = load_faiss(records, args.embeddings_path)
        else:
            print("Building embeddings (this may take a while)...")
            index, _ = build_faiss(records, args.embeddings_path)
        print("Retrieving with embeddings/FAISS...")
        results = retrieve_embed(records, questions, k=args.k, index=index)

    print(f"Writing stubs to {args.out}...")
    write_stubs(args.out, records, results)
    print("Done.")


if __name__ == "__main__":
    main()


