"""Utility functions for embedding Spite posts.

This module keeps the original CLI behaviour while exposing helpers so other
parts of the codebase (e.g. the DataManager) can reuse the exact same logic
for building the corpus, metadata, and embeddings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_posts(path: Path | str) -> list:
    """Load the raw JSON dump of posts/comments."""

    path = Path(path)
    with path.open() as handle:
        print(f"Loading {path}...")
        return json.load(handle)


def flatten_posts(posts: Iterable[dict]) -> Tuple[List[str], List[dict]]:
    """Flatten posts/comments into corpus texts plus metadata entries."""

    corpus: List[str] = []
    metadata: List[dict] = []

    for obj in tqdm(list(posts)):
        obj_model = obj["model"]
        obj_id = obj["pk"]

        if obj_model == "blog.post":
            post = obj["fields"]
            data = post["title"]
            post_content = post["content"]
            if post_content:
                data += f"\n{post_content}"

            metadata.append(
                {
                    "type": "post",
                    "id": obj_id,
                    "title": post["title"],
                    "author": post.get("author"),
                    "anon_uuid": post.get("anon_uuid"),
                    "display_name": post.get("display_name", ""),
                }
            )

        elif obj_model == "blog.comment":
            comment = obj["fields"]
            name = comment["name"]
            content = comment["content"]
            data = f"{name}\n{content}" if name else content

            metadata.append(
                {
                    "type": "comment",
                    "id": obj_id,
                    "name": comment.get("name", ""),
                    "content": comment.get("content", ""),
                    "post_id": comment.get("post"),
                }
            )
        else:
            # Preserve entries we don't recognise, in case new types appear later.
            fields = obj.get("fields", {})
            text = fields.get("content") or fields.get("text") or ""
            data = text
            metadata.append({"type": obj_model, "id": obj_id})

        corpus.append(data)

    return corpus, metadata


def describe_corpus(corpus: Sequence[str]) -> None:
    """Emit a quick shape/preview for parity with the original script."""

    corpus_df = pd.DataFrame(corpus)
    print(corpus_df.shape)
    print(corpus_df.head())


def embed_corpus(
    corpus: Sequence[str],
    *,
    model: SentenceTransformer | None = None,
    model_name: str = "all-mpnet-base-v2",
    batch_size: int = 1000,
    show_progress: bool = True,
) -> Tuple[np.ndarray, SentenceTransformer]:
    """Generate sentence embeddings with batch progress logging."""

    if model is None:
        model = SentenceTransformer(model_name)
    print(model.device)

    print(f"Embedding {len(corpus)} entries...")
    batches = []

    for i in range(0, len(corpus), batch_size):
        batch = corpus[i : i + batch_size]
        total_batches = (len(corpus) + batch_size - 1) // batch_size
        print(f"Processing batch {i // batch_size + 1}/{total_batches} ({len(batch)} entries)")

        try:
            batch_embeddings = model.encode(batch, show_progress_bar=show_progress)
            batches.append(batch_embeddings)
            print(f"Batch {i // batch_size + 1} completed. Shape: {batch_embeddings.shape}")
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Error in batch {i // batch_size + 1}: {exc}")
            raise

    print("Combining batches...")
    embeddings = np.vstack(batches) if batches else np.empty((0, model.get_sentence_embedding_dimension()))
    print(f"Final embeddings shape: {embeddings.shape}")
    if embeddings.size:
        print(f"Sample embedding (first 5 values): {embeddings[0][:5]}")

    return embeddings, model


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Create an in-memory FAISS index mirroring the CLI script."""

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_outputs(
    corpus: Sequence[str],
    metadata: Sequence[dict],
    embeddings: np.ndarray,
    *,
    corpus_path: Path | str,
    metadata_path: Path | str,
    embeddings_path: Path | str,
) -> None:
    """Persist the corpus, metadata, and embeddings to disk."""

    corpus_path = Path(corpus_path)
    metadata_path = Path(metadata_path)
    embeddings_path = Path(embeddings_path)

    np.save(embeddings_path, embeddings)
    with corpus_path.open("w+") as corpus_file:
        json.dump(list(corpus), corpus_file)
    with metadata_path.open("w+") as metadata_file:
        json.dump(list(metadata), metadata_file)

    print(f"Saved {len(corpus)} entries with metadata")
    print(f"Posts: {len([m for m in metadata if m.get('type') == 'post'])}")
    print(f"Comments: {len([m for m in metadata if m.get('type') == 'comment'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/spite_dump.json")
    parser.add_argument("--corpus_file", type=str, default="data/spite_corpus.json")
    parser.add_argument("--metadata_file", type=str, default="data/spite_metadata.json")
    parser.add_argument("--embeddings_file", type=str, default="data/spite_embeddings.npy")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--model_name", type=str, default="all-mpnet-base-v2")
    args = parser.parse_args()

    posts = load_posts(args.input_file)
    corpus, metadata = flatten_posts(posts)
    describe_corpus(corpus)
    embeddings, _ = embed_corpus(corpus, batch_size=args.batch_size, model_name=args.model_name)
    build_faiss_index(embeddings)
    save_outputs(
        corpus,
        metadata,
        embeddings,
        corpus_path=args.corpus_file,
        metadata_path=args.metadata_file,
        embeddings_path=args.embeddings_file,
    )


if __name__ == "__main__":
    main()
