#!/usr/bin/env python3
"""
Generate a Spite lorebook and optional few-shots from the existing corpus.

Outputs:
- spite_lorebook.json
- spite_fewshots.txt (optional)

This script uses light-weight statistics, phrase mining, and optional
transformers sentiment to summarize style and recurring memes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
from collections import Counter
from typing import List, Dict, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# Optional RAG deps
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

try:
    from transformers import pipeline
except Exception:  # transformers optional
    pipeline = None

DEFAULT_CORPUS_PATH = "data/spite_corpus.json"
DEFAULT_LOREBOOK_PATH = "data/spite_lorebook.json"
DEFAULT_FEWSHOTS_PATH = "data/spite_fewshots.txt"

# Heuristic slang/alias seeds
SLANG_SEED = {
    "schizo", "bit", "canon", "admin", "dimes", "dimes square", "bucharest",
    "spite", "nazbol", "vack", "vegas", "reading", "transylvania", "goon",
}
ALIASES_SEED = {
    "jayden": ["jaden", "child of prophecy"],
}

# Extra stopwords & filters for lore/phrases/entities
CUSTOM_STOPWORDS = {
    "yeah", "lol", "lmao", "ok", "okay", "like", "just", "really", "think",
    "know", "want", "people", "good", "bad", "new", "yall", "gonna", "im",
    "dont", "didnt", "cant", "won't", "aint", "ur", "u", "idk",
}
MONTHS = {"january","february","march","april","may","june","july","august","september","october","november","december"}
DAYS = {"monday","tuesday","wednesday","thursday","friday","saturday","sunday"}
STOPWORDS = set(ENGLISH_STOP_WORDS) | CUSTOM_STOPWORDS | MONTHS | DAYS

# Banlist for junk few-shot terms
BAN_TERMS = {
    "big", "small", "who", "what", "where", "when", "why", "how",
    "schlong",  # too noisy without context
}

# Helper patterns
VERB_HINT = re.compile(r"\b(is|was|are|were|has|have|does|did|made|became|called|means|runs|posts)\b", re.I)


def load_corpus(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Corpus must be a JSON list of strings")
        return [str(x) for x in data]


def is_repetitive_spam(text: str, *, top_token_ratio: float = 0.35, unique_ratio: float = 0.45,
                       bigram_ratio: float = 0.25) -> bool:
    """Detect extremely repetitive posts that drown out organic topics."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 12:
        return False

    total = len(words)
    unigram_counts = Counter(words)
    most_common_unigram = unigram_counts.most_common(1)[0][1] / total
    unique_unigram_ratio = len(unigram_counts) / total
    if most_common_unigram >= top_token_ratio and unique_unigram_ratio <= unique_ratio:
        return True

    # Catch repeated phrases like "peter vack" x N
    bigram_total = max(total - 1, 1)
    bigram_counts = Counter(zip(words, words[1:]))
    if bigram_counts:
        top_bigram = bigram_counts.most_common(1)[0][1] / bigram_total
        if top_bigram >= bigram_ratio:
            return True

    return False


def filter_spammy_docs(corpus: List[str]) -> Tuple[List[str], int]:
    filtered: List[str] = []
    removed = 0
    for doc in corpus:
        if is_repetitive_spam(doc):
            removed += 1
            continue
        filtered.append(doc)
    return filtered, removed


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def simhash_signature(tokens: List[str], bits: int = 64) -> int:
    if not tokens:
        return 0
    vector = [0] * bits
    counts = Counter(tokens)
    for token, weight in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "big")
        for i in range(bits):
            mask = 1 << i
            if h & mask:
                vector[i] += weight
            else:
                vector[i] -= weight
    fingerprint = 0
    for i, value in enumerate(vector):
        if value > 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def dedupe_near_duplicates(corpus: List[str], *, threshold: int = 8) -> Tuple[List[str], int]:
    """Drop posts whose SimHash is within the given Hamming distance of an earlier post."""
    if threshold <= 0:
        return corpus, 0

    band_masks = []
    band_bits = 16
    for offset in range(0, 64, band_bits):
        mask = ((1 << band_bits) - 1) << offset
        band_masks.append((mask, offset))

    buckets: Dict[Tuple[int, int], List[int]] = {}
    fingerprints: List[int] = []
    kept: List[str] = []
    removed = 0

    for doc in corpus:
        tokens = tokenize_words(doc)
        if not tokens:
            kept.append(doc)
            fingerprints.append(0)
            continue

        fp = simhash_signature(tokens)
        is_duplicate = False
        candidates: List[int] = []
        for band_idx, (mask, shift) in enumerate(band_masks):
            key = ((fp & mask) >> shift, band_idx)
            if key in buckets:
                candidates.extend(buckets[key])

        for candidate_idx in candidates:
            if hamming_distance(fp, fingerprints[candidate_idx]) <= threshold:
                removed += 1
                is_duplicate = True
                break

        if is_duplicate:
            continue

        idx = len(fingerprints)
        fingerprints.append(fp)
        kept.append(doc)
        for band_idx, (mask, shift) in enumerate(band_masks):
            key = ((fp & mask) >> shift, band_idx)
            buckets.setdefault(key, []).append(idx)

    return kept, removed


def sentence_lengths(text: str) -> List[int]:
    parts = re.split(r"[.!?\n]+", text)
    lengths = [len(p.strip().split()) for p in parts if p and p.strip()]
    return lengths or [len(text.split())]


def basic_stats(corpus: List[str], sample_size: int) -> Dict:
    random.seed(42)
    idxs = list(range(len(corpus)))
    random.shuffle(idxs)
    idxs = idxs[: min(sample_size, len(idxs))]

    all_sent_lens: List[int] = []
    styles: Counter = Counter()

    for i in idxs:
        t = corpus[i]
        all_sent_lens.extend(sentence_lengths(t))

        low = t.lower()
        if "?" in t:
            styles["questioning"] += 1
        if "!" in t:
            styles["emphatic"] += 1
        if len(low.split()) < 20:
            styles["brief"] += 1
        if any(w in low for w in ("actually", "literally")):
            styles["ironic"] += 1

    avg_len = statistics.mean(all_sent_lens) if all_sent_lens else 0.0
    std_len = statistics.pstdev(all_sent_lens) if len(all_sent_lens) > 1 else 0.0
    return {
        "avg_sentence_length": float(avg_len),
        "sentence_std": float(std_len),
        "response_styles": dict(styles),
        "sampled_docs": len(idxs),
    }


def compute_sentiment(corpus: List[str], sample_size: int) -> Dict:
    if pipeline is None:
        return {"sentiment_mean": 0.0, "sentiment_std": 0.0, "note": "transformers pipeline unavailable"}
    n = min(sample_size, len(corpus))
    texts = corpus[:n]
    sent = pipeline("sentiment-analysis")
    scores = []
    for t in texts:
        try:
            out = sent(t[:512])[0]
            s = out.get("score", 0.5)
            s = s if out.get("label", "POSITIVE") == "POSITIVE" else -s
            scores.append(float(s))
        except Exception:
            continue
    if not scores:
        return {"sentiment_mean": 0.0, "sentiment_std": 0.0}
    return {
        "sentiment_mean": float(statistics.mean(scores)),
        "sentiment_std": float(statistics.pstdev(scores) if len(scores) > 1 else 0.0),
    }


def mine_phrases(
    corpus: List[str],
    max_features: int = 5000,
    min_df: int = 5,
    max_df: float = 0.1,
) -> List[Tuple[str, int]]:
    vectorizer = CountVectorizer(
        ngram_range=(2, 3),
        stop_words="english",
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )
    dtm = vectorizer.fit_transform(corpus)
    vocab = np.array(vectorizer.get_feature_names_out())
    counts = np.asarray(dtm.sum(axis=0)).ravel()
    raw = list(zip(vocab.tolist(), counts.tolist()))

    def keep_phrase(p: str) -> bool:
        if "http" in p or "www" in p:
            return False
        toks = [t for t in p.split() if t.isalpha()]
        if not toks:
            return False
        non_stop = sum(1 for t in toks if t.lower() not in STOPWORDS)
        return non_stop >= 1

    pairs = [(p, c) for p, c in raw if keep_phrase(p)]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def mine_entities(
    corpus: List[str],
    top_k: int = 100,
    max_doc_ratio: float = 0.1,
) -> List[Tuple[str, int]]:
    pattern = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
    freq_counter: Counter = Counter()
    doc_counter: Counter = Counter()
    common_block = {"I","We","You","The","And","But","Or","Not"}
    total_docs = len(corpus)
    for t in corpus:
        seen_in_doc = set()
        for m in pattern.findall(t):
            low = m.lower()
            if low in STOPWORDS or m in common_block or low in MONTHS or low in DAYS:
                continue
            if len(m) <= 2 or len(m) >= 40:
                continue
            if re.fullmatch(r"[A-Z][a-z]+", m) is None and re.fullmatch(r"[A-Z]+", m) is None:
                # skip weird tokens
                pass
            freq_counter[m] += 1
            seen_in_doc.add(m)
        for ent in seen_in_doc:
            doc_counter[ent] += 1

    if not total_docs:
        return []

    filtered: List[Tuple[str, int]] = []
    for ent, count in freq_counter.most_common():
        doc_ratio = doc_counter[ent] / total_docs
        if doc_ratio > max_doc_ratio:
            continue
        filtered.append((ent, count))
        if len(filtered) >= top_k:
            break
    return filtered


def build_aliases(entities: List[Tuple[str, int]]) -> Dict[str, List[str]]:
    aliases = dict(ALIASES_SEED)
    for ent, _ in entities[:50]:
        low = ent.lower()
        if low not in aliases:
            variants = set()
            variants.add(re.sub(r"[aeiou]", "", low))
            variants.add(low.replace("y", "i"))
            variants.add(low.replace("i", "y"))
            variants = {v for v in variants if v != low and len(v) > 2}
            if variants:
                aliases[low] = sorted(variants)
    return aliases


def select_memes(phrases: List[Tuple[str, int]], entities: List[Tuple[str, int]]) -> List[str]:
    top_entities = {e[0].lower() for e in entities[:50]}
    memes: List[str] = []
    for p, c in phrases:
        if any(tok in SLANG_SEED for tok in p.split()) or any(tok in top_entities for tok in p.split()):
            memes.append(p)
        if len(memes) >= 50:
            break
    return memes


def build_lorebook(stats: Dict, sentiment: Dict, entities: List[Tuple[str, int]], phrases: List[Tuple[str, int]]) -> Dict:
    aliases = build_aliases(entities)
    memes = select_memes(phrases, entities)
    return {
        "style": {
            "avg_sentence_length": stats.get("avg_sentence_length", 0.0),
            "sentence_std": stats.get("sentence_std", 0.0),
            "sentiment_mean": sentiment.get("sentiment_mean", 0.0),
            "sentiment_std": sentiment.get("sentiment_std", 0.0),
            "response_styles": stats.get("response_styles", {}),
        },
        "entities": entities[:100],
        "memes": memes,
        "aliases": aliases,
        "slang": sorted(SLANG_SEED),
        "notes": [
            "Use in-jokes naturally; do not explain them",
            "Prefer quoting phrases seen in memes/entities",
            "Lean into terminally online cadence; short, punchy lines unless asked for depth",
        ],
    }


def sample_context_lines(corpus: List[str], term: str, max_lines: int = 2) -> List[str]:
    term_low = term.lower()
    hits_raw = [t for t in corpus if term_low in t.lower()]
    # Prefer hits with some predicate/definition-like verb nearby
    scored: List[Tuple[float, str]] = []
    for h in hits_raw:
        low = h.lower()
        score = 0.0
        if VERB_HINT.search(low):
            score += 1.0
        # Avoid extremely short lines
        if len(low.split()) > 6:
            score += 0.2
        scored.append((score, h))
    scored.sort(key=lambda x: x[0], reverse=True)
    hits = [h for _, h in scored]
    random.shuffle(hits)
    out: List[str] = []
    for h in hits[:10]:
        sent = re.split(r"[.!?\n]", h)[0]
        words = sent.strip().split()
        out.append(" ".join(words[:30]).strip())
        if len(out) >= max_lines:
            break
    return out


def generate_fewshots(corpus: List[str], entities: List[Tuple[str, int]], memes: List[str], k: int) -> List[Tuple[str, str]]:
    # Filter terms: no banned, no ultra-generic, length >= 3
    raw_terms = [e[0] for e in entities[:50]] + memes[:80]
    terms = []
    seen = set()
    for t in raw_terms:
        base = t.strip()
        low = base.lower()
        if low in seen:
            continue
        if any(w in BAN_TERMS for w in low.split()):
            continue
        if len(low) < 3:
            continue
        seen.add(low)
        terms.append(base)
    random.shuffle(terms)
    pairs: List[Tuple[str, str]] = []
    for term in terms:
        if len(pairs) >= k:
            break
        # Better question form: prefer specific nouns/entities trigger
        if term.istitle() or len(term.split()) == 1:
            q = f"Who is {term}?"
        else:
            q = f"What is {term}?"
        ctx = sample_context_lines(corpus, term, max_lines=2)
        if not ctx:
            continue
        # Build a grounded answer; stitch two fragments with a minimal connective
        a = "; ".join(ctx)
        a = a.strip()
        if not a.endswith("."):
            a += "."
        pairs.append((q, a))
    return pairs


# ---------------- RAG-based few-shots ----------------

def init_rag(embeddings_path: str = "spite_embeddings.npy"):
    if not os.path.exists(embeddings_path):
        return None, None
    model = SentenceTransformer("all-mpnet-base-v2")
    emb = np.load(embeddings_path)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    return model, index


def rag_search(model, index, corpus: List[str], query: str, k: int = 8) -> List[Tuple[int, str, float]]:
    q = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q, k)
    out = []
    for idx, dist in zip(I[0], D[0]):
        sim = 1.0 / (1.0 + float(dist))
        out.append((int(idx), corpus[int(idx)], sim))
    return out


def generate_fewshots_rag(corpus: List[str], model, index, terms: List[str], k: int, per_term_k: int = 6) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for term in terms:
        if len(pairs) >= k:
            break
        low = term.lower()
        if any(w in BAN_TERMS for w in low.split()):
            continue
        q = f"Who is {term}?" if term.istitle() or len(term.split()) == 1 else f"What is {term}?"
        results = rag_search(model, index, corpus, term, k=per_term_k)
        if not results:
            continue
        # Number contexts and stitch a grounded answer with light citations
        numbered = []
        for n, (_, txt, sim) in enumerate(results, start=1):
            snippet = re.split(r"[\n]+", txt.strip())[0]
            snippet = snippet[:160]
            numbered.append(f"[{n}] {snippet}")
        cite_idxs = [1]
        if len(numbered) >= 2:
            cite_idxs.append(2)
        citations = "".join(f"[{i}]" for i in cite_idxs)
        # Prefer the most informative snippet (longer and includes a copula or action verb)
        def snippet_score(s: str) -> float:
            base = len(s.split()) / 10.0
            if VERB_HINT.search(s):
                base += 1.0
            return base
        best = max([re.split(r"[\n]", r[1])[0][:220] for r in results], key=snippet_score)
        answer = f"{best} {citations}".strip()
        pairs.append((q, answer))
    return pairs


def save_fewshots(pairs: List[Tuple[str, str]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for q, a in pairs:
            f.write(f"Q: {q}\n")
            f.write(f"A: {a}\n\n")


def generate_lore_from_corpus(
    corpus: List[str],
    *,
    sample: int = 4000,
    fewshots: int = 5,
    rag_fewshots: bool = False,
    embeddings_path: str = "spite_embeddings.npy",
    min_df: int = 5,
    max_df: float = 0.1,
    max_entity_doc_ratio: float = 0.1,
    skip_spam_filter: bool = False,
    near_dup_threshold: int = 8,
) -> Tuple[Dict, List[Tuple[str, str]], Dict[str, int]]:
    """Mirror the CLI pipeline but operate on an in-memory corpus."""

    work_corpus = list(corpus)
    stats_summary = {
        "initial_corpus_size": len(work_corpus),
        "spam_filtered": 0,
        "near_duplicates_removed": 0,
    }

    if not skip_spam_filter:
        print("Filtering repetitive spam...")
        filtered_corpus, removed = filter_spammy_docs(work_corpus)
        if removed:
            pct = removed / max(len(work_corpus), 1)
            print(f"Filtered {removed} spam-like posts ({pct:.1%}).")
        else:
            print("No spam-like posts filtered.")
        work_corpus = filtered_corpus
        stats_summary["spam_filtered"] = removed
        print(f"Corpus after filtering: {len(work_corpus)} posts")

    if near_dup_threshold > 0:
        print("Removing near-duplicate posts...")
        work_corpus, removed = dedupe_near_duplicates(work_corpus, threshold=near_dup_threshold)
        if removed:
            pct = removed / max(len(work_corpus) + removed, 1)
            print(f"Removed {removed} near-duplicate posts ({pct:.1%}).")
        else:
            print("No near-duplicates detected.")
        print(f"Corpus after dedupe: {len(work_corpus)} posts")
        stats_summary["near_duplicates_removed"] = removed

    print("Computing style stats...")
    stats = basic_stats(work_corpus, sample)
    print("Stats:", stats)

    print("Running sentiment analysis (sampled)...")
    sentiment = compute_sentiment(work_corpus, sample)
    print("Sentiment:", sentiment)

    print("Mining phrases...")
    phrases = mine_phrases(work_corpus, min_df=min_df, max_df=max_df)
    print(f"Top phrases: {phrases[:10]}")

    print("Mining entities...")
    entities = mine_entities(work_corpus, max_doc_ratio=max_entity_doc_ratio)
    print(f"Top entities: {entities[:10]}")

    print("Building lorebook...")
    lore = build_lorebook(stats, sentiment, entities, phrases)

    fewshot_pairs: List[Tuple[str, str]] = []
    if fewshots > 0:
        print("Generating few-shots...")
        if rag_fewshots:
            model, index = init_rag(embeddings_path)
            if model is None or index is None:
                print("RAG not available (missing embeddings). Falling back to heuristic few-shots.")
                memes_only = [p for p, _ in phrases]
                fewshot_pairs = generate_fewshots(work_corpus, entities, memes_only, fewshots)
            else:
                clean_entities = [e for e, _ in entities if e.lower() not in STOPWORDS and e.lower() not in BAN_TERMS][:50]
                clean_memes = [p for p, _ in phrases if all(w not in STOPWORDS and w not in BAN_TERMS for w in p.split())][:80]
                terms = clean_entities + clean_memes
                random.shuffle(terms)
                fewshot_pairs = generate_fewshots_rag(work_corpus, model, index, terms, fewshots)
        else:
            memes_only = [p for p, _ in phrases]
            fewshot_pairs = generate_fewshots(work_corpus, entities, memes_only, fewshots)
        print(f"Generated {len(fewshot_pairs)} few-shots")

    stats_summary["final_corpus_size"] = len(work_corpus)
    return lore, fewshot_pairs, stats_summary


def main():
    parser = argparse.ArgumentParser(description="Generate Spite lorebook and few-shots")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--out-lore", default=DEFAULT_LOREBOOK_PATH)
    parser.add_argument("--out-fewshots", default=DEFAULT_FEWSHOTS_PATH)
    parser.add_argument("--sample", type=int, default=4000, help="documents to sample for stats/sentiment")
    parser.add_argument("--fewshots", type=int, default=5, help="number of few-shot pairs to write")
    parser.add_argument("--rag-fewshots", action="store_true", help="use RAG to build few-shots")
    parser.add_argument("--embeddings", default="spite_embeddings.npy")
    parser.add_argument("--min-df", type=int, default=5, help="min document frequency for phrase mining")
    parser.add_argument("--max-df", type=float, default=0.1, help="max document frequency proportion for phrase mining")
    parser.add_argument("--max-entity-doc-ratio", type=float, default=0.1, help="drop entities seen in more than this share of posts")
    parser.add_argument("--skip-spam-filter", action="store_true", help="disable heuristic spam filtering for repetitive posts")
    parser.add_argument("--near-dup-threshold", type=int, default=8,
                        help="SimHash Hamming distance threshold for collapsing near-duplicate posts (0 disables)")
    args = parser.parse_args()

    print("Loading corpus...")
    corpus = load_corpus(args.corpus)
    print(f"Loaded {len(corpus)} posts")

    if not args.skip_spam_filter:
        print("Filtering repetitive spam...")
        filtered_corpus, removed = filter_spammy_docs(corpus)
        if removed:
            pct = removed / max(len(corpus), 1)
            print(f"Filtered {removed} spam-like posts ({pct:.1%}).")
        else:
            print("No spam-like posts filtered.")
        corpus = filtered_corpus
        print(f"Corpus after filtering: {len(corpus)} posts")

    if args.near_dup_threshold > 0:
        print("Removing near-duplicate posts...")
        corpus, dup_removed = dedupe_near_duplicates(corpus, threshold=args.near_dup_threshold)
        if dup_removed:
            pct = dup_removed / max(len(corpus) + dup_removed, 1)
            print(f"Removed {dup_removed} near-duplicate posts ({pct:.1%}).")
        else:
            print("No near-duplicates detected.")
        print(f"Corpus after dedupe: {len(corpus)} posts")

    lore, pairs, _ = generate_lore_from_corpus(
        corpus,
        sample=args.sample,
        fewshots=args.fewshots,
        rag_fewshots=args.rag_fewshots,
        embeddings_path=args.embeddings,
        min_df=args.min_df,
        max_df=args.max_df,
        max_entity_doc_ratio=args.max_entity_doc_ratio,
        skip_spam_filter=args.skip_spam_filter,
        near_dup_threshold=args.near_dup_threshold,
    )

    with open(args.out_lore, "w", encoding="utf-8") as f:
        json.dump(lore, f, indent=2)
    print(f"Saved lorebook to {args.out_lore}")

    if args.fewshots > 0:
        save_fewshots(pairs, args.out_fewshots)
        print(f"Saved {len(pairs)} few-shots to {args.out_fewshots}")

    print("Done.")


if __name__ == "__main__":
    main()
