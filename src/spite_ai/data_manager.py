"""Unified data preparation pipeline for Spite AI.

The DataManager wraps the standalone scripts (embedding, style analysis,
lorebook generation) into a staged workflow with caching, progress tracking,
and optional filtering toggles. It reuses the helper functions exposed by the
scripts to guarantee behavioural parity while avoiding duplicate work across
runs.
"""

from __future__ import annotations

import json
import time
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.append(str(MODULE_ROOT))

import embed_posts
import analyze_spite_style as style_module
import generate_lore_and_fewshots as lore_module


DEFAULT_CACHE_DIR = Path("data/.cache")


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: str
    details: Dict[str, object]


class DataManager:
    """Coordinate dataset preparation tasks with caching and checkpoints."""

    STAGE_ORDER = ("corpus", "embeddings", "style", "lore")

    def __init__(
        self,
        input_path: Path | str,
        *,
        config: Optional[Config] = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        self.config = config or Config.from_env()
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.cache_dir / "data_manager_manifest.json"
        self.progress_path = self.cache_dir / "data_manager_progress.json"

        self._manifest = self._load_manifest()
        self._progress = self._load_progress()

        self._input_hash = self._hash_file(self.input_path)
        if self._manifest.get("input_hash") != self._input_hash:
            self._manifest = {"input_hash": self._input_hash, "stages": {}}

        self._progress.update(
            {
                "input_path": str(self.input_path),
                "input_hash": self._input_hash,
                "stages": self._progress.get("stages", {}),
            }
        )

        # Working state populated as stages run or when reading from disk.
        self._corpus: List[str] | None = None
        self._metadata: List[dict] | None = None
        self._corpus_hash: Optional[str] = None
        self._corpus_stats: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API

    def run(
        self,
        *,
        steps: Optional[Sequence[str]] = None,
        force: bool | Sequence[str] = False,
        apply_filters: bool = True,
        near_dup_threshold: int = style_module.NEAR_DUP_THRESHOLD,
        rag_fewshots: bool = False,
        fewshots: int = 5,
        sample_size: int = 4000,
        min_df: int = 5,
        max_df: float = 0.1,
        max_entity_doc_ratio: float = 0.1,
    ) -> Dict[str, StageResult]:
        """Run selected stages, skipping cached work when possible."""

        requested = tuple(steps) if steps else self.STAGE_ORDER
        force_set = set(requested if force is True else force or [])
        results: Dict[str, StageResult] = {}

        for stage in requested:
            if stage not in self.STAGE_ORDER:
                raise ValueError(f"Unknown stage: {stage}")

            stage_force = stage in force_set
            stage_result = self._run_stage(
                stage,
                force=stage_force,
                apply_filters=apply_filters,
                near_dup_threshold=near_dup_threshold,
                rag_fewshots=rag_fewshots,
                fewshots=fewshots,
                sample_size=sample_size,
                min_df=min_df,
                max_df=max_df,
                max_entity_doc_ratio=max_entity_doc_ratio,
            )
            results[stage] = stage_result

        return results

    # ------------------------------------------------------------------
    # Stage runners

    def _run_stage(
        self,
        stage: str,
        *,
        force: bool,
        apply_filters: bool,
        near_dup_threshold: int,
        rag_fewshots: bool,
        fewshots: int,
        sample_size: int,
        min_df: int,
        max_df: float,
        max_entity_doc_ratio: float,
    ) -> StageResult:
        stage_method = getattr(self, f"_stage_{stage}")
        stage_signature = None
        outputs: Sequence[Path | str] = ()
        details: Dict[str, object] = {}

        if stage == "corpus":
            stage_signature = self._signature(
                stage=stage,
                input_hash=self._input_hash,
                filters=apply_filters,
                near_dup_threshold=near_dup_threshold,
            )
            outputs = (self.config.CORPUS_PATH, self.config.METADATA_PATH)
        elif stage == "embeddings":
            self._ensure_corpus_loaded()
            stage_signature = self._signature(
                stage=stage,
                corpus_hash=self._corpus_hash,
                model=self.config.MODEL_NAME,
            )
            outputs = (self.config.EMBEDDINGS_PATH,)
        elif stage == "style":
            self._ensure_corpus_loaded()
            stage_signature = self._signature(
                stage=stage,
                corpus_hash=self._corpus_hash,
                near_dup_threshold=near_dup_threshold,
            )
            outputs = (self.config.STYLE_PROFILE_PATH, self.config.SYSTEM_PROMPT_PATH)
        elif stage == "lore":
            self._ensure_corpus_loaded()
            stage_signature = self._signature(
                stage=stage,
                corpus_hash=self._corpus_hash,
                fewshots=fewshots,
                rag=rag_fewshots,
                min_df=min_df,
                max_df=max_df,
                max_entity_ratio=max_entity_doc_ratio,
                near_dup_threshold=near_dup_threshold,
            )
            outputs = (self.config.LOREBOOK_PATH,)
            if fewshots > 0:
                outputs += (self.config.FEWSHOT_PATH,)
        else:
            raise ValueError(f"Unhandled stage: {stage}")

        if not force and self._stage_cached(stage, stage_signature, outputs):
            details = {"reason": "cache-hit"}
            self._record_progress(stage, "skipped", details)
            return StageResult(stage, "skipped", details)

        self._record_progress(stage, "in_progress", {})
        try:
            details = stage_method(
                apply_filters=apply_filters,
                near_dup_threshold=near_dup_threshold,
                rag_fewshots=rag_fewshots,
                fewshots=fewshots,
                sample_size=sample_size,
                min_df=min_df,
                max_df=max_df,
                max_entity_doc_ratio=max_entity_doc_ratio,
            )
        except Exception as exc:  # pragma: no cover - guard for CLI usage
            fail_details = {"error": str(exc)}
            self._record_progress(stage, "failed", fail_details)
            raise

        self._manifest.setdefault("stages", {})[stage] = {
            "signature": stage_signature,
            "outputs": [str(Path(o)) for o in outputs],
            "timestamp": time.time(),
            "details": details,
        }
        self._save_manifest()

        self._record_progress(stage, "completed", details)
        return StageResult(stage, "completed", details)

    def _stage_corpus(
        self,
        *,
        apply_filters: bool,
        near_dup_threshold: int,
        **_: object,
    ) -> Dict[str, object]:
        posts = embed_posts.load_posts(self.input_path)
        corpus, metadata = embed_posts.flatten_posts(posts)
        embed_posts.describe_corpus(corpus)

        stats = {
            "raw_entries": len(corpus),
            "spam_removed": 0,
            "near_duplicates_removed": 0,
        }

        if apply_filters:
            filtered_corpus, filtered_metadata, filter_stats = self._apply_filters(
                corpus,
                metadata,
                near_dup_threshold=near_dup_threshold,
            )
            corpus = filtered_corpus
            metadata = filtered_metadata
            stats.update(filter_stats)

        embeddings_placeholder = np.empty((0,))  # Maintains parity with saver expectations.
        embed_posts.save_outputs(
            corpus,
            metadata,
            embeddings_placeholder,
            corpus_path=self.config.CORPUS_PATH,
            metadata_path=self.config.METADATA_PATH,
            embeddings_path=self._temporary_embeddings_path(),
        )

        # Remove placeholder embeddings to avoid confusion.
        tmp_path = Path(self._temporary_embeddings_path())
        if tmp_path.exists():
            tmp_path.unlink()

        self._corpus = list(corpus)
        self._metadata = list(metadata)
        self._corpus_hash = self._hash_corpus(self._corpus)
        self._corpus_stats = stats
        stats["corpus_hash"] = self._corpus_hash
        return stats

    def _stage_embeddings(
        self,
        **_: object,
    ) -> Dict[str, object]:
        self._ensure_corpus_loaded()
        embeddings, model = embed_posts.embed_corpus(
            self._corpus,
            model_name=self.config.MODEL_NAME,
        )
        embed_posts.build_faiss_index(embeddings)
        embed_posts.save_outputs(
            self._corpus,
            self._metadata,
            embeddings,
            corpus_path=self.config.CORPUS_PATH,
            metadata_path=self.config.METADATA_PATH,
            embeddings_path=self.config.EMBEDDINGS_PATH,
        )

        return {
            "entries": len(self._corpus),
            "embedding_dim": int(model.get_sentence_embedding_dimension()),
            "corpus_hash": self._corpus_hash,
            "device": str(model.device),
        }

    def _stage_style(
        self,
        *,
        near_dup_threshold: int,
        **_: object,
    ) -> Dict[str, object]:
        self._ensure_corpus_loaded()
        profile = style_module.analyze_corpus_style(
            self._corpus,
            near_dup_threshold=near_dup_threshold,
        )
        prompt = style_module.generate_style_prompt(profile)

        with Path(self.config.STYLE_PROFILE_PATH).open("w") as profile_file:
            json.dump(profile, profile_file, indent=2)
        with Path(self.config.SYSTEM_PROMPT_PATH).open("w") as prompt_file:
            prompt_file.write(prompt)

        return {
            "entries_analyzed": profile.get("total_posts_analyzed", 0),
            "spam_filtered": profile.get("spam_filtered_posts", 0),
            "near_duplicates_removed": profile.get("near_duplicate_removed_posts", 0),
            "corpus_hash": self._corpus_hash,
        }

    def _stage_lore(
        self,
        *,
        near_dup_threshold: int,
        rag_fewshots: bool,
        fewshots: int,
        sample_size: int,
        min_df: int,
        max_df: float,
        max_entity_doc_ratio: float,
        **_: object,
    ) -> Dict[str, object]:
        self._ensure_corpus_loaded()
        lore, pairs, stats_summary = lore_module.generate_lore_from_corpus(
            self._corpus,
            sample=sample_size,
            fewshots=fewshots,
            rag_fewshots=rag_fewshots,
            embeddings_path=self.config.EMBEDDINGS_PATH,
            min_df=min_df,
            max_df=max_df,
            max_entity_doc_ratio=max_entity_doc_ratio,
            skip_spam_filter=False,
            near_dup_threshold=near_dup_threshold,
        )

        with Path(self.config.LOREBOOK_PATH).open("w", encoding="utf-8") as lore_file:
            json.dump(lore, lore_file, indent=2)

        if fewshots > 0:
            lore_module.save_fewshots(pairs, self.config.FEWSHOT_PATH)

        stats_summary.update(
            {
                "fewshots_written": len(pairs) if fewshots > 0 else 0,
                "rag": rag_fewshots,
                "corpus_hash": self._corpus_hash,
            }
        )
        return stats_summary

    # ------------------------------------------------------------------
    # Helpers

    def _apply_filters(
        self,
        corpus: Sequence[str],
        metadata: Sequence[dict],
        *,
        near_dup_threshold: int,
    ) -> Tuple[List[str], List[dict], Dict[str, int]]:
        filtered_corpus, spam_removed = style_module.filter_spammy_docs(corpus)
        filtered_metadata = self._align_metadata(corpus, metadata, filtered_corpus)

        deduped_corpus, dup_removed = style_module.dedupe_near_duplicates(
            filtered_corpus,
            threshold=near_dup_threshold,
        )
        deduped_metadata = self._align_metadata(filtered_corpus, filtered_metadata, deduped_corpus)

        return (
            list(deduped_corpus),
            list(deduped_metadata),
            {
                "spam_removed": spam_removed,
                "near_duplicates_removed": dup_removed,
                "final_entries": len(deduped_corpus),
            },
        )

    @staticmethod
    def _align_metadata(
        original_texts: Sequence[str],
        original_metadata: Sequence[dict],
        filtered_texts: Sequence[str],
    ) -> List[dict]:
        filtered_meta: List[dict] = []
        pointer = 0
        for text, meta in zip(original_texts, original_metadata):
            if pointer >= len(filtered_texts):
                break
            if text == filtered_texts[pointer]:
                filtered_meta.append(meta)
                pointer += 1
        return filtered_meta

    def _ensure_corpus_loaded(self) -> None:
        if self._corpus is not None and self._metadata is not None:
            return

        corpus_path = Path(self.config.CORPUS_PATH)
        metadata_path = Path(self.config.METADATA_PATH)
        if not corpus_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Corpus or metadata missing; run corpus stage first.")

        with corpus_path.open() as corpus_file:
            self._corpus = json.load(corpus_file)
        with metadata_path.open() as metadata_file:
            self._metadata = json.load(metadata_file)

        self._corpus_hash = self._hash_corpus(self._corpus)

    # ------------------------------------------------------------------
    # Persistence helpers

    def _load_manifest(self) -> Dict[str, object]:
        if self.manifest_path.exists():
            with self.manifest_path.open() as handle:
                return json.load(handle)
        return {"input_hash": None, "stages": {}}

    def _save_manifest(self) -> None:
        with self.manifest_path.open("w") as handle:
            json.dump(self._manifest, handle, indent=2)

    def _load_progress(self) -> Dict[str, object]:
        if self.progress_path.exists():
            with self.progress_path.open() as handle:
                return json.load(handle)
        return {"stages": {}}

    def _save_progress(self) -> None:
        with self.progress_path.open("w") as handle:
            json.dump(self._progress, handle, indent=2)

    def _record_progress(self, stage: str, status: str, details: Dict[str, object]) -> None:
        stages = self._progress.setdefault("stages", {})
        stages[stage] = {
            "status": status,
            "details": details,
            "timestamp": time.time(),
        }
        self._save_progress()

    # ------------------------------------------------------------------
    # Hashing & caching helpers

    @staticmethod
    def _hash_file(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _hash_corpus(corpus: Iterable[str]) -> str:
        hasher = hashlib.sha256()
        delim = "\u241f"  # unit separator to keep text boundaries clear
        for text in corpus:
            hasher.update(text.encode("utf-8", errors="ignore"))
            hasher.update(delim.encode("utf-8"))
        return hasher.hexdigest()

    @staticmethod
    def _signature(**kwargs: object) -> str:
        return hashlib.sha256(json.dumps(kwargs, sort_keys=True).encode("utf-8")).hexdigest()

    def _stage_cached(
        self,
        stage: str,
        signature: Optional[str],
        outputs: Sequence[Path | str],
    ) -> bool:
        if signature is None:
            return False

        entry = self._manifest.get("stages", {}).get(stage)
        if not entry or entry.get("signature") != signature:
            return False

        for output in outputs:
            if not Path(output).exists():
                return False
        return True

    def _temporary_embeddings_path(self) -> str:
        return str(self.cache_dir / "_temp_embeddings.npy")
