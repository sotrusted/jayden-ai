#!/usr/bin/env python3
"""Command-line entry point for orchestrating Spite AI data assets."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

import analyze_spite_style as style_module
from spite_ai import Config, DataManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage Spite AI datasets")
    parser.add_argument("--input", default="data/spite_dump.json", help="Path to raw dump file")
    parser.add_argument(
        "--steps",
        nargs="*",
        choices=DataManager.STAGE_ORDER,
        help="Subset of stages to run (default: all)",
    )
    parser.add_argument(
        "--force",
        nargs="*",
        metavar="STAGE",
        choices=(*DataManager.STAGE_ORDER, "all"),
        help="Force specific stages (or 'all') to re-run even if cached",
    )
    parser.add_argument("--no-filters", action="store_true", help="Skip spam + near-duplicate filtering")
    parser.add_argument(
        "--near-dup-threshold",
        type=int,
        default=style_module.NEAR_DUP_THRESHOLD,
        help="SimHash threshold for dedupe (used in corpus/style/lore stages)",
    )
    parser.add_argument("--rag-fewshots", action="store_true", help="Use RAG few-shots when generating lore")
    parser.add_argument("--fewshots", type=int, default=5, help="How many few-shot pairs to generate")
    parser.add_argument("--sample-size", type=int, default=4000, help="Sample size for stats/sentiment")
    parser.add_argument("--min-df", type=int, default=5, help="Min document frequency for phrase mining")
    parser.add_argument("--max-df", type=float, default=0.1, help="Max document frequency proportion for phrase mining")
    parser.add_argument(
        "--max-entity-doc-ratio",
        type=float,
        default=0.1,
        help="Drop entities appearing in more than this share of posts",
    )
    parser.add_argument("--progress-only", action="store_true", help="Print existing progress state and exit")
    return parser.parse_args()


def resolve_force(force_args: Sequence[str] | None, steps: Sequence[str] | None) -> bool | Sequence[str]:
    if not force_args:
        return False
    if "all" in force_args:
        return True
    return force_args


def main() -> None:
    args = parse_args()
    config = Config.from_env()
    manager = DataManager(args.input, config=config)

    if args.progress_only:
        progress_path = manager.progress_path
        if progress_path.exists():
            with progress_path.open() as handle:
                print(json.dumps(json.load(handle), indent=2))
        else:
            print("No progress recorded yet.")
        return

    force = resolve_force(args.force, args.steps)
    results = manager.run(
        steps=args.steps,
        force=force,
        apply_filters=not args.no_filters,
        near_dup_threshold=args.near_dup_threshold,
        rag_fewshots=args.rag_fewshots,
        fewshots=args.fewshots,
        sample_size=args.sample_size,
        min_df=args.min_df,
        max_df=args.max_df,
        max_entity_doc_ratio=args.max_entity_doc_ratio,
    )

    for stage in DataManager.STAGE_ORDER:
        result = results.get(stage)
        if not result:
            continue
        print(f"[{stage}] {result.status}")
        if result.details:
            print(json.dumps(result.details, indent=2))


if __name__ == "__main__":
    main()
