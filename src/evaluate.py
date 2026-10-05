"""CLI: evaluate a JSONL dataset of RAG Q/A pairs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .scorer import score_answer


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG answer quality evaluator")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.dataset)
    results = [
        score_answer(
            example_id=row["id"],
            question=row["question"],
            context=row["context"],
            answer=row["answer"],
        ).to_dict()
        for row in rows
    ]

    error_counts = Counter()
    for r in results:
        for e in r["error_categories"]:
            error_counts[e] += 1

    report = {
        "evaluator": "manalipansuriya57",
        "n": len(results),
        "avg_groundedness": round(sum(r["groundedness"] for r in results) / max(len(results), 1), 3),
        "avg_relevance": round(sum(r["relevance"] for r in results) / max(len(results), 1), 3),
        "avg_hallucination_risk": round(
            sum(r["hallucination_risk"] for r in results) / max(len(results), 1), 3
        ),
        "error_category_counts": dict(error_counts),
        "per_question": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.out}")
    print(
        f"n={report['n']} groundedness={report['avg_groundedness']} "
        f"relevance={report['avg_relevance']} "
        f"hallucination_risk={report['avg_hallucination_risk']}"
    )
    print("errors:", report["error_category_counts"])


if __name__ == "__main__":
    main()
