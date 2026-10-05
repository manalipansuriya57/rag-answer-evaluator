"""Score a single RAG answer against retrieved context."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .textutil import extract_claim_units, overlap_ratio, unsupported_atoms


@dataclass
class ScoreResult:
    id: str
    groundedness: float
    relevance: float
    hallucination_risk: float
    error_categories: list[str]
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def score_answer(
    *,
    example_id: str,
    question: str,
    context: str,
    answer: str,
    grounded_threshold: float = 0.35,
    relevance_threshold: float = 0.2,
) -> ScoreResult:
    claims = extract_claim_units(answer)
    if not claims:
        return ScoreResult(
            id=example_id,
            groundedness=0.0,
            relevance=0.0,
            hallucination_risk=1.0,
            error_categories=["empty_answer"],
            notes=["Answer was empty"],
        )

    claim_scores = [overlap_ratio(claim, context) for claim in claims]
    groundedness = sum(claim_scores) / len(claim_scores)
    relevance = overlap_ratio(answer, question)
    atoms = unsupported_atoms(answer, context)
    # Risk rises with unsupported atoms and falls with groundedness
    hallucination_risk = min(1.0, (len(atoms) * 0.25) + max(0.0, 0.6 - groundedness))

    errors: list[str] = []
    notes: list[str] = []

    if groundedness < grounded_threshold:
        errors.append("ungrounded")
        notes.append(f"Low claim-context overlap ({groundedness:.2f})")
    if relevance < relevance_threshold:
        errors.append("irrelevant")
        notes.append(f"Low answer-question overlap ({relevance:.2f})")
    if atoms:
        errors.append("hallucination")
        notes.append("Unsupported atoms: " + ", ".join(atoms[:8]))
    if hallucination_risk >= 0.5 and "hallucination" not in errors:
        errors.append("hallucination")
        notes.append(f"Elevated hallucination_risk ({hallucination_risk:.2f})")

    return ScoreResult(
        id=example_id,
        groundedness=round(groundedness, 3),
        relevance=round(relevance, 3),
        hallucination_risk=round(hallucination_risk, 3),
        error_categories=errors,
        notes=notes,
    )
