"""Lexical helpers for groundedness / relevance / hallucination heuristics."""

from __future__ import annotations

import re
from collections import Counter

STOP = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "is",
    "are",
    "was",
    "were",
    "be",
    "as",
    "at",
    "by",
    "with",
    "from",
    "that",
    "this",
    "it",
    "you",
    "your",
    "can",
    "if",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower()) if t not in STOP]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def overlap_ratio(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta:
        return 0.0
    ca, cb = Counter(ta), Counter(tb)
    inter = sum((ca & cb).values())
    return inter / max(len(ta), 1)


def extract_claim_units(answer: str) -> list[str]:
    """Split answer into sentence-level claims for support checks."""
    sents = sentences(answer)
    return sents if sents else [answer.strip()]


NUM_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
PROPER_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]+\b")


def unsupported_atoms(answer: str, context: str) -> list[str]:
    """Numbers / capitalized tokens in answer missing from context (case-insensitive for numbers)."""
    ctx_lower = context.lower()
    bad: list[str] = []
    for num in NUM_RE.findall(answer):
        if num.lower() not in ctx_lower:
            bad.append(num)
    for prop in PROPER_RE.findall(answer):
        if prop.lower() in {"i", "it"}:
            continue
        if prop.lower() not in ctx_lower:
            bad.append(prop)
    return bad
