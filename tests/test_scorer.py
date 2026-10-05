from src.scorer import score_answer


def test_good_answer_low_risk():
    r = score_answer(
        example_id="t1",
        question="What is the refund window for annual plans?",
        context="Annual subscription refunds are available within 14 days of purchase.",
        answer="Annual plans can be refunded within 14 days of purchase.",
    )
    assert r.groundedness > 0.3
    assert "hallucination" not in r.error_categories


def test_bad_answer_flags_hallucination():
    r = score_answer(
        example_id="t2",
        question="What is the refund window for annual plans?",
        context="Annual subscription refunds are available within 14 days of purchase.",
        answer="There is a 90-day unconditional refund on Neptune plans.",
    )
    assert "hallucination" in r.error_categories or r.hallucination_risk >= 0.5
