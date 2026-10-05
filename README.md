# RAG Answer Quality Evaluator

Scores chatbot answers for **groundedness**, **relevance**, and **hallucination** against source documents. Emits a per-question error report used to tune prompts and retrieval.

## Stack

Python · embedding-free lexical overlap metrics (demo) · optional OpenAI judge via `OPENAI_API_KEY`

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.evaluate --dataset data/sample_eval.jsonl --out reports/latest.json
```

## Metrics

| Metric | Meaning |
|--------|---------|
| groundedness | Share of answer claims supported by retrieved context |
| relevance | Overlap between answer and question intent terms |
| hallucination_risk | Unsupported named entities / numbers not in context |

No live vector DB required for the sample run. Swap `src/retriever.py` for pgvector when wiring production retrieval.
