# Retrieval LLM Toolkit

A hands-on implementation of retrieval-augmented generation components and LLM evaluation tools, built from the ground up — no LangChain, no abstractions hiding the internals.

> Every piece is written from scratch so the behaviour is fully transparent and easy to reason about.

---

## What It Does

This toolkit covers the core building blocks that underlie modern LLM-powered search and question-answering systems:

- **Retrieval** — dense vector search with FAISS, keyword-based BM25, and a hybrid fusion layer using Reciprocal Rank Fusion
- **Query expansion** — HyDE generates a hypothetical answer first, then searches with that, improving recall on difficult queries
- **Reranking** — a cross-encoder scores candidates by relevance; MMR handles diversity
- **Evaluation** — a RAGAS-inspired suite measuring faithfulness, answer relevancy, context precision and recall, all running offline
- **Prompt engineering** — zero-shot and few-shot chain-of-thought, dynamic example selection, Pydantic structured outputs
- **Text-to-SQL** — natural language to SQL with schema introspection and an automatic self-correction loop on errors

---

## Results

| Retriever | Recall@3 |
|---|---|
| Dense (FAISS) | 0.80 |
| BM25 | 0.67 |
| **Hybrid (RRF)** | **0.87** |
| HyDE | 0.73 |
| CrossEncoder Reranker | 0.80 |

| RAGAS Metric | Score |
|---|---|
| Faithfulness | 0.78 |
| Answer Relevancy | 0.72 |
| Context Precision | 0.80 |
| Context Recall | 0.75 |

---

## Project Structure

```
rrs/
├── llm.py                  # OpenAI wrapper + offline MockLLM
├── pipeline.py             # End-to-end RAG pipeline
├── chunking/               # Fixed, recursive, semantic chunkers
├── retrieval/              # FAISS, BM25, RRF, HyDE, rerankers
├── evaluation/             # Faithfulness, relevancy, RAGAS
├── prompts/                # CoT, few-shot, structured output
└── sql_agent/              # Text-to-SQL with self-correction
scripts/                    # Runnable demo scripts
tests/                      # Unit + integration tests
```

---

## Quickstart

```bash
git clone https://github.com/Ritikkumar0730/retrieval-llm-toolkit
cd retrieval-llm-toolkit
pip install -r requirements.txt
```

### Run demos (no API key needed)

```bash
python scripts/run_chunking_comparison.py
python scripts/run_retrieval_benchmark.py
python scripts/run_rag_evaluation.py
python scripts/run_sql_agent.py
```

### With a real OpenAI model

```bash
export OPENAI_API_KEY="sk-..."
python scripts/run_rag_evaluation.py
```

### Tests

```bash
pytest tests/ -v -m "not integration"
```

---

## Design Notes

**No LangChain** — writing each component from scratch makes the internals visible. There are no magic abstractions to debug around.

**Offline by default** — a deterministic MockLLM means all four demo scripts run without any API key. Plug in a real model by setting `OPENAI_API_KEY`.

**Text-to-SQL for annotation platforms** — the SQL agent seeds a realistic annotation database (annotators, tasks, annotations tables) so queries like "which annotators completed the most NLP tasks this week?" work out of the box.

---

## Dependencies

- `sentence-transformers` — bi-encoder embeddings
- `faiss-cpu` — vector index
- `rank-bm25` — sparse retrieval
- `pydantic` — structured output parsing
- `numpy`
- `openai` — optional
