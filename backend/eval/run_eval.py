"""
Runs the test set against the real pipeline and scores it. Sets up a
fresh in-memory notebook from sample.pdf (via the actual ingestion
functions, not the HTTP API) so this can run standalone or in CI
without a running server.

Usage: python -m backend.eval.run_eval
Exits with code 1 if any score falls below thresholds.yaml — this is
what makes it usable as a CI gate.
"""

import json
import sys
import uuid
import yaml
from pathlib import Path

from backend.ingest.loaders import load_document
from backend.ingest.chunker import chunk_pages
from backend.ingest.embed import embed_chunks, build_faiss_index
from backend.ingest.index_store import save_index
from backend.providers.router import get_provider
from backend.retrieval.hybrid import hybrid_search
from backend.retrieval.rerank import rerank
from backend.agents.synthesizer import synthesize_answer
from backend.agents.conflict_checker import check_conflicts

EVAL_DIR = Path(__file__).parent
SAMPLE_PDF = EVAL_DIR.parent.parent / "sample.pdf"  # project root


def setup_test_notebook() -> str:
    """Indexes sample.pdf fresh, returns a notebook_id to query against."""
    notebook_id = f"eval-{uuid.uuid4()}"
    with open(SAMPLE_PDF, "rb") as f:
        file_bytes = f.read()

    doc = load_document("sample.pdf", file_bytes)
    chunks = chunk_pages(doc_id="eval-doc", pages=doc["content"])
    provider = get_provider()
    embeddings = embed_chunks(chunks, provider)
    index, id_map = build_faiss_index(chunks, embeddings)
    save_index(notebook_id, index, id_map)
    return notebook_id


def run_single_test(notebook_id: str, test_case: dict, provider) -> dict:
    """Runs one question through the real pipeline, scores it against expectations."""
    question = test_case["question"]
    candidates = hybrid_search(notebook_id, question, provider, top_k=20)
    top_chunks = rerank(question, candidates, top_k=5)

    retrieved_pages = {c["page"] for c in top_chunks}
    retrieval_hit = test_case["expected_page"] in retrieved_pages

    result = synthesize_answer(question, top_chunks)
    answer_lower = result["answer"].lower()
    keyword_hits = sum(1 for kw in test_case["expected_keywords"] if kw.lower() in answer_lower)
    keyword_match_rate = keyword_hits / len(test_case["expected_keywords"])

    citation_pages = {c["page"] for c in result["citations"]}
    citation_correct = test_case["expected_page"] in citation_pages

    conflict_result = check_conflicts(question, top_chunks)
    conflict_expected = test_case.get("expect_conflict", False)
    conflict_correct = conflict_result["has_conflict"] == conflict_expected

    return {
        "question": question,
        "retrieval_hit": retrieval_hit,
        "keyword_match_rate": keyword_match_rate,
        "citation_correct": citation_correct,
        "conflict_correct": conflict_correct,
    }


def run_eval():
    with open(EVAL_DIR / "test_set.json") as f:
        test_set = json.load(f)
    with open(EVAL_DIR / "thresholds.yaml") as f:
        thresholds = yaml.safe_load(f)

    print(f"Setting up test notebook from {SAMPLE_PDF}...")
    notebook_id = setup_test_notebook()
    provider = get_provider()

    print(f"Running {len(test_set)} test cases...\n")
    results = [run_single_test(notebook_id, tc, provider) for tc in test_set]

    retrieval_hit_rate = sum(r["retrieval_hit"] for r in results) / len(results)
    avg_keyword_match = sum(r["keyword_match_rate"] for r in results) / len(results)
    citation_accuracy = sum(r["citation_correct"] for r in results) / len(results)
    conflict_accuracy = sum(r["conflict_correct"] for r in results) / len(results)

    print(f"{'Question':<55} {'Retr':<6} {'Kw%':<6} {'Cite':<6} {'Conf':<6}")
    for r in results:
        print(f"{r['question'][:53]:<55} "
              f"{'✓' if r['retrieval_hit'] else '✗':<6} "
              f"{r['keyword_match_rate']*100:.0f}%{'':<3} "
              f"{'✓' if r['citation_correct'] else '✗':<6} "
              f"{'✓' if r['conflict_correct'] else '✗':<6}")

    print(f"\nRetrieval hit rate:  {retrieval_hit_rate:.1%}  (min: {thresholds['min_retrieval_hit_rate']:.1%})")
    print(f"Keyword match rate:  {avg_keyword_match:.1%}  (min: {thresholds['min_keyword_match_rate']:.1%})")
    print(f"Citation accuracy:   {citation_accuracy:.1%}  (min: {thresholds['min_citation_accuracy']:.1%})")
    print(f"Conflict accuracy:   {conflict_accuracy:.1%}  (min: {thresholds['min_conflict_accuracy']:.1%})")

    passed = (
        retrieval_hit_rate >= thresholds["min_retrieval_hit_rate"] and
        avg_keyword_match >= thresholds["min_keyword_match_rate"] and
        citation_accuracy >= thresholds["min_citation_accuracy"] and
        conflict_accuracy >= thresholds["min_conflict_accuracy"]
    )

    print(f"\n{'PASSED' if passed else 'FAILED'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    run_eval()