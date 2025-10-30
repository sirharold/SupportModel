
"""
Retrieval metrics with ranx (MAP, MRR, nDCG, Precision, Recall, F1) — Colab friendly.
-------------------------------------------------------------------------------------
Usage in Colab:
    !pip -q install ranx==0.3.20 pandas==2.2.2

    from metrics_evaluator import evaluate_retrieval_ranx, to_ranx_dicts_from_lists

    # Example minimal inputs (binary relevance):
    qrels_dict = {
        "q1": {"d1": 1, "d5": 1},
        "q2": {"d2": 1}
    }
    run_dict = {
        "q1": {"d1": 3.2, "d3": 2.7, "d5": 1.1},
        "q2": {"d2": 0.9, "d8": 0.4}
    }

    results = evaluate_retrieval_ranx(qrels_dict, run_dict, k_list=(1,3,5,10), per_query=True)
    results["macro"]   # macro-averages
    results["per_q"]   # per-query DataFrame

Inputs accepted
---------------
- qrels_dict: dict[str, dict[str, int/float]]
    Mapping query_id -> {doc_id -> relevance}. Can be binary (0/1) or graded (0..n).
- run_dict: dict[str, dict[str, float]]
    Mapping query_id -> {doc_id -> score}. Higher score = higher rank.

If your current pipeline keeps lists, use helpers:
-------------------------------------------------
    qrels_dict, run_dict = to_ranx_dicts_from_lists(
        gold=[{"query_id": "...", "relevant_ids": ["d1","d2"]}, ...],
        retrieved=[{"query_id": "...", "docs": [{"doc_id":"d1","score":3.4}, ...]}, ...],
        graded=False  # set True and pass "relevances" to support graded
    )

Why ranx?
---------
- Fast & lightweight (pure Python/NumPy), perfect for Colab.
- Implements standard IR metrics with the same definitions used in TREC-style evaluation.
- Returns macro and per-query metrics in one shot.
"""
from typing import Dict, Iterable, List, Tuple, Union, Sequence, Optional
import math
import pandas as pd

# We import lazily inside functions so importing this module does not require ranx immediately.
_RANX_WARNING = (
    "The 'ranx' package is required. In Colab run: !pip -q install ranx==0.3.20 pandas==2.2.2"
)

MetricValue = Union[int, float]

def safe_metrics_available() -> bool:
    """Return True if 'ranx' can be imported, else False."""
    try:
        import ranx  # noqa: F401
        return True
    except Exception:
        return False

def to_ranx_dicts_from_lists(
    gold: Sequence[Dict],
    retrieved: Sequence[Dict],
    graded: bool = False,
    relevance_key: str = "relevance",
    doc_id_key: str = "doc_id",
    score_key: str = "score",
    relevant_ids_key: str = "relevant_ids",
) -> Tuple[Dict[str, Dict[str, MetricValue]], Dict[str, Dict[str, float]]]:
    """Convert common list-based inputs to ranx-friendly dicts.

    Parameters
    ----------
    gold : list of dicts
        Each item like:
            graded=False: {"query_id": "q1", "relevant_ids": ["d1","d5"]}
            graded=True:  {"query_id": "q1", "relevances": [{"doc_id":"d1","relevance":2}, ...]}
        For graded=True, use `relevant_ids_key` = "relevances" and include `relevance_key` per item.

    retrieved : list of dicts
        Each item like:
            {"query_id": "q1", "docs": [{"doc_id":"d1","score":3.4},{"doc_id":"d3","score":2.1}]}

    Returns
    -------
    (qrels_dict, run_dict)
        qrels_dict: {q_id: {doc_id: rel, ...}, ...}
        run_dict:   {q_id: {doc_id: score, ...}, ...}
    """
    qrels_dict = {}
    run_dict = {}

    # Build qrels
    for item in gold:
        qid = item.get("query_id")
        if qid is None:
            # Skip malformed entries silently
            continue

        if graded:
            rel_items = item.get(relevant_ids_key, [])  # usually "relevances"
            relmap = {}
            for r in rel_items:
                did = r.get(doc_id_key)
                rel = r.get(relevance_key, 0)
                if did is None:
                    continue
                relmap[str(did)] = float(rel)
            if relmap:
                qrels_dict[str(qid)] = relmap
        else:
            ids = item.get(relevant_ids_key, [])
            relmap = {str(d): 1.0 for d in ids}
            if relmap:
                qrels_dict[str(qid)] = relmap

    # Build run
    for item in retrieved:
        qid = item.get("query_id")
        if qid is None:
            continue
        docs = item.get("docs", [])
        score_map = {}
        for d in docs:
            did = d.get(doc_id_key)
            sc = d.get(score_key, None)
            if did is None or sc is None:
                continue
            score_map[str(did)] = float(sc)
        if score_map:
            run_dict[str(qid)] = score_map

    return qrels_dict, run_dict

def _expand_metric_list(k_list: Sequence[int]) -> List[str]:
    metrics = []
    for k in k_list:
        metrics.extend([
            f"map@{k}",      # Mean Average Precision @k
            f"mrr@{k}",      # Mean Reciprocal Rank @k
            f"ndcg@{k}",     # Normalized Discounted Cumulative Gain @k
            f"precision@{k}",
            f"recall@{k}",
            f"f1@{k}",
        ])
    # Also add overall (no cutoff) metrics where applicable
    metrics.extend(["map", "mrr", "ndcg"])
    return metrics

def evaluate_retrieval_ranx(
    qrels_dict: Dict[str, Dict[str, MetricValue]],
    run_dict: Dict[str, Dict[str, float]],
    k_list: Sequence[int] = (1, 3, 5, 10),
    per_query: bool = True,
    drop_queries_with_no_qrels: bool = True,
) -> Dict[str, Union[dict, "pd.DataFrame"]]:
    """Evaluate retrieval results with ranx in one call.

    Parameters
    ----------
    qrels_dict : {query_id: {doc_id: relevance, ...}, ...}
        Binary (0/1) or graded (>1) relevance values supported.
    run_dict : {query_id: {doc_id: score, ...}, ...}
        Scores used to rank documents per query (higher is better).
    k_list : sequence of int
        Cutoffs for @k metrics to compute.
    per_query : bool
        If True, also return a DataFrame with per-query metrics.
    drop_queries_with_no_qrels : bool
        If True, queries with no relevant documents in qrels are removed before evaluation.

    Returns
    -------
    dict with:
        - "macro": dict[str, float] with macro averages
        - "per_q": pandas.DataFrame (if per_query=True), else None
    """
    try:
        from ranx import Qrels, Run, evaluate
    except Exception as e:
        raise ImportError(_RANX_WARNING) from e

    # Optionally drop queries lacking qrels
    if drop_queries_with_no_qrels:
        filtered_run = {qid: docs for qid, docs in run_dict.items() if qid in qrels_dict}
    else:
        filtered_run = run_dict

    # Construct ranx structures
    qrels = Qrels(qrels_dict)
    run = Run(filtered_run)

    # Prepare metric list
    metrics = _expand_metric_list(k_list)

    # Evaluate
    macro = evaluate(qrels, run, metrics=metrics, per_query=False)

    result = {"macro": macro, "per_q": None}

    if per_query:
        # Evaluate per-query; ranx returns nested mapping {metric: {qid: value, ...}, ...}
        perq = evaluate(qrels, run, metrics=metrics, per_query=True)
        # Normalize into DataFrame
        # Build rows per query id
        rows = {}
        for metric, qmap in perq.items():
            for qid, val in qmap.items():
                rows.setdefault(qid, {})[metric] = val
        df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
        df.index.name = "query_id"
        result["per_q"] = df

    return result

# ---------------------- OPTIONAL FALLBACK (no ranking metrics) ----------------------
# If installing libraries is not possible, you can still compute *top-k classification-style*
# precision/recall/F1 per query with this fallback. This is NOT a replacement for MAP/MRR/nDCG.
def topk_precision_recall_f1_fallback(
    qrels_dict: Dict[str, Dict[str, MetricValue]],
    run_dict: Dict[str, Dict[str, float]],
    k: int = 10,
) -> pd.DataFrame:
    """Compute per-query precision/recall/F1 at k without external deps (no MAP/MRR/nDCG).

    Returns
    -------
    DataFrame with columns: [precision@k, recall@k, f1@k]
    """
    import numpy as np
    rows = []
    for qid, rels in qrels_dict.items():
        relevant = set([d for d, r in rels.items() if r > 0])
        ranked = sorted(run_dict.get(qid, {}).items(), key=lambda x: x[1], reverse=True)
        topk = [d for d, _ in ranked[:k]]
        topk_set = set(topk)

        tp = len(topk_set & relevant)
        fp = len(topk_set - relevant)
        fn = len(relevant - topk_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rows.append({"query_id": qid, f"precision@{k}": precision, f"recall@{k}": recall, f"f1@{k}": f1})
    return pd.DataFrame(rows).set_index("query_id")
