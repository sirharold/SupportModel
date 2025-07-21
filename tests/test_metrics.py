import sys, os
import math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.metrics import compute_precision_recall_f1, compute_mrr, compute_ndcg

ranked = [
    {"link": "l1"},
    {"link": "l2"},
    {"link": "l3"},
]
relevant = ["l1", "l3", "l4"]

def test_precision_recall_f1():
    p, r, f1 = compute_precision_recall_f1(ranked, relevant)
    assert round(p, 2) == 0.67
    assert round(r, 2) == 0.67
    assert round(f1, 2) == 0.67

def test_mrr():
    assert compute_mrr(ranked, ["l3"]) == 1/3

def test_ndcg():
    ndcg = compute_ndcg(ranked, ["l1", "l3"])
    assert round(ndcg, 3) == round((1 + 0 + 0.5) / (1 + 1/math.log2(3)), 3)


