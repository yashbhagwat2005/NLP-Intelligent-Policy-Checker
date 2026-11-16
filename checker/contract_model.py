"""
CONTRACT CHECK - checks a non-disclosure agreement against the 17 ContractNLI rules
using the two-stage model trained in src/train_models.py.

  Stage A  evidence finder  : scores every clause of the contract for each rule, keeps the top 3
  Stage B  compliance judge : reads those 3 clauses and predicts
                              COMPLIANT / NON_COMPLIANT / NOT_ADDRESSED
"""
import os

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from .engine import split_clauses

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pipeline = None
_rules = None


def load():
    global _pipeline, _rules
    if _pipeline is None:
        _pipeline = joblib.load(os.path.join(ROOT, "models", "contractnli_pipeline.joblib"))
        _rules = pd.read_csv(os.path.join(ROOT, "data", "processed", "rules.csv"))
    return _pipeline, _rules


def check_contract(text, document):
    pipeline, rules = load()
    clauses = [c["clause"] for c in split_clauses(text, document)]
    if not clauses:
        return [], 0
    X = pipeline["vec"].transform(clauses)

    rows = []
    for rule in rules.itertuples():
        # Stage A: how likely is each clause to be evidence for this rule?
        scores = pipeline["retrievers"][rule.rule_id].predict_proba(X)[:, 1]
        top = np.argsort(scores)[::-1][:pipeline["top_k"]]
        evidence_text = " ".join(clauses[i] for i in top)

        # Stage B: same features as in training = TF-IDF of the evidence + best Stage A score
        features = hstack([pipeline["j_vec"].transform([evidence_text]),
                           csr_matrix([[scores[top[0]] * 3]])]).tocsr()
        judge = pipeline["judges"][rule.rule_id]
        probabilities = judge.predict_proba(features)[0]
        status = judge.classes_[probabilities.argmax()]
        confidence = float(probabilities.max())

        rows.append({
            "rule_id": rule.rule_id.upper(), "area": "", "rule": rule.rule_text,
            "rule_type": rule.rule_name, "status": status,
            "evidence": "" if status == "NOT_ADDRESSED" else clauses[top[0]],
            "source": document, "score": round(confidence, 2),
            "reason": f"Model decision from the 3 most relevant clauses ({confidence:.0%} confident).",
            "weak": confidence < 0.60,
        })
    return rows, len(clauses)
