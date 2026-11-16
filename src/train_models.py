"""
STEP 3 - MACHINE LEARNING PIPELINE (two stages)

  Stage A  EVIDENCE FINDER  -  "Which clauses in this document talk about rule R?"
           A document has ~70 clauses, only ~2 are relevant, so we must find them first.
             A0  random clauses                         (sanity baseline)
             A1  TF-IDF cosine similarity to rule text  (unsupervised, no training)
             A2  Logistic Regression clause classifier  (supervised, one per rule)
           Metric: Recall@k = % of true evidence clauses found in the top-k.

  Stage B  COMPLIANCE JUDGE  -  "Given that evidence, is the rule followed?"
           Output: COMPLIANT / NON_COMPLIANT / NOT_ADDRESSED
             B0  always predict the most common label          (baseline)
             B1  predict the most common label for each rule   (stronger baseline)
             B2  TF-IDF of retrieved clauses + Logistic Regression, one model per rule
           Metrics: accuracy, macro-F1 (treats all 3 classes equally), per-class F1.

Train on train split, tune nothing on test, report on the held-out test split.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

from style import COLORS, LABELS, PRETTY, BLUE, ORANGE, INK2, MUTED, plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(OUT, "figures")
MODELS = os.path.join(ROOT, "models")
TOP_K = 3          # how many clauses Stage A hands to Stage B
KS = [1, 3, 5, 10]
RNG = np.random.default_rng(42)


def load(split):
    clauses = pd.read_csv(os.path.join(PROC, f"clauses_{split}.csv"))
    clauses["clause_text"] = clauses["clause_text"].fillna("")
    pairs = pd.read_csv(os.path.join(PROC, f"pairs_{split}.csv"))
    pairs["evidence_clause_ids"] = pairs["evidence_clause_ids"].fillna("").astype(str)
    pairs["gold"] = pairs["evidence_clause_ids"].apply(lambda s: {int(x) for x in s.split()})
    return clauses, pairs


def clause_labels(clauses, pairs, rule_id):
    """1 if the clause is evidence for this rule, else 0."""
    gold = pairs[pairs.rule_id == rule_id].set_index("doc_id")["gold"].to_dict()
    return np.array([int(c in gold.get(d, set())) for d, c in zip(clauses.doc_id, clauses.clause_id)])


def recall_at_k(scores, clauses, pairs, rule_id, k):
    """Of the true evidence clauses, what fraction appears in our top-k?"""
    df = clauses[["doc_id", "clause_id"]].assign(score=scores)
    hits, total = 0, 0
    for _, row in pairs[(pairs.rule_id == rule_id) & (pairs.label != "NOT_ADDRESSED")].iterrows():
        if not row.gold:
            continue
        top = df[df.doc_id == row.doc_id].nlargest(k, "score").clause_id
        hits += len(row.gold & set(top))
        total += len(row.gold)
    return hits, total


def top_k_text(scores, clauses, k):
    """For each document, join the text of its k highest-scoring clauses + the best score."""
    df = clauses.assign(score=scores)
    out = {}
    for doc_id, g in df.groupby("doc_id"):
        top = g.nlargest(k, "score")
        out[doc_id] = (" ".join(top.clause_text), top.score.max())
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    rules = pd.read_csv(os.path.join(PROC, "rules.csv"))
    tr_c, tr_p = load("train")
    te_c, te_p = load("test")

    # one shared vocabulary for clauses (word unigrams + bigrams)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True,
                          stop_words="english")
    X_tr = vec.fit_transform(tr_c.clause_text)
    X_te = vec.transform(te_c.clause_text)

    # ===================== STAGE A: evidence finder =====================
    rec = {m: {k: [0, 0] for k in KS} for m in ["A0 Random", "A1 TF-IDF similarity", "A2 Supervised LogReg"]}
    retrieved_tr, retrieved_te = {}, {}
    retrievers = {}

    for _, r in rules.iterrows():
        rid = r.rule_id
        # A0 random
        s0 = RNG.random(len(te_c))
        # A1 unsupervised similarity to the rule sentence
        s1 = cosine_similarity(X_te, vec.transform([r.rule_text])).ravel()
        # A2 supervised: learn what evidence for THIS rule looks like
        y = clause_labels(tr_c, tr_p, rid)
        clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000)
        clf.fit(X_tr, y)
        retrievers[rid] = clf
        s2 = clf.predict_proba(X_te)[:, 1]

        for name, s in [("A0 Random", s0), ("A1 TF-IDF similarity", s1), ("A2 Supervised LogReg", s2)]:
            for k in KS:
                h, t = recall_at_k(s, te_c, te_p, rid, k)
                rec[name][k][0] += h
                rec[name][k][1] += t

        # hand the best clauses to Stage B. For TRAIN we use cross-fitted scores would be ideal;
        # we use in-sample scores here (simple, slightly optimistic for training only).
        retrieved_tr[rid] = top_k_text(clf.predict_proba(X_tr)[:, 1], tr_c, TOP_K)
        retrieved_te[rid] = top_k_text(s2, te_c, TOP_K)
        print(f"  stage A trained for {rid:6s} ({r.rule_name})")

    recall_df = pd.DataFrame({m: {f"Recall@{k}": v[k][0] / v[k][1] for k in KS} for m, v in rec.items()}).T
    print("\nSTAGE A - evidence retrieval on TEST set")
    print(recall_df.map("{:.1%}".format).to_string())
    recall_df.to_csv(os.path.join(OUT, "stageA_retrieval_recall.csv"))

    # ===================== STAGE B: compliance judge =====================
    j_vec = TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True)
    j_vec.fit([t for rid in retrieved_tr for t, _ in retrieved_tr[rid].values()])

    def feats(pairs, retrieved, rid):
        sub = pairs[pairs.rule_id == rid]
        texts = [retrieved[rid][d][0] for d in sub.doc_id]
        best = np.array([[retrieved[rid][d][1]] for d in sub.doc_id])
        return hstack([j_vec.transform(texts), csr_matrix(best * 3)]).tocsr(), sub

    majority = tr_p.label.mode()[0]
    rule_prior = tr_p.groupby("rule_id").label.agg(lambda s: s.mode()[0])
    preds = {"B0 Majority class": [], "B1 Per-rule prior": [], "B2 Retrieval + LogReg": []}
    gold_all, rule_all = [], []
    judges = {}

    for rid in rules.rule_id:
        Xb_tr, sub_tr = feats(tr_p, retrieved_tr, rid)
        Xb_te, sub_te = feats(te_p, retrieved_te, rid)
        if sub_tr.label.nunique() > 1:
            judge = LogisticRegression(C=2.0, class_weight="balanced", max_iter=3000)
            judge.fit(Xb_tr, sub_tr.label)
            p = judge.predict(Xb_te)
        else:
            judge, p = None, np.array([sub_tr.label.iloc[0]] * len(sub_te))
        judges[rid] = judge
        preds["B0 Majority class"] += [majority] * len(sub_te)
        preds["B1 Per-rule prior"] += [rule_prior[rid]] * len(sub_te)
        preds["B2 Retrieval + LogReg"] += list(p)
        gold_all += list(sub_te.label)
        rule_all += [rid] * len(sub_te)

    results = []
    for name, p in preds.items():
        results.append({
            "model": name,
            "accuracy": accuracy_score(gold_all, p),
            "macro_F1": f1_score(gold_all, p, average="macro", labels=LABELS),
            **{f"F1_{PRETTY[l]}": f1_score(gold_all, p, labels=[l], average="macro") for l in LABELS},
        })
    res_df = pd.DataFrame(results).set_index("model")
    res_df.to_csv(os.path.join(OUT, "stageB_classification_results.csv"))
    print("\nSTAGE B - compliance classification on TEST set")
    print(res_df.map("{:.3f}".format).to_string())
    best = preds["B2 Retrieval + LogReg"]
    report = classification_report(gold_all, best, labels=LABELS, digits=3)
    print("\nDetailed report for B2:\n" + report)

    per_rule = (pd.DataFrame({"rule_id": rule_all, "gold": gold_all, "pred": best})
                .groupby("rule_id").apply(lambda g: (g.gold == g.pred).mean(), include_groups=False)
                .rename("accuracy").reset_index().merge(rules, on="rule_id"))
    per_rule.to_csv(os.path.join(OUT, "stageB_per_rule_accuracy.csv"), index=False)

    # ===================== figures =====================
    # Stage A recall curve
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = {"A0 Random": MUTED, "A1 TF-IDF similarity": ORANGE, "A2 Supervised LogReg": BLUE}
    for m in recall_df.index:
        ax.plot(KS, recall_df.loc[m].values, marker="o", linewidth=2, markersize=7, color=colors[m], label=m)
        ax.text(KS[-1] + 0.3, recall_df.loc[m].values[-1], f"{recall_df.loc[m].values[-1]:.0%}",
                va="center", color=INK2)
    ax.set_xticks(KS)
    ax.set_xlabel("k  (number of clauses the system returns)")
    ax.set_ylabel("Recall@k  (evidence found)")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Stage A: finding the evidence clauses")
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(FIG, "04_stageA_recall.png"))
    plt.close(fig)

    # Stage B model comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(res_df))
    ax.bar(x - 0.18, res_df.accuracy, 0.34, color=MUTED, label="Accuracy", edgecolor="#fcfcfb", linewidth=1.5)
    ax.bar(x + 0.18, res_df.macro_F1, 0.34, color=BLUE, label="Macro-F1", edgecolor="#fcfcfb", linewidth=1.5)
    for i, (a, f) in enumerate(zip(res_df.accuracy, res_df.macro_F1)):
        ax.text(i - 0.18, a + 0.015, f"{a:.2f}", ha="center", color=INK2, fontsize=9)
        ax.text(i + 0.18, f + 0.015, f"{f:.2f}", ha="center", color=INK2, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(res_df.index)
    ax.set_ylim(0, 1)
    ax.set_title("Stage B: compliance decision - baselines vs our model")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(FIG, "05_stageB_model_comparison.png"))
    plt.close(fig)

    # confusion matrix
    cm = confusion_matrix(gold_all, best, labels=LABELS)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > cm.max() * 0.55 else "#0b0b0b")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([PRETTY[l] for l in LABELS])
    ax.set_yticklabels([PRETTY[l] for l in LABELS])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual (lawyer label)")
    ax.set_title("Confusion matrix - B2 on test set")
    fig.savefig(os.path.join(FIG, "06_confusion_matrix.png"))
    plt.close(fig)

    joblib.dump({"vec": vec, "retrievers": retrievers, "j_vec": j_vec, "judges": judges, "top_k": TOP_K},
                os.path.join(MODELS, "contractnli_pipeline.joblib"))
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump({"stageA_recall": recall_df.to_dict(orient="index"),
                   "stageB": res_df.to_dict(orient="index")}, f, indent=2)
    print(f"Saved models to {MODELS} and figures to {FIG}")


if __name__ == "__main__":
    main()
