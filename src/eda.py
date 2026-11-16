"""
STEP 2 - EXPLORATORY DATA ANALYSIS
Answers: what does the data look like? How often are rules followed / broken / missing?

Saves charts to outputs/figures/ and a summary table to outputs/eda_summary.csv
"""
import os

import pandas as pd

from style import COLORS, LABELS, PRETTY, BLUE, INK2, plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
FIG = os.path.join(ROOT, "outputs", "figures")


def main():
    os.makedirs(FIG, exist_ok=True)
    rules = pd.read_csv(os.path.join(PROC, "rules.csv"))
    pairs = pd.concat(
        [pd.read_csv(os.path.join(PROC, f"pairs_{s}.csv")).assign(split=s) for s in ["train", "dev", "test"]]
    )
    pairs = pairs.merge(rules[["rule_id", "rule_name"]], on="rule_id")
    docs = pairs.drop_duplicates("doc_id")

    # ---- 1. overall label distribution ------------------------------------------------
    counts = pairs["label"].value_counts().reindex(LABELS)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar([PRETTY[l] for l in LABELS], counts.values, color=[COLORS[l] for l in LABELS], width=0.55)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / counts.sum():.0%})",
                ha="center", va="bottom", color=INK2)
    ax.set_title("How often do documents follow the rules?")
    ax.set_ylabel("(document, rule) pairs")
    ax.set_ylim(0, counts.max() * 1.25)
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(FIG, "01_label_distribution.png"))
    plt.close(fig)

    # ---- 2. per-rule breakdown (which rules are most often broken?) ---------------------
    per_rule = (pd.crosstab(pairs["rule_name"], pairs["label"], normalize="index")
                .reindex(columns=LABELS).sort_values("NON_COMPLIANT"))
    fig, ax = plt.subplots(figsize=(9, 6.5))
    left = pd.Series(0.0, index=per_rule.index)
    for l in LABELS:
        ax.barh(per_rule.index, per_rule[l], left=left, color=COLORS[l], label=PRETTY[l],
                edgecolor="#fcfcfb", linewidth=1.5)
        left += per_rule[l]
    for i, (name, v) in enumerate(per_rule["NON_COMPLIANT"].items()):
        if v >= 0.03:
            ax.text(1.01, i, f"{v:.0%} violated", va="center", color=INK2, fontsize=8.5)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Compliance status for each of the 17 rules")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.13), ncol=3)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelcolor=INK2)
    fig.savefig(os.path.join(FIG, "02_per_rule_breakdown.png"))
    plt.close(fig)

    # ---- 3. document size ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(docs["n_clauses"], bins=40, color=BLUE, edgecolor="#fcfcfb")
    med = docs["n_clauses"].median()
    ax.axvline(med, color=INK2, linestyle="--", linewidth=1)
    ax.text(med * 1.05, ax.get_ylim()[1] * 0.9, f"median = {med:.0f} clauses", color=INK2)
    ax.set_title("How long are the documents?")
    ax.set_xlabel("clauses per document")
    ax.set_ylabel("documents")
    fig.savefig(os.path.join(FIG, "03_document_length.png"))
    plt.close(fig)

    # ---- 4. "needle in a haystack": evidence clauses vs. total clauses -----------------
    ev = pairs[pairs["label"] != "NOT_ADDRESSED"].copy()
    ev["n_evidence"] = ev["evidence_clause_ids"].fillna("").str.split().str.len()
    ev["evidence_share"] = ev["n_evidence"] / ev["n_clauses"]

    summary = pd.DataFrame({
        "metric": [
            "documents", "rules checked per document", "(document, rule) pairs",
            "total clauses", "median clauses per document",
            "compliant pairs", "non-compliant pairs", "not-addressed pairs",
            "avg evidence clauses per decision", "avg % of document that is evidence",
        ],
        "value": [
            docs["doc_id"].nunique(), rules.shape[0], len(pairs),
            int(docs["n_clauses"].sum()), int(med),
            counts["COMPLIANT"], counts["NON_COMPLIANT"], counts["NOT_ADDRESSED"],
            round(ev["n_evidence"].mean(), 2), f"{ev['evidence_share'].mean():.1%}",
        ],
    })
    summary.to_csv(os.path.join(ROOT, "outputs", "eda_summary.csv"), index=False)
    print(summary.to_string(index=False))
    print("\nPer-rule violation rate (top 5):")
    print(per_rule["NON_COMPLIANT"].sort_values(ascending=False).head(5).map("{:.1%}".format).to_string())
    print(f"\nCharts saved to {FIG}")


if __name__ == "__main__":
    main()
