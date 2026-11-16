"""
STEP 4 - END-TO-END DEMO: check a company's procedures against its policies

Input : data/custom/policies.json        (the rules the company must follow)
        data/custom/procedures/*.txt     (what each department actually documents)
Output: outputs/compliance_report.html   (readable audit report)
        outputs/compliance_results.csv
        + accuracy against data/custom/ground_truth.csv (labelled by us)

How it decides, for every rule:
  1. SPLIT    every procedure document into clauses (one numbered point / sentence each)
  2. RETRIEVE the clause most similar to the rule (TF-IDF, character n-grams so that
              "report" ~ "reported", "password" ~ "passwords")
              -> if even the best clause is not similar enough  => NOT_ADDRESSED
  3. JUDGE    the evidence clause
              - numeric rules  ("at least 12 characters", "within 24 hours"):
                    extract the number + unit from the clause, convert units, compare
              - obligations    ("must be enabled"): flag weak wording such as
                    "optional", "not mandatory", "encouraged"      => NON_COMPLIANT
              - prohibitions   ("must not be shared"): the clause must contain a
                    negation ("not", "never", "prohibited"), else  => NON_COMPLIANT
  4. EXPLAIN  every decision with the evidence clause, the document it came from, and a reason
"""
import glob
import html
import json
import os
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

from style import COLORS, LABELS, PRETTY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOM = os.path.join(ROOT, "data", "custom")
OUT = os.path.join(ROOT, "outputs")

SIMILARITY_THRESHOLD = 0.20   # below this, we say the rule is not addressed anywhere

TO_DAYS = {"hour": 1 / 24, "day": 1, "week": 7, "month": 30, "year": 365}
FREQUENCY_WORDS = {  # phrase -> (value, unit)
    r"once a year|annually|yearly|every year": (1, "year"),
    r"quarterly|every quarter": (3, "month"),
    r"monthly|every month": (1, "month"),
    r"weekly|every week": (1, "week"),
    r"same day|last working day|immediately": (0, "day"),
}
WEAK_WORDS = r"\b(optional|not mandatory|not required|encouraged|at (their|the) discretion|may choose|if possible|recommended)\b"
NEGATION_WORDS = r"\b(not|never|no|prohibited|forbidden|banned|disallowed|without)\b"


def split_clauses(path):
    doc = os.path.basename(path).replace(".txt", "").replace("_", " ")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = re.sub(r"^\s*\d+\.\s*", "", line).strip()   # drop "3. " numbering
            if not line or line.isupper():                      # skip blank lines & titles
                continue
            for sent in re.split(r"(?<=[.!?])\s+", line):
                if len(sent.split()) >= 4:
                    rows.append({"document": doc, "clause": sent})
    return rows


def extract_quantity(clause, unit):
    """Return (value, unit, matched_text) for the quantity in the clause, or None."""
    text = clause.lower()
    if unit == "characters":
        m = re.search(r"(\d+)\s*(characters|chars)", text)
        return (int(m.group(1)), "characters", m.group(0)) if m else None
    m = re.search(r"(\d[\d,]*)\s*(hour|day|week|month|year)s?", text)
    if m:
        return int(m.group(1).replace(",", "")), m.group(2), m.group(0)
    for pattern, (v, u) in FREQUENCY_WORDS.items():
        m = re.search(pattern, text)
        if m:
            return v, u, m.group(0)
    return None


def judge(rule, clause):
    check = rule["check"]
    ctype = check["type"]

    if ctype in ("min", "max"):
        q = extract_quantity(clause, check["unit"])
        if q is not None:
            val, unit, found = q
            if check["unit"] == "characters":
                have, need = val, check["value"]
            else:
                have = val * TO_DAYS[unit]
                need = check["value"] * TO_DAYS[check["unit"].rstrip("s")]
            ok = have >= need if ctype == "min" else have <= need
            word = "at least" if ctype == "min" else "at most"
            reason = (f'Procedure says "{found}"; policy requires {word} '
                      f'{check["value"]} {check["unit"]}.')
            return ("COMPLIANT" if ok else "NON_COMPLIANT"), reason
        ctype = "obligation"   # no number found -> fall back to wording check

    if ctype == "obligation":
        m = re.search(WEAK_WORDS, clause.lower())
        if m:
            return "NON_COMPLIANT", f'Mandatory rule, but procedure uses weak wording: "{m.group(0)}".'
        return "COMPLIANT", "Procedure contains a matching requirement with no weakening language."

    if ctype == "prohibition":
        m = re.search(NEGATION_WORDS, clause.lower())
        if m:
            return "COMPLIANT", f'Procedure restricts the activity ("{m.group(0)}").'
        return "NON_COMPLIANT", "Policy forbids this, but the procedure describes doing it without restriction."

    raise ValueError(ctype)


def check_compliance(policies, clauses):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    queries = [r["text"] + " " + r["keywords"] for r in policies["rules"]]
    vec.fit([c["clause"] for c in clauses] + queries)
    C = vec.transform([c["clause"] for c in clauses])
    sims = cosine_similarity(vec.transform(queries), C)

    results = []
    for rule, row in zip(policies["rules"], sims):
        best = row.argmax()
        ev = clauses[best]
        if row[best] < SIMILARITY_THRESHOLD:
            label = "NOT_ADDRESSED"
            reason = f"No procedure clause is similar enough (best similarity {row[best]:.2f} < {SIMILARITY_THRESHOLD})."
        else:
            label, reason = judge(rule, ev["clause"])
        results.append({
            "rule_id": rule["id"], "domain": rule["domain"], "rule": rule["text"],
            "prediction": label, "similarity": round(float(row[best]), 3),
            "evidence_document": ev["document"], "evidence": ev["clause"], "reason": reason,
        })
    return pd.DataFrame(results)


def write_html(df, org, acc, f1, n_docs):
    counts = df.prediction.value_counts().reindex(LABELS, fill_value=0)
    score = counts["COMPLIANT"] / len(df)
    cards = "".join(
        f'<div class="card"><div class="num" style="color:{COLORS[l]}">{counts[l]}</div>'
        f'<div class="lbl">{PRETTY[l]}</div></div>' for l in LABELS)
    domain_rows = ""
    for dom, g in df.groupby("domain", sort=False):
        c = g.prediction.value_counts()
        domain_rows += (f"<tr><td>{dom}</td><td>{len(g)}</td>"
                        + "".join(f"<td>{c.get(l, 0)}</td>" for l in LABELS) + "</tr>")
    rows = ""
    for r in df.itertuples():
        correct = "&#10003;" if r.prediction == r.ground_truth else f"&#10007; (truth: {PRETTY[r.ground_truth]})"
        ev = "&mdash;" if r.prediction == "NOT_ADDRESSED" else (
            f'<q>{html.escape(r.evidence)}</q><div class="src">{html.escape(r.evidence_document)} '
            f'&middot; similarity {r.similarity:.2f}</div>')
        rows += (f'<tr><td><b>{r.rule_id}</b><div class="src">{r.domain}</div></td>'
                 f"<td>{html.escape(r.rule)}</td>"
                 f'<td><span class="pill" style="background:{COLORS[r.prediction]}">{PRETTY[r.prediction]}</span></td>'
                 f"<td>{ev}</td><td>{html.escape(r.reason)}</td><td>{correct}</td></tr>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Compliance Report</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f9f9f7;color:#0b0b0b}}
h1{{margin:0}} .sub{{color:#52514e;margin:4px 0 20px}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}}
.card{{background:#fff;border:1px solid #e1e0d9;border-radius:10px;padding:14px 20px;min-width:130px}}
.num{{font-size:30px;font-weight:700}} .lbl{{color:#52514e;font-size:13px}}
table{{border-collapse:collapse;width:100%;background:#fff;margin-bottom:24px;font-size:14px}}
th,td{{border-bottom:1px solid #e1e0d9;padding:8px 10px;text-align:left;vertical-align:top}}
th{{background:#f0efec}} q{{font-style:italic}} .src{{color:#898781;font-size:12px;margin-top:3px}}
.pill{{color:#fff;border-radius:999px;padding:3px 10px;font-size:12px;white-space:nowrap}}
</style></head><body>
<h1>Policy Compliance Report</h1>
<div class="sub">{html.escape(org)} &middot; {len(df)} rules checked against
{n_docs} procedure documents</div>
<div class="cards">
<div class="card"><div class="num">{score:.0%}</div><div class="lbl">Compliance score</div></div>{cards}
<div class="card"><div class="num">{acc:.0%}</div><div class="lbl">Checker accuracy vs. our labels<br>(macro-F1 {f1:.2f})</div></div>
</div>
<h3>By policy area</h3>
<table><tr><th>Area</th><th>Rules</th>{"".join(f"<th>{PRETTY[l]}</th>" for l in LABELS)}</tr>{domain_rows}</table>
<h3>Rule-by-rule findings</h3>
<table><tr><th>Rule</th><th>Policy requirement</th><th>Status</th><th>Evidence found</th><th>Why</th><th>Correct?</th></tr>
{rows}</table></body></html>"""
    with open(os.path.join(OUT, "compliance_report.html"), "w", encoding="utf-8") as f:
        f.write(page)


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(CUSTOM, "policies.json"), encoding="utf-8") as f:
        policies = json.load(f)
    clauses = []
    paths = sorted(glob.glob(os.path.join(CUSTOM, "procedures", "*.txt")))
    for path in paths:
        clauses += split_clauses(path)
    print(f"Loaded {len(policies['rules'])} rules and {len(clauses)} procedure clauses\n")

    df = check_compliance(policies, clauses)
    truth = pd.read_csv(os.path.join(CUSTOM, "ground_truth.csv"))
    df = df.merge(truth[["rule_id", "label"]].rename(columns={"label": "ground_truth"}), on="rule_id")
    acc = accuracy_score(df.ground_truth, df.prediction)
    f1 = f1_score(df.ground_truth, df.prediction, average="macro", labels=LABELS)

    for r in df.itertuples():
        mark = "OK " if r.prediction == r.ground_truth else "XX "
        print(f"{mark}{r.rule_id:7s} {PRETTY[r.prediction]:14s} sim={r.similarity:.2f}  {r.reason}")
    print(f"\nAccuracy vs ground truth: {acc:.1%}   macro-F1: {f1:.3f}")

    df.to_csv(os.path.join(OUT, "compliance_results.csv"), index=False)
    write_html(df, policies["organisation"], acc, f1, len(paths))
    print(f"Report written to {os.path.join(OUT, 'compliance_report.html')}")


if __name__ == "__main__":
    main()
