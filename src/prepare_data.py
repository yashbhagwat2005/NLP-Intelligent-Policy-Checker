"""
STEP 1 - DATA PREPARATION
Converts the raw ContractNLI JSON into simple, flat CSV tables.

ContractNLI (Stanford NLP, 2021): 607 real Non-Disclosure Agreements.
For every document and every one of 17 "rules" (hypotheses) a lawyer labelled:
    Entailment     -> the document FOLLOWS the rule        -> COMPLIANT
    Contradiction  -> the document BREAKS the rule         -> NON_COMPLIANT
    NotMentioned   -> the document does not talk about it  -> NOT_ADDRESSED
They also marked WHICH clauses (spans) are the evidence.

Outputs (data/processed/):
    rules.csv               - the 17 rules
    clauses_<split>.csv     - every clause of every document (one row per clause)
    pairs_<split>.csv       - one row per (document, rule) with the compliance label
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "contract-nli")
OUT = os.path.join(ROOT, "data", "processed")

LABEL_MAP = {
    "Entailment": "COMPLIANT",
    "Contradiction": "NON_COMPLIANT",
    "NotMentioned": "NOT_ADDRESSED",
}


def clean(text):
    # the raw text has some broken quote characters; normalise whitespace too
    return " ".join(text.replace("�", '"').split())


def main():
    os.makedirs(OUT, exist_ok=True)
    rules_written = False

    for split in ["train", "dev", "test"]:
        with open(os.path.join(RAW, f"{split}.json"), encoding="utf-8") as f:
            data = json.load(f)

        if not rules_written:
            rules = pd.DataFrame(
                [
                    {"rule_id": k, "rule_name": v["short_description"], "rule_text": v["hypothesis"]}
                    for k, v in data["labels"].items()
                ]
            ).sort_values("rule_id", key=lambda s: s.str.split("-").str[1].astype(int))
            rules.to_csv(os.path.join(OUT, "rules.csv"), index=False)
            rules_written = True

        clause_rows, pair_rows = [], []
        for doc in data["documents"]:
            text = doc["text"]
            clauses = [clean(text[s:e]) for s, e in doc["spans"]]
            for i, c in enumerate(clauses):
                clause_rows.append({"doc_id": doc["id"], "clause_id": i, "clause_text": c})

            for rule_id, ann in doc["annotation_sets"][0]["annotations"].items():
                evidence = " ".join(clauses[i] for i in ann["spans"])
                pair_rows.append(
                    {
                        "doc_id": doc["id"],
                        "rule_id": rule_id,
                        "label": LABEL_MAP[ann["choice"]],
                        "evidence_clause_ids": " ".join(map(str, ann["spans"])),
                        "evidence_text": evidence,
                        "doc_type": doc["document_type"],
                        "n_clauses": len(clauses),
                        "doc_chars": len(text),
                    }
                )

        pd.DataFrame(clause_rows).to_csv(os.path.join(OUT, f"clauses_{split}.csv"), index=False)
        pd.DataFrame(pair_rows).to_csv(os.path.join(OUT, f"pairs_{split}.csv"), index=False)
        print(f"{split:5s}: {len(data['documents']):4d} documents, "
              f"{len(clause_rows):6d} clauses, {len(pair_rows):5d} (document, rule) pairs")

    print(f"\nSaved processed tables to {OUT}")


if __name__ == "__main__":
    main()
