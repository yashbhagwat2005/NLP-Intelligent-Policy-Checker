# Intelligent Policy Compliance Checker

**One line:** the system reads an organisation's policies and its procedure documents. For every rule it says whether the procedures are **Compliant**, **Non-compliant** or **Not addressed**, and it shows the clause that proves it.

Think of it as an **NLP-powered document auditor**.

---

## 0. The web app

```bash
pip install -r requirements.txt
python app.py                # then open http://127.0.0.1:5000
```

The page opens straight away. The sentence-embedding model loads in the background for about a minute, so the **first** policy check waits for it; after that a check takes under a second. Start the app a few minutes before a demo.

The very first run after cloning also downloads the Sentence-BERT model (about 90 MB) into `models/all-MiniLM-L6-v2/`, so it needs an internet connection once. The trained contract model (`models/contractnli_pipeline.joblib`) is included in the repository.

| Page | What it does |
|---|---|
| **Policy check** | Upload or paste a policy, upload procedure documents (`.txt`, `.pdf`, `.docx`), get a report. "Run the sample company" uses the NovaTech files. |
| **Contract check** | Upload one NDA. It is checked against the 17 ContractNLI rules with the trained two-stage model from `src/train_models.py`. |
| **Method** | Explains both checks and shows the test results. |

The report lists every rule with its status, the evidence sentence, the source document and the reason. It can be filtered by status, downloaded as CSV, or printed.

| File | What it does |
|---|---|
| `app.py` | Flask routes: forms, running a check, report page, CSV download |
| `checker/readers.py` | Reads `.txt`, `.pdf` and `.docx` uploads into plain text |
| `checker/policy_parser.py` | Turns a policy document into rules and works out each rule's type (min, max, obligation, prohibition) from its wording |
| `checker/engine.py` | Policy check: splits documents into sentences, finds the best sentence per rule, judges it |
| `checker/contract_model.py` | Contract check: loads `models/contractnli_pipeline.joblib` and runs Stage A + Stage B |
| `templates/`, `static/style.css` | The pages (plain HTML and CSS, about 25 lines of JavaScript) |
| `data/samples/` | Sample policy handbook and sample NDA used by the "Run the sample" buttons |

**What changed in the policy check since the mid-term (step 4):**

- **No hand-written hints.** At mid-term each rule had keywords and a check type typed in by us (`policies.json`). Now the rule type is detected from the wording and no keywords are used, so any policy document works.
- **Hybrid search.** The best sentence is found with the average of TF-IDF similarity (word overlap) and Sentence-BERT similarity (`all-MiniLM-L6-v2`, meaning). This was the planned fix for wording differences such as "revoked" vs "disabled".
- **Weak matches are never proof of compliance.** If the best sentence scores below 0.40, a "Compliant" verdict becomes "Not addressed" and the row is marked *review*. A possible violation on a weak match is still shown, also marked *review*.

**Result on NovaTech: 18 of 19 rules correct**, with no hand-written keywords or rule types. The one error is ACC-02 (quarterly access reviews): the right sentence is found, but its score of 0.35 is below 0.40, so it is reported as "Not addressed, review" and not as Compliant. Two things to say honestly: ACC-03 ("shared accounts" vs "common login") gets the right label from the wrong sentence, and the two thresholds (0.22 and 0.40) were chosen on these same 19 rules, so this is a demonstration, not a benchmark.

The contract check, run on raw text with the app's own sentence splitter, scores **82.1% accuracy on the 123 test contracts** (81.2% with the dataset's own clause boundaries).

**Known limits:** money amounts and percentages are not compared (only time periods and character counts); a prohibition can be flagged from a sentence that is only on a related topic; scanned PDFs are not read; reports are kept in memory and disappear when the app stops.

Sections 1–7 below describe the mid-term work (data, EDA, the trained pipeline and its results).

---

## 1. How to run the mid-term pipeline

```bash
pip install -r requirements.txt
python run_all.py            # runs all 4 steps (about 3–5 min)
```
Then open `midterm_demo.ipynb` (the presentation walkthrough) or `outputs/compliance_report.html` (the audit report).

| File | What it does |
|---|---|
| `src/prepare_data.py` | Step 1: converts the raw dataset into simple CSV tables |
| `src/eda.py` | Step 2: analyses the data and makes the charts |
| `src/train_models.py` | Step 3: trains and evaluates the 2-stage ML pipeline |
| `src/compliance_checker.py` | Step 4: end-to-end demo on company policies and the HTML report |
| `midterm_demo.ipynb` | Everything above in one notebook, already executed |

---

## 2. Datasets

### (a) ContractNLI — real, benchmark data (Stanford NLP, 2021)
- **607 real legal agreements** (NDAs), **48,058 clauses**, **17 rules**, which gives 10,319 (document, rule) pairs
- Lawyers labelled every (document, rule) pair, and we map their labels to ours:

| Dataset label | Our label | Meaning |
|---|---|---|
| Entailment | **Compliant** | the document follows the rule |
| Contradiction | **Non-compliant** | the document breaks the rule |
| NotMentioned | **Not addressed** | the document is silent on the rule |

- They also marked **which clauses are the evidence**. That lets us train and test the "find the evidence" step too.
- Split: 423 train / 61 dev / 123 test documents. All results below are on the **test** documents the model never saw.

### (b) NovaTech demo corpus — created by us
A fictional company with **19 policy rules** in the 6 areas from our proposal (Information Security, Employee Privacy, Data Retention, Workplace Safety, Financial Procedures, Access Control). It also has **5 department procedure documents** (IT, HR, Finance, Facilities, Warehouse). We hand-labelled the correct answer for every rule in `data/custom/ground_truth.csv`.

---

## 3. What the data told us (EDA)

- 49% of pairs are compliant, 11% non-compliant and 40% not addressed. **Non-compliance is the rare class**, which makes it the hardest to learn.
- Some rules are violated far more often than others (chart `02_per_rule_breakdown.png`).
- **Key insight:** the evidence for a rule is only **~2.7% of a document** (about 2 clauses out of about 70). The system has to **find** the right clauses before it can **decide**. That is why we use two stages.

---

## 4. The pipeline

```
 Policy rule ─┐
              ▼
 Document ──► [ Stage A: EVIDENCE FINDER ] ──► top-3 clauses ──► [ Stage B: COMPLIANCE JUDGE ] ──► label + evidence
              "where does the doc talk                            "do these clauses follow,
               about this rule?"                                   break, or ignore the rule?"
```

**Stage A — Evidence Finder.** Each clause is turned into numbers with **TF-IDF** (important words get high weight). A **Logistic Regression** model per rule learns what the evidence for that rule looks like.

**Stage B — Compliance Judge.** It takes the top-3 clauses from Stage A, turns them into TF-IDF features, and a Logistic Regression per rule predicts one of the 3 labels.

### Results on the 123 unseen test documents

**Stage A — Recall@k (% of the true evidence clauses found in the top k)**

| Method | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| Random (sanity check) | 1.2% | 4.3% | 7.5% | 14.6% |
| TF-IDF similarity (no training) | 13.4% | 28.8% | 39.9% | 54.3% |
| **Supervised Logistic Regression (ours)** | **42.8%** | **73.3%** | **81.1%** | **88.7%** |

**Stage B — Compliance classification**

| Model | Accuracy | Macro-F1 | F1 Compliant | F1 Non-compliant | F1 Not addressed |
|---|---|---|---|---|---|
| Always predict the most common label | 0.463 | 0.211 | 0.633 | 0.000 | 0.000 |
| Most common label per rule | 0.659 | 0.617 | 0.730 | 0.506 | 0.614 |
| **Retrieval + Logistic Regression (ours)** | **0.812** | **0.767** | **0.859** | **0.637** | **0.805** |

**How to say it:** "Our model reaches **81% accuracy**, compared with 66% for the best simple baseline. Within the top 3 clauses it finds **73% of the evidence a lawyer marked**, against 29% for plain keyword similarity."

> **Why macro-F1?** Accuracy can look good by ignoring the rare non-compliant class. Macro-F1 averages the F1 score of all 3 classes equally, so the rare class counts just as much.

---

## 5. End-to-end demo (company policies)

`src/compliance_checker.py` does 4 things for every rule:
1. **Split** each procedure document into clauses.
2. **Retrieve** the most similar clause. It uses character-level TF-IDF, so "report" matches "reported". If the best similarity is below 0.20, the rule is **Not addressed**.
3. **Judge** the clause:
   - **Numeric rules** ("at least 12 characters", "within 24 hours"): extract the number and unit, convert units (for example "quarterly" becomes 3 months), and compare.
   - **Obligations** ("must be enabled"): weak words like *optional*, *encouraged* or *not mandatory* mean **Non-compliant**.
   - **Prohibitions** ("must not be shared"): the clause must contain a restriction (*not*, *never*, *prohibited*); otherwise it is **Non-compliant**.
4. **Explain** the result: show the evidence clause, the document it came from, and the reason.

**Result: 17 of 19 rules correct (89.5% accuracy, macro-F1 0.85).** Example findings:

| Rule | Procedure says | Verdict |
|---|---|---|
| Passwords ≥ 12 characters | "minimum of 8 characters" | Non-compliant |
| MFA for remote access | "MFA is optional for VPN users" | Non-compliant |
| CCTV kept ≤ 30 days | "stored … for 90 days" | Non-compliant |
| Fire drill every 6 months | "held once a year" | Non-compliant |
| Financial records ≥ 8 years | "retained for 10 years" | Compliant |
| Vendor anti-bribery checks | nothing in any document | Not addressed |

---

## 6. Honest limitations and plan after mid-term

The 2 errors in the demo show exactly what to build next:

| Error | Why it happened | Fix (next phase) |
|---|---|---|
| ACC-03 "shared accounts must not be used" → said *Not addressed* | The procedure says "**common login**". TF-IDF matches words, not meaning. | **Sentence embeddings** (Sentence-BERT) for meaning-based search |
| RET-03 "backups must be **tested for restoration**" → said *Compliant* | It found the backup clause, but that clause never mentions testing. | A **transformer NLI model** (for example DeBERTa fine-tuned on ContractNLI) that checks whether the clause actually implies the rule |

Other next steps:
- Improve the non-compliant class (F1 0.64) with class balancing or data augmentation.
- Accept PDF and DOCX uploads.
- Build a simple web interface.
- Compare against transformer baselines on ContractNLI.

---

## 7. Likely viva questions

- **Why two stages?** The evidence is only about 3% of a document. Feeding the whole document to a classifier adds a lot of noise, so we find the evidence first and then judge it.
- **Why TF-IDF + Logistic Regression and not deep learning yet?** It is fast, explainable (we can see which words drive a decision) and a strong baseline. Transformers come next and will be measured against it.
- **What is Recall@k?** Out of all the clauses a lawyer marked as evidence, the share that shows up in our top k.
- **Is the demo data real?** The ML results use the real ContractNLI dataset. The company demo is a fictional corpus we wrote to show the full workflow on the policy areas from our proposal.
- **Is there any data leakage?** The test documents are never used for training. One thing to note: Stage B is trained on Stage A's top clauses for the *training* documents, scored by the same model, which is slightly optimistic. Cross-fitting is planned.
