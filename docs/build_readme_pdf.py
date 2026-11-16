"""
Builds README.pdf: a project report that explains the problem, what we built,
the maths behind every component, and the results.

    python docs/build_readme_pdf.py

Reads metrics/figures from outputs/ and the demo data from data/custom/,
so re-running after run_all.py keeps the numbers in sync.
"""
import json
import os
import tempfile
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image as PILImage
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(OUT, "figures")
CUSTOM = os.path.join(ROOT, "data", "custom")
PDF_PATH = os.environ.get("README_PDF", os.path.join(ROOT, "README.pdf"))
TMP = tempfile.mkdtemp(prefix="readme_eq_")

# ---------------------------------------------------------------- fonts & colours
FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
for name, file in [("Body", "DejaVuSans.ttf"), ("Body-Bold", "DejaVuSans-Bold.ttf"),
                   ("Body-Italic", "DejaVuSans-Oblique.ttf"), ("Body-BoldItalic", "DejaVuSans-BoldOblique.ttf"),
                   ("Mono", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, os.path.join(FONT_DIR, file)))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic",
                              boldItalic="Body-BoldItalic")

NAVY = colors.HexColor("#1f3a5f")
BLUE = colors.HexColor("#2f6db5")
INK = colors.HexColor("#1b1b1b")
MUTED = colors.HexColor("#5f5e5a")
RULE = colors.HexColor("#d9d7cf")
TINT = colors.HexColor("#eef3f9")
ZEBRA = colors.HexColor("#f7f7f4")
GREEN = colors.HexColor("#2e7d4f")
RED = colors.HexColor("#b3402f")
AMBER = colors.HexColor("#9a6a12")
LABEL_COL = {"COMPLIANT": GREEN, "NON_COMPLIANT": RED, "NOT_ADDRESSED": AMBER}
PRETTY = {"COMPLIANT": "Compliant", "NON_COMPLIANT": "Non-compliant", "NOT_ADDRESSED": "Not addressed"}

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

# ---------------------------------------------------------------- styles
S = {
    "title": ParagraphStyle("title", fontName="Body-Bold", fontSize=22, leading=27, textColor=NAVY),
    "subtitle": ParagraphStyle("subtitle", fontName="Body", fontSize=11.5, leading=16, textColor=MUTED),
    "h1": ParagraphStyle("h1", fontName="Body-Bold", fontSize=15, leading=19, textColor=NAVY,
                         spaceBefore=14, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=11.5, leading=15, textColor=BLUE,
                         spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Body", fontSize=9.4, leading=13.6, textColor=INK, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=9.4, leading=13.4, textColor=INK,
                             leftIndent=13, bulletIndent=3, spaceAfter=2.5),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.2, leading=10.6, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Body-Bold", fontSize=8.2, leading=10.6, textColor=INK),
    "small": ParagraphStyle("small", fontName="Body", fontSize=7.4, leading=9.4, textColor=INK),
    "head": ParagraphStyle("head", fontName="Body-Bold", fontSize=8.2, leading=10.6, textColor=colors.white),
    "caption": ParagraphStyle("caption", fontName="Body-Italic", fontSize=8.2, leading=11, textColor=MUTED,
                              alignment=TA_CENTER, spaceBefore=3, spaceAfter=8),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8.2, leading=11, textColor=INK,
                           backColor=ZEBRA, borderPadding=6, leftIndent=6, rightIndent=6,
                           spaceBefore=4, spaceAfter=10),
    "callout": ParagraphStyle("callout", fontName="Body", fontSize=9.4, leading=13.6, textColor=INK,
                              backColor=TINT, borderPadding=8, leftIndent=8, rightIndent=8,
                              spaceBefore=6, spaceAfter=12),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items, style="bullet"):
    return [Paragraph(t, S[style], bulletText="•") for t in items]


def table(rows, widths, header=True, zebra=True, style="cell", align_right=()):
    """rows: list of lists of str (str is wrapped in a Paragraph)."""
    data = []
    for i, row in enumerate(rows):
        st = "head" if (header and i == 0) else style
        data.append([c if not isinstance(c, str) else Paragraph(c, S[st]) for c in row])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
    ]
    if header:
        cmds.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    if zebra:
        for r in range(1 if header else 0, len(rows)):
            if r % 2 == 0:
                cmds.append(("BACKGROUND", (0, r), (-1, r), ZEBRA))
    t.setStyle(TableStyle(cmds))
    return t


def highlight_row(t, row, color=TINT):
    t.setStyle(TableStyle([("BACKGROUND", (0, row), (-1, row), color)]))
    return t


_eq_n = [0]


def eq(latex, size=12.5):
    """Render a LaTeX (matplotlib mathtext) formula to a PNG and return a centred Image."""
    _eq_n[0] += 1
    path = os.path.join(TMP, f"eq{_eq_n[0]:02d}.png")
    fig = plt.figure(figsize=(0.01, 0.01))
    latex = latex.replace(r"\frac", r"\dfrac")   # display-size fractions
    fig.text(0, 0, f"${latex}$", fontsize=size, color="#1b1b1b")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)
    w, h = PILImage.open(path).size
    w_pt, h_pt = w / 300 * 72, h / 300 * 72
    if w_pt > CONTENT_W - 20:
        h_pt *= (CONTENT_W - 20) / w_pt
        w_pt = CONTENT_W - 20
    img = Image(path, width=w_pt, height=h_pt)
    img.hAlign = "CENTER"
    return KeepTogether([Spacer(1, 3), img, Spacer(1, 6)])


def figure(name, caption, width=CONTENT_W * 0.78):
    path = os.path.join(FIG, name)
    w, h = PILImage.open(path).size
    img = Image(path, width=width, height=width * h / w)
    img.hAlign = "CENTER"
    return KeepTogether([img, P(caption, "caption")])


def figure_pair(a, b, caption):
    cells = []
    for name in (a, b):
        path = os.path.join(FIG, name)
        w, h = PILImage.open(path).size
        width = CONTENT_W / 2 - 8
        cells.append(Image(path, width=width, height=width * h / w))
    t = Table([cells], colWidths=[CONTENT_W / 2] * 2)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    return KeepTogether([t, P(caption, "caption")])


# ---------------------------------------------------------------- pipeline diagram
def pipeline_diagram():
    W, H = CONTENT_W, 128
    d = Drawing(W, H)

    def box(x, y, w, h, lines, fill, stroke, bold_first=True, text_color=INK, size=7.6):
        d.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=fill, strokeColor=stroke, strokeWidth=0.9))
        lh = size + 2.6
        top = y + h / 2 + (len(lines) - 1) * lh / 2 - size / 3
        for i, ln in enumerate(lines):
            font = "Body-Bold" if (bold_first and i == 0) else "Body"
            d.add(String(x + w / 2, top - i * lh, ln, fontName=font, fontSize=size,
                         fillColor=text_color, textAnchor="middle"))

    def arrow(x1, y1, x2, y2):
        d.add(Line(x1, y1, x2 - 4, y2, strokeColor=MUTED, strokeWidth=1.1))
        d.add(Polygon([x2, y2, x2 - 6, y2 + 3.2, x2 - 6, y2 - 3.2], fillColor=MUTED, strokeColor=MUTED))

    pale = colors.HexColor("#f4f3ee")
    blue_fill = colors.HexColor("#e3edf8")
    # inputs
    box(0, 76, 78, 40, ["Policy rule  r", "e.g. “shall not", "reverse engineer”"], pale, RULE)
    box(0, 12, 78, 40, ["Document  D", "split into", "~80 clauses"], pale, RULE)
    # stage A
    ax, aw = 98, 104
    box(ax, 22, aw, 84, ["STAGE A", "Evidence Finder", "", "TF-IDF features", "+ Logistic Regression", "scores every clause",
                         "score = P(evidence)"], blue_fill, BLUE)
    arrow(78, 96, ax, 76)
    arrow(78, 32, ax, 52)
    # top-k
    tx, tw = ax + aw + 18, 64
    box(tx, 42, tw, 44, ["Top-3", "clauses", "+ best score"], pale, RULE)
    arrow(ax + aw, 64, tx, 64)
    # stage B
    bx, bw = tx + tw + 18, 104
    box(bx, 22, bw, 84, ["STAGE B", "Compliance Judge", "", "TF-IDF of evidence", "+ multinomial", "Logistic Regression",
                         "3-way decision"], blue_fill, BLUE)
    arrow(tx + tw, 64, bx, 64)
    # output
    ox = bx + bw + 18
    ow = W - ox
    box(ox, 30, ow, 68, ["OUTPUT", "label", "+ evidence", "clauses"], colors.HexColor("#e6f2ea"), GREEN)
    arrow(bx + bw, 64, ox, 64)
    return d


# ---------------------------------------------------------------- page decoration
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, PAGE_H - 13 * mm, PAGE_W - MARGIN, PAGE_H - 13 * mm)
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, PAGE_H - 11 * mm, "Intelligent Policy Compliance Checker")
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 11 * mm, "Project README & Technical Report")
    canvas.line(MARGIN, 13 * mm, PAGE_W - MARGIN, 13 * mm)
    canvas.drawRightString(PAGE_W - MARGIN, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


# ---------------------------------------------------------------- web platform numbers
def platform_results(truth, hand_checks):
    """Run the web app's own engine so the numbers in Section 11 are measured, not typed."""
    import glob
    import sys
    sys.path.insert(0, ROOT)
    from checker.contract_model import check_contract
    from checker.engine import check_policies, split_clauses
    from checker.policy_parser import parse_policy

    def read(path):
        with open(path, encoding="utf-8") as f:
            return f.read()

    # policy check on the NovaTech sample, using only the policy text
    rules = parse_policy(read(os.path.join(ROOT, "data", "samples", "NovaTech_Policy_Handbook.txt")))
    clauses = []
    for path in sorted(glob.glob(os.path.join(CUSTOM, "procedures", "*.txt"))):
        clauses += split_clauses(read(path), os.path.basename(path)[:-4].replace("_", " "))
    rows = check_policies(rules, clauses)
    type_match = 0
    for row, rule in zip(rows, rules):
        row["check"], row["truth"] = rule["check"], truth[row["rule_id"]]
        hand = hand_checks[row["rule_id"]]
        type_match += rule["check"]["type"] == hand["type"] and rule["check"].get("value") == hand.get("value")

    # contract check on the raw text of the 123 test contracts
    label = {"Entailment": "COMPLIANT", "Contradiction": "NON_COMPLIANT", "NotMentioned": "NOT_ADDRESSED"}
    test = json.loads(read(os.path.join(ROOT, "data", "raw", "contract-nli", "test.json")))
    hits = total = 0
    for doc in test["documents"]:
        gold = {k.upper(): label[a["choice"]] for k, a in doc["annotation_sets"][0]["annotations"].items()}
        predicted, _ = check_contract(doc["text"], "contract")
        hits += sum(r["status"] == gold[r["rule_id"]] for r in predicted)
        total += len(predicted)

    return {"rows": rows, "correct": sum(r["status"] == r["truth"] for r in rows), "type_match": type_match,
            "contract_acc": hits / total}


# ---------------------------------------------------------------- content
def build():
    metrics = json.load(open(os.path.join(OUT, "metrics.json"), encoding="utf-8"))
    eda = pd.read_csv(os.path.join(OUT, "eda_summary.csv")).set_index("metric")["value"]
    per_rule = pd.read_csv(os.path.join(OUT, "stageB_per_rule_accuracy.csv"))
    demo = pd.read_csv(os.path.join(OUT, "compliance_results.csv"))
    policies = json.load(open(os.path.join(CUSTOM, "policies.json"), encoding="utf-8"))
    rule_check = {r["id"]: r["check"] for r in policies["rules"]}
    demo_acc = (demo.prediction == demo.ground_truth).mean()

    A = metrics["stageA_recall"]
    B = metrics["stageB"]
    a2, b2 = A["A2 Supervised LogReg"], B["B2 Retrieval + LogReg"]

    plat = platform_results(demo.set_index("rule_id").ground_truth.to_dict(), rule_check)
    story = []

    # ---------------- title block
    story += [
        Spacer(1, 6),
        P("Intelligent Policy Compliance Checker", "title"),
        Spacer(1, 4),
        P("Project README &amp; Technical Report — Natural Language Processing course project",
          "subtitle"),
        Spacer(1, 10),
    ]
    summary = table([
        ["<b>What it does</b>", "Reads an organisation’s policy rules and its procedure documents. For every rule it "
                         "decides <b>Compliant</b>, <b>Non-compliant</b> or <b>Not addressed</b>, and points to the "
                         "clause that proves it."],
        ["<b>Approach</b>", "Two-stage NLP pipeline: an <b>evidence finder</b> (TF-IDF + Logistic Regression) ranks "
                     "clauses, then a <b>compliance judge</b> (multinomial Logistic Regression) labels the rule."],
        ["<b>Data</b>", "ContractNLI (607 real NDAs, 48,058 clauses, 17 rules, lawyer labels) + NovaTech, a fictional "
                 "company corpus we wrote (19 rules, 5 procedure documents)."],
        ["<b>Headline results</b>", f"<b>{b2['accuracy']:.1%}</b> accuracy and <b>{b2['macro_F1']:.3f}</b> macro-F1 on "
                             f"123 unseen contracts (best baseline 65.9%). Finds <b>{a2['Recall@3']:.1%}</b> of "
                             f"lawyer-marked evidence in the top 3 clauses. Company demo: "
                             f"<b>{int(round(demo_acc * len(demo)))}/{len(demo)}</b> rules correct."],
        ["<b>Web platform</b>", "A Flask web app built after the mid-term: upload a policy and procedure documents "
                                "(.txt, .pdf, .docx) and get an evidence-backed report. Rule types are detected "
                                "automatically and search combines TF-IDF with Sentence-BERT. Company demo without "
                                f"any hand-written hints: <b>{plat['correct']}/{len(plat['rows'])}</b> rules correct."],
    ], [32 * mm, CONTENT_W - 32 * mm], header=False, zebra=False)
    summary.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), TINT)]))
    story += [summary, Spacer(1, 6)]

    story.append(P("Contents", "h2"))
    toc = ["1. Problem statement", "2. What we did (summary of work)", "3. Datasets", "4. Exploratory data analysis",
           "5. System architecture", "6. Technical &amp; mathematical foundations", "7. Results on ContractNLI",
           "8. End-to-end demo on company policies", "9. Error analysis &amp; limitations",
           "10. Plan after mid-term (and status)", "11. Web platform (built after the mid-term)",
           "12. How to run &amp; project structure", "Appendix A–B", ""]
    half = (len(toc) + 1) // 2
    story.append(table([[P(toc[i], "cell"), P(toc[i + half], "cell") if i + half < len(toc) else ""]
                        for i in range(half)], [CONTENT_W / 2] * 2, header=False, zebra=False))

    # ---------------- 1. problem statement
    story.append(P("1. Problem statement", "h1"))
    story.append(P(
        "Every organisation writes <b>policies</b> (what must happen: “passwords must be at least 12 "
        "characters”) and every department writes <b>procedures</b> (what actually happens: “passwords "
        "must have a minimum of 8 characters”). Checking that the procedures obey the policies is called a "
        "<b>compliance audit</b>. Today it is done by hand: an auditor reads every document, finds the relevant "
        "sentence for each rule, and decides whether it follows the rule. This is slow, expensive, inconsistent "
        "between auditors, and it does not scale when there are hundreds of rules and documents that change often."))
    story.append(P(
        "The goal of this project is to <b>automate the first pass of a compliance audit with NLP</b>. The hard "
        "parts are linguistic:"))
    story += bullets([
        "<b>Needle in a haystack.</b> The evidence for a rule is usually one or two sentences inside a long document "
        f"(only about {eda['avg % of document that is evidence']} of a contract).",
        "<b>Same meaning, different words.</b> A policy says “shared accounts”; the procedure says "
        "“common login”.",
        "<b>Small words flip the answer.</b> “MFA is <i>mandatory</i>” vs “MFA is <i>optional</i>”; "
        "“data <i>may</i> be shared” vs “data <i>shall not</i> be shared”.",
        "<b>Numbers and units.</b> “within 24 hours” vs “within 4 hours”, “every 6 months” vs "
        "“once a year”.",
        "<b>Silence is an answer.</b> If no document mentions a rule, the correct output is <i>Not addressed</i>, "
        "not a guess.",
        "<b>Explainability.</b> An auditor will not trust a bare label; the system must show the clause it used.",
    ])
    story.append(P("1.1 Formal definition", "h2"))
    story.append(P(
        "Let <i>R</i> = {<i>r</i><sub>1</sub>, …, <i>r</i><sub>m</sub>} be a set of policy rules and let a document "
        "be a sequence of clauses <i>D</i> = (<i>c</i><sub>1</sub>, …, <i>c</i><sub>n</sub>). For every pair "
        "(<i>D</i>, <i>r</i>) the system must output a label and a small evidence set:"))
    story.append(eq(r"f(D,\,r) \;=\; (\hat{y},\ E), \qquad \hat{y}\in\{\mathrm{C},\ \mathrm{NC},\ \mathrm{NA}\},\qquad E\subseteq D,\ |E| = k"))
    story.append(P("where C = Compliant, NC = Non-compliant and NA = Not addressed, and <i>E</i> is the set of "
                   "<i>k</i> clauses used as evidence."))
    story.append(P(
        "Because almost all clauses are irrelevant to a given rule, we <b>factor</b> the problem into two easier ones: "
        "first find the evidence, then judge only the evidence."))
    story.append(eq(r"P(y\mid D,\,r)\ \approx\ P(y\mid E_k(D,r),\,r), \qquad "
                    r"E_k(D,r)=\mathrm{the\ }k\mathrm{\ clauses\ of\ }D\mathrm{\ with\ the\ highest\ relevance\ }s_r(c)"))
    story.append(P("1.2 Objectives", "h2"))
    story += bullets([
        "Map each policy rule to the clause(s) that address it (<b>evidence retrieval</b>).",
        "Classify each (document, rule) pair as Compliant / Non-compliant / Not addressed (<b>3-way classification</b>).",
        "Explain every decision with the evidence clause, its source document, and a reason.",
        "Measure everything on held-out data against honest baselines.",
        "Cover the six policy areas from our proposal: Information Security, Employee Privacy, Data Retention, "
        "Workplace Safety, Financial Procedures, Access Control.",
    ])

    # ---------------- 2. what we did
    story.append(P("2. What we did (summary of work)", "h1"))
    story.append(table([
        ["#", "Step", "What we built", "Output"],
        ["1", "Data preparation<br/><font size=7 color='#5f5e5a'>src/prepare_data.py</font>",
         "Parsed ContractNLI JSON; split documents into clauses using the dataset’s span offsets; mapped "
         "lawyer labels to our 3 labels; kept the evidence clause IDs for every pair.",
         "rules.csv, clauses_*.csv, pairs_*.csv"],
        ["2", "Exploratory analysis<br/><font size=7 color='#5f5e5a'>src/eda.py</font>",
         "Label balance, per-rule breakdown, document lengths, evidence share. Found that evidence is only "
         "~2.7% of a document → motivated the two-stage design.",
         "3 figures, eda_summary.csv"],
        ["3", "ML pipeline<br/><font size=7 color='#5f5e5a'>src/train_models.py</font>",
         "Stage A: 3 retrievers (random, TF-IDF similarity, supervised LogReg × 17 rules). Stage B: 3 judges "
         "(majority, per-rule prior, retrieval + LogReg × 17 rules). Evaluated on 123 test contracts.",
         "metrics.json, 3 figures, saved model (.joblib)"],
        ["4", "End-to-end checker<br/><font size=7 color='#5f5e5a'>src/compliance_checker.py</font>",
         "Wrote a fictional company (19 rules, 5 procedure docs, hand-labelled answers). Built clause splitting, "
         "char-n-gram retrieval, a rule engine for numbers/units, obligations and prohibitions, and an HTML "
         "audit report.",
         "compliance_report.html, compliance_results.csv"],
        ["5", "Presentation",
         "Executed notebook that walks through all steps; README with results and viva notes; this report.",
         "midterm_demo.ipynb, README.md, README.pdf"],
        ["6", "Web platform (after mid-term)<br/><font size=7 color='#5f5e5a'>app.py, checker/</font>",
         "Flask web app with a policy check (any uploaded policy and documents), a contract check (the trained "
         "model) and a method page. Added automatic rule typing, TF-IDF + Sentence-BERT retrieval and a "
         "review flag for weak matches (Section 11).",
         "Running app, report page, CSV download"],
    ], [8 * mm, 34 * mm, CONTENT_W - 8 * mm - 34 * mm - 38 * mm, 38 * mm]))

    # ---------------- 3. datasets
    story.append(P("3. Datasets", "h1"))
    story.append(P("3.1 ContractNLI (real benchmark, Stanford NLP, 2021)", "h2"))
    story.append(P(
        f"ContractNLI contains <b>{int(eda['documents'])} real Non-Disclosure Agreements</b>. For each contract, "
        f"lawyers checked the same <b>{int(eda['rules checked per document'])} rules</b> (called hypotheses) and "
        "labelled whether the contract entails, contradicts or does not mention each rule. They also marked "
        "<b>which clauses are the evidence</b>, which lets us train and test the retrieval step, not just the final "
        "label. We map its natural-language-inference labels to compliance labels:"))
    story.append(table([
        ["ContractNLI label", "Our label", "Meaning"],
        ["Entailment", "<font color='#2e7d4f'><b>Compliant</b></font>", "The document follows the rule."],
        ["Contradiction", "<font color='#b3402f'><b>Non-compliant</b></font>", "The document breaks the rule."],
        ["NotMentioned", "<font color='#9a6a12'><b>Not addressed</b></font>", "The document is silent on the rule."],
    ], [40 * mm, 38 * mm, CONTENT_W - 78 * mm]))
    story.append(Spacer(1, 6))
    story.append(table([
        ["Split", "Documents", "Clauses", "(document, rule) pairs", "Used for"],
        ["Train", "423", "32,895", "7,191", "fitting all models"],
        ["Dev", "61", "5,102", "1,037", "held back (hyper-parameter tuning planned)"],
        ["Test", "123", "10,061", "2,091", "all reported results; never seen in training"],
        ["<b>Total</b>", "<b>607</b>", "<b>48,058</b>", "<b>10,319</b>", ""],
    ], [22 * mm, 24 * mm, 24 * mm, 38 * mm, CONTENT_W - 108 * mm]))
    story.append(P("3.2 NovaTech demo corpus (created by us)", "h2"))
    story.append(P(
        "ContractNLI only covers NDAs, but our proposal targets corporate policies. So we wrote a fictional company, "
        "<b>NovaTech Solutions</b>, with <b>19 policy rules</b> across the six policy areas, and <b>5 department "
        "procedure documents</b> (IT, HR, Finance, Facilities &amp; Safety, Warehouse) containing 37 clauses. We "
        "deliberately planted compliant clauses, violations (wrong numbers, weak wording) and missing topics, and "
        "hand-labelled the correct answer for every rule in <font face='Mono'>ground_truth.csv</font>. Each rule in "
        "<font face='Mono'>policies.json</font> also has a <i>check type</i> (min, max, obligation, prohibition) and "
        "search keywords."))

    # ---------------- 4. EDA
    story.append(P("4. Exploratory data analysis", "h1"))
    story.append(table([
        ["Statistic", "Value", "Why it matters"],
        ["Compliant / Non-compliant / Not addressed",
         f"{int(eda['compliant pairs']):,} / {int(eda['non-compliant pairs']):,} / {int(eda['not-addressed pairs']):,}"
         "<br/>(49% / 11% / 40%)",
         "Non-compliance is the <b>rare class</b> → accuracy alone is misleading; we report macro-F1 and use "
         "class weights."],
        ["Median clauses per document", f"{eda['median clauses per document']}",
         "A document is long; reading all of it for each rule adds noise."],
        ["Evidence clauses per decision", f"{eda['avg evidence clauses per decision']}",
         "A decision depends on about 2 clauses."],
        ["Share of document that is evidence", f"{eda['avg % of document that is evidence']}",
         "<b>Key insight:</b> find the evidence first, then judge it → two-stage pipeline."],
    ], [52 * mm, 36 * mm, CONTENT_W - 88 * mm]))
    story.append(Spacer(1, 8))
    story.append(figure_pair("01_label_distribution.png", "03_document_length.png",
                             "Figure 1. Left: label distribution (non-compliant is rare). "
                             "Right: document length in clauses."))
    story.append(figure("02_per_rule_breakdown.png",
                        "Figure 2. Label mix per rule: some rules are violated much more often than others, "
                        "so the per-rule prior is already a strong baseline.", width=CONTENT_W * 0.86))

    # ---------------- 5. architecture
    story.append(P("5. System architecture", "h1"))
    story.append(P(
        "The pipeline mirrors how a human auditor works: <b>locate</b> the relevant passage, then <b>judge</b> it. "
        "Separate models are trained for each rule, because evidence for “no reverse engineering” looks "
        "nothing like evidence for “survival of obligations”."))
    story.append(KeepTogether([Spacer(1, 4), pipeline_diagram(),
                               P("Figure 3. Two-stage pipeline used on ContractNLI.", "caption")]))
    story.append(P(
        "For the company demo the same <i>locate → judge → explain</i> structure is used, but with only 19 "
        "rules and 37 clauses there is nothing to train on, so Stage A is unsupervised similarity search and Stage B "
        "is a transparent rule engine (Section 6.5)."))

    # ---------------- 6. maths
    story.append(PageBreak())
    story.append(P("6. Technical &amp; mathematical foundations", "h1"))
    story.append(P("6.1 Turning text into vectors: TF-IDF", "h2"))
    story.append(P(
        "A machine learning model needs numbers. TF-IDF represents a clause <i>c</i> as a vector with one dimension per "
        "vocabulary term <i>t</i> (a word or an n-gram). A term gets a high weight if it is <b>frequent in this "
        "clause</b> but <b>rare across the collection</b>. We use scikit-learn’s variant with sub-linear term "
        "frequency and smoothed IDF:"))
    story.append(eq(r"\mathrm{tf}'(t,c) = 1+\ln\,\mathrm{tf}(t,c) \qquad\qquad "
                    r"\mathrm{idf}(t) = \ln\frac{1+N}{1+\mathrm{df}(t)}+1"))
    story.append(eq(r"x_{c,t} = \mathrm{tf}'(t,c)\cdot\mathrm{idf}(t), \qquad\qquad "
                    r"\hat{\mathbf{x}}_c = \frac{\mathbf{x}_c}{\Vert\mathbf{x}_c\Vert_2}"))
    story.append(P(
        "Here tf(<i>t</i>, <i>c</i>) is the raw count of <i>t</i> in clause <i>c</i>, <i>N</i> is the number of clauses "
        "and df(<i>t</i>) is the number of clauses containing <i>t</i>. The logarithm in tf′ stops a word repeated "
        "five times from counting five times as much. The +1 terms avoid division by zero. Each vector is L2-normalised "
        "so long and short clauses are comparable."))
    story.append(P(
        "<b>Worked example.</b> With N = 32,895 training clauses, a word that appears in 30 of them gets "
        "idf = ln(32,896 / 31) + 1 ≈ <b>7.97</b>, while a word that appears in 15,000 of them gets "
        "idf = ln(32,896 / 15,001) + 1 ≈ <b>1.79</b>. The rare, specific word is weighted about 4.5× more.",
        "callout"))
    story.append(table([
        ["Where", "Analyzer", "n-grams", "Other settings"],
        ["Stage A (clauses)", "words", "1–2", "English stop words removed, min_df = 2, max 50,000 features"],
        ["Stage B (evidence)", "words", "1–3", "min_df = 2; trigrams capture phrases like “shall not disclose”"],
        ["Company demo", "characters (char_wb)", "3–5",
         "matches word variants: “report” and “reported” share “rep”, “epo”, "
         "“port”, …"],
    ], [32 * mm, 34 * mm, 18 * mm, CONTENT_W - 84 * mm]))

    story.append(P("6.2 Measuring similarity: cosine similarity", "h2"))
    story.append(P("The unsupervised retriever (A1) and the company demo rank clauses by the cosine of the angle "
                   "between the rule vector <b>q</b> and the clause vector <b>c</b>:"))
    story.append(eq(r"\cos(\mathbf{q},\mathbf{c}) = \frac{\mathbf{q}\cdot\mathbf{c}}{\Vert\mathbf{q}\Vert_2\,"
                    r"\Vert\mathbf{c}\Vert_2} = \sum_t \hat{q}_t\,\hat{c}_t \ \in [0,1]"))
    story.append(P("Because TF-IDF weights are never negative, the score is between 0 (no shared terms) and 1 "
                   "(identical direction). Since the vectors are already normalised it is just a dot product, which "
                   "is fast on sparse matrices."))

    story.append(P("6.3 Stage A: supervised evidence finder (binary Logistic Regression)", "h2"))
    story.append(P(
        "Plain similarity only rewards word overlap with the rule sentence. Instead, for each rule <i>r</i> we train a "
        "classifier that learns <i>what evidence for that rule looks like</i> from the lawyers’ annotations. Each "
        "training clause gets label <i>y</i> = 1 if it was marked as evidence for <i>r</i>, otherwise 0. The model "
        "outputs a relevance score:"))
    story.append(eq(r"s_r(c) = P(y=1\mid\hat{\mathbf{x}}_c) = \sigma\!\left(\mathbf{w}_r^{\top}\hat{\mathbf{x}}_c+b_r"
                    r"\right), \qquad \sigma(z)=\frac{1}{1+e^{-z}}"))
    story.append(P("The weights are learned by minimising the L2-regularised, class-weighted log-loss "
                   "(binary cross-entropy):"))
    story.append(eq(r"\min_{\mathbf{w},b}\ \ \frac{1}{2}\Vert\mathbf{w}\Vert_2^2\ +\ C\sum_{i=1}^{n}\alpha_{y_i}"
                    r"\left[-y_i\ln p_i-(1-y_i)\ln(1-p_i)\right], \qquad p_i=s_r(c_i)"))
    story.append(P(
        "<b>C</b> = 4 controls the trade-off between fitting the data and keeping weights small (larger C = weaker "
        "regularisation). <b>α</b> are “balanced” class weights. Without them the model could reach "
        "~98% accuracy by never predicting “evidence”. With <i>n</i> samples, <i>K</i> classes and "
        "<i>n<sub>j</sub></i> samples in class <i>j</i>:"))
    story.append(eq(r"\alpha_j = \frac{n}{K\cdot n_j}\qquad\Rightarrow\qquad \mathrm{if\ 2\%\ of\ clauses\ are\ "
                    r"evidence:}\ \ \alpha_1=\frac{1}{2\times0.02}=25,\ \ \alpha_0=\frac{1}{2\times0.98}\approx0.51"))
    story.append(P(
        "So missing one true evidence clause costs about 49× more than one false alarm. The optimisation is convex "
        "and is solved with L-BFGS. At inference every clause of a new document is scored, and the <b>top k = 3</b> "
        "clauses are passed on. This is 17 models, one per rule. Each learned weight is readable: a large positive "
        "<i>w<sub>t</sub></i> means term <i>t</i> signals evidence for that rule, which makes the model explainable."))

    story.append(P("6.4 Stage B: compliance judge (multinomial Logistic Regression)", "h2"))
    story.append(P(
        "For each (document, rule) pair we concatenate the text of the top-3 clauses (⊕), turn it into a TF-IDF "
        "vector φ(·) with its own 1–3-gram vocabulary, and append the best Stage A score. That score "
        "tells the judge how confident the retriever was, which matters for “Not addressed”:"))
    story.append(eq(r"\mathbf{z}_{D,r} = \left[\ \phi(c_{(1)}\oplus c_{(2)}\oplus c_{(3)})\ ;\ "
                    r"3\cdot\max_{c\in D}\,s_r(c)\ \right]"))
    story.append(P("A softmax over the three labels gives class probabilities, and the prediction is the most "
                   "probable label:"))
    story.append(eq(r"P(y=j\mid\mathbf{z}) = \frac{\exp(\mathbf{w}_j^{\top}\mathbf{z}+b_j)}"
                    r"{\sum_{l=1}^{3}\exp(\mathbf{w}_l^{\top}\mathbf{z}+b_l)}, \qquad "
                    r"\hat{y}=\arg\max_j P(y=j\mid\mathbf{z})"))
    story.append(eq(r"\min_{W,\mathbf{b}}\ \ \frac{1}{2}\Vert W\Vert_F^2\ -\ C\sum_{i}\alpha_{y_i}\ln P(y_i\mid"
                    r"\mathbf{z}_i), \qquad C=2,\ \ \alpha\ \mathrm{balanced}"))
    story.append(P(
        "Words such as <i>not</i>, <i>shall not</i>, <i>may</i>, <i>except</i> and <i>only</i> get large weights for "
        "the Non-compliant or Compliant class, depending on the rule. Two simple baselines show how much of the result "
        "comes from learning: <b>B0</b> always predicts the overall most common label, and <b>B1</b> predicts the most "
        "common label for that rule in the training set."))

    story.append(P("6.5 Company demo: similarity search + transparent rule engine", "h2"))
    story.append(P(
        "<b>Retrieval and the “Not addressed” decision.</b> Each rule query <b>q</b><sub>r</sub> (rule text + "
        "keywords) is compared with all 37 procedure clauses using character 3–5-gram TF-IDF. If even the best "
        "match is weak, nothing in the procedures talks about the rule:"))
    story.append(eq(r"j^{*}=\arg\max_j\ \cos(\mathbf{q}_r,\mathbf{c}_j), \qquad "
                    r"\cos(\mathbf{q}_r,\mathbf{c}_{j^{*}})<\tau=0.20\ \ \Rightarrow\ \ \mathrm{Not\ addressed}"))
    story.append(P(
        "<b>Numeric rules</b> (type <i>min</i> / <i>max</i>). Regular expressions extract a value <i>v</i> and unit "
        "<i>u</i> from the evidence clause (e.g. “60 days”, “4 hours”). Frequency words are mapped to "
        "quantities (annually → 1 year, quarterly → 3 months, monthly → 1 month, weekly → 1 week, "
        "same day / immediately → 0 days). Everything is converted to days and compared with the policy value "
        "<i>h</i>*:"))
    story.append(eq(r"h=v\cdot\lambda(u),\quad \lambda(\mathrm{hour})=\frac{1}{24},\ \lambda(\mathrm{day})=1,\ "
                    r"\lambda(\mathrm{week})=7,\ \lambda(\mathrm{month})=30,\ \lambda(\mathrm{year})=365"))
    story.append(eq(r"\mathrm{min\ rule:}\ \ \mathrm{Compliant}\Leftrightarrow h\geq h^{*}\qquad\qquad "
                    r"\mathrm{max\ rule:}\ \ \mathrm{Compliant}\Leftrightarrow h\leq h^{*}"))
    story.append(P(
        "<b>Worked example (SAF-01).</b> Policy: fire drills at least every 6 months, a <i>max</i> rule with "
        "<i>h</i>* = 6 × 30 = 180 days. Procedure: “held <i>once a year</i>” → (1, year) → "
        "<i>h</i> = 365 days. Since 365 &gt; 180 the verdict is <b>Non-compliant</b>. Password length is compared "
        "directly in characters (8 &lt; 12 → Non-compliant).", "callout"))
    story.append(P(
        "<b>Obligations</b> (“must be enabled”): Non-compliant if the clause contains weakening language "
        "(<i>optional, not mandatory, not required, encouraged, recommended, may choose, if possible, at their "
        "discretion</i>), otherwise Compliant. <b>Prohibitions</b> (“must not be shared”): Compliant only if the "
        "clause contains a restriction word (<i>not, never, no, prohibited, forbidden, banned, disallowed, without</i>). "
        "A numeric rule whose clause has no number falls back to the obligation check. Every verdict stores the "
        "evidence clause, its document, the similarity and a sentence explaining the reason."))

    story.append(P("6.6 Evaluation metrics", "h2"))
    story.append(P(
        "<b>Recall@k (Stage A).</b> <i>G</i><sub>D,r</sub> is the set of clauses the lawyers marked as evidence and "
        "<i>E</i><sub>k</sub> is our top-k. Pairs labelled Not addressed have no evidence and are skipped:"))
    story.append(eq(r"\mathrm{Recall@}k=\frac{\sum_{(D,r)}\ |\,G_{D,r}\cap E_k(D,r)\,|}{\sum_{(D,r)}\ |\,G_{D,r}\,|}"))
    story.append(P(
        "<b>Sanity check:</b> a random ranker should find about k / n̄ of the evidence. Test documents have "
        "n̄ = 10,061 / 123 ≈ 81.8 clauses, so Recall@1 ≈ 1/81.8 ≈ 1.2%, which is exactly what the "
        "random baseline A0 scores. This confirms the evaluation code is correct.", "callout"))
    story.append(P("<b>Classification metrics (Stage B).</b> For each class <i>j</i>, with true positives (TP), "
                   "false positives (FP) and false negatives (FN):"))
    story.append(eq(r"P_j=\frac{TP_j}{TP_j+FP_j},\qquad R_j=\frac{TP_j}{TP_j+FN_j},\qquad "
                    r"F1_j=\frac{2\,P_j R_j}{P_j+R_j}"))
    story.append(eq(r"\mathrm{Accuracy}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\hat{y}_i=y_i], \qquad "
                    r"\mathrm{Macro\ F1}=\frac{1}{3}\sum_{j=1}^{3}F1_j"))
    story.append(P(
        "Accuracy is dominated by the large classes; macro-F1 gives the rare Non-compliant class the same weight as "
        "the others, so a model cannot look good by ignoring violations. For B2 on the test set: Non-compliant "
        "precision 0.604 and recall 0.673 give F1 = 2(0.604)(0.673)/(1.277) = <b>0.637</b>, and "
        f"macro-F1 = (0.859 + 0.637 + 0.805) / 3 = <b>{b2['macro_F1']:.3f}</b>."))

    # ---------------- 7. results
    story.append(P("7. Results on ContractNLI (123 unseen test contracts)", "h1"))
    story.append(P("7.1 Stage A: finding the evidence", "h2"))
    rows = [["Method", "Recall@1", "Recall@3", "Recall@5", "Recall@10"]]
    for name, v in A.items():
        b = name.startswith("A2")
        fmt = (lambda s: f"<b>{s}</b>") if b else (lambda s: s)
        rows.append([fmt(name)] + [fmt(f"{v[f'Recall@{k}']:.1%}") for k in (1, 3, 5, 10)])
    t = table(rows, [CONTENT_W - 4 * 26 * mm] + [26 * mm] * 4)
    story.append(highlight_row(t, 3))
    story.append(Spacer(1, 6))
    story.append(figure("04_stageA_recall.png",
                        "Figure 4. Recall@k. The supervised retriever finds 73% of the evidence in its top 3, "
                        "2.5× the unsupervised similarity baseline.", width=CONTENT_W * 0.66))

    story.append(P("7.2 Stage B: the compliance decision", "h2"))
    rows = [["Model", "Accuracy", "Macro-F1", "F1 Compliant", "F1 Non-compliant", "F1 Not addressed"]]
    for name, v in B.items():
        b = name.startswith("B2")
        fmt = (lambda s: f"<b>{s}</b>") if b else (lambda s: s)
        rows.append([fmt(name)] + [fmt(f"{v[k]:.3f}") for k in
                                   ("accuracy", "macro_F1", "F1_Compliant", "F1_Non-compliant", "F1_Not addressed")])
    t = table(rows, [CONTENT_W - 126 * mm, 21 * mm, 21 * mm, 26 * mm, 28 * mm, 30 * mm])
    story.append(highlight_row(t, 3))
    story.append(Spacer(1, 6))
    story.append(table([
        ["B2 per class", "Precision", "Recall", "F1", "Support"],
        ["Compliant", "0.837", "0.881", "0.859", "968"],
        ["Non-compliant", "0.604", "0.673", "0.637", "220"],
        ["Not addressed", "0.842", "0.771", "0.805", "903"],
        ["Macro average", "0.761", "0.775", "0.767", "2,091"],
    ], [CONTENT_W - 4 * 26 * mm] + [26 * mm] * 4))
    story.append(Spacer(1, 6))
    story.append(figure_pair("05_stageB_model_comparison.png", "06_confusion_matrix.png",
                             "Figure 5. Left: our model vs. baselines. Right: confusion matrix of B2. Most remaining "
                             "errors are between Compliant and Not addressed."))
    story.append(P(
        f"<b>Interpretation.</b> The full pipeline improves accuracy from 65.9% to <b>{b2['accuracy']:.1%}</b> and "
        f"macro-F1 from 0.617 to <b>{b2['macro_F1']:.3f}</b> over the strongest baseline. The rare Non-compliant "
        "class is still the hardest (F1 0.637), which is expected with only 11% of the training data. Per-rule "
        f"accuracy ranges from {per_rule.accuracy.min():.0%} to {per_rule.accuracy.max():.0%} (Appendix B). Rules "
        "with very distinctive wording (no solicitation, no reverse engineering) are easiest; vague rules about "
        "keeping or returning information are hardest."))

    # ---------------- 8. demo
    story.append(P("8. End-to-end demo on company policies (NovaTech)", "h1"))
    story.append(P(
        f"Running <font face='Mono'>compliance_checker.py</font> on the 19 NovaTech rules gives "
        f"<b>{int(round(demo_acc * len(demo)))}/{len(demo)} correct ({demo_acc:.1%} accuracy, macro-F1 0.847)</b> "
        "against our hand labels, and writes an HTML audit report with evidence for every decision."))
    rows = [["Rule", "Type", "Policy requirement", "Evidence found in procedures", "Predicted", "Truth"]]
    for r in demo.itertuples():
        ev = "—" if r.prediction == "NOT_ADDRESSED" else escape(r.evidence)
        ok = r.prediction == r.ground_truth
        pred = f"<font color='{LABEL_COL[r.prediction].hexval().replace('0x', '#')}'><b>{PRETTY[r.prediction]}</b></font>"
        truth = PRETTY[r.ground_truth] + ("" if ok else " <font color='#b3402f'><b>✗</b></font>")
        rows.append([f"<b>{r.rule_id}</b>", rule_check[r.rule_id]["type"], escape(r.rule), ev, pred, truth])
    data = [[Paragraph(c, S["head"] if i == 0 else S["small"]) for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=[17 * mm, 20 * mm, 39 * mm, CONTENT_W - 17 * mm - 20 * mm - 39 * mm - 52 * mm,
                               26 * mm, 26 * mm], repeatRows=1)
    cmds = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    for i, r in enumerate(demo.itertuples(), start=1):
        if r.prediction != r.ground_truth:
            cmds.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fbeceA")))
        elif i % 2 == 0:
            cmds.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(cmds))
    story.append(t)
    story.append(P("Table: all 19 NovaTech rules. Rows shaded red are the two errors.", "caption"))

    # ---------------- 9. errors & limitations
    story.append(P("9. Error analysis &amp; limitations", "h1"))
    story.append(table([
        ["Error", "What happened", "Root cause", "Fix (next phase)"],
        ["ACC-03<br/>“shared accounts must not be used”",
         "Predicted <i>Not addressed</i> (best similarity 0.16 &lt; 0.20); truth is Non-compliant.",
         "The procedure says “common login”. TF-IDF matches surface strings, not meaning.",
         "Sentence embeddings (Sentence-BERT) for meaning-based retrieval."],
        ["RET-03<br/>“backups must be tested for restoration every quarter”",
         "Predicted <i>Compliant</i>; truth is Not addressed.",
         "It found a backup clause (similarity 0.21), but that clause never mentions testing. "
         "“No weak words” is not the same as “the clause implies the rule”.",
         "Transformer NLI model (e.g. DeBERTa fine-tuned on ContractNLI) to check entailment."],
    ], [30 * mm, 42 * mm, CONTENT_W - 30 * mm - 42 * mm - 44 * mm, 44 * mm]))
    story.append(Spacer(1, 6))
    story += bullets([
        "<b>Bag-of-n-grams has no real understanding</b> of negation scope, synonyms or long-range structure; it relies "
        "on surface cues.",
        "<b>Slightly optimistic training signal.</b> Stage B is trained on Stage A’s top clauses for the training "
        "documents, scored by a Stage A model that saw those same documents. Test results are still clean (test "
        "documents are never used in training), but cross-fitting (k-fold out-of-sample scores) is the correct fix.",
        "<b>Domain gap.</b> The learned models are trained on NDAs; the company demo uses a hand-written rule engine. "
        "The NovaTech corpus is small (19 rules) and written by us, so its 89.5% is a demonstration, not a benchmark.",
        "<b>Hand-tuned pieces:</b> the 0.20 threshold, word lists and per-rule check types were set by hand.",
        "<b>Non-compliant F1 of 0.64</b> is the weakest point and the most important class for an auditor.",
    ])

    # ---------------- 10. future
    story.append(P("10. Plan after mid-term (and status)", "h1"))
    done = "<font color='#2e7d4f'><b>Done</b></font>"
    story.append(table([
        ["Planned work", "Expected benefit", "Status"],
        ["Replace TF-IDF retrieval with Sentence-BERT embeddings (bi-encoder) and compare Recall@k",
         "Matches meaning, not words (fixes ACC-03-type errors).",
         done + " for the policy check, as a TF-IDF + Sentence-BERT hybrid (Section 11). Not applied to "
                "ContractNLI."],
        ["Fine-tune a transformer NLI model (DeBERTa / Legal-BERT) as the Stage B judge",
         "Checks whether the clause actually implies the rule (fixes RET-03-type errors).",
         "Not done. A simpler safeguard was added instead: weak matches cannot prove compliance."],
        ["Class balancing, data augmentation, threshold tuning on the dev split",
         "Raise Non-compliant F1.", "Not done."],
        ["Cross-fitted Stage A scores for training Stage B", "Removes the optimistic training signal.",
         "Not done."],
        ["PDF / DOCX ingestion and a simple web interface", "Usable by a non-technical auditor.",
         done + " (Section 11)."],
    ], [CONTENT_W * 0.38, CONTENT_W * 0.28, CONTENT_W * 0.34]))

    # ---------------- 11. web platform
    story.append(P("11. Web platform (built after the mid-term)", "h1"))
    story.append(P(
        "The problem statement asks for a system that <i>analyses documents, reports or organisational procedures "
        "to identify compliance with predefined policies and regulations</i>. After the mid-term we turned the "
        "checker into a small web application (Flask with plain HTML and CSS, no database) that a non-technical "
        "person can use. It has three pages:"))
    story.append(table([
        ["Page", "Input", "What happens"],
        ["Policy check", "A policy document (uploaded or pasted) and one or more procedure documents "
                         "(.txt, .pdf, .docx)",
         "Every rule in the policy is checked against all sentences of the documents with the method in 11.1–11.3."],
        ["Contract check", "One non-disclosure agreement",
         "Checked against the 17 ContractNLI rules with the trained two-stage model of Sections 5–7."],
        ["Method", "—", "Explains both checks and shows the test results."],
    ], [28 * mm, 62 * mm, CONTENT_W - 90 * mm]))
    story.append(Spacer(1, 4))
    story.append(P(
        "The report shows, for every rule, the status, the evidence sentence, the document it came from and the "
        "reason. It can be filtered by status, downloaded as CSV and printed."))

    story.append(P("11.1 From a policy document to rules (automatic rule typing)", "h2"))
    story.append(P(
        "At mid-term every rule in <font face='Mono'>policies.json</font> carried search keywords and a check type "
        "that we typed by hand. That does not work for a document a user uploads. The platform therefore reads the "
        "policy text itself: headings become policy areas, every sentence with a requirement word (<i>must, shall, "
        "required, prohibited…</i>) becomes a rule, and the wording decides the rule type:"))
    story.append(table([
        ["Wording found in the rule", "Detected type", "Example"],
        ["A quantity with <i>every, once, within, at most, not more than, annually, quarterly…</i>",
         "max (value, unit)", "“reported within 24 hours” → max 24 hours"],
        ["A quantity with <i>at least, minimum, no less than</i>", "min (value, unit)",
         "“at least 12 characters” → min 12 characters"],
        ["<i>must not, shall not, prohibited, never</i> (and no quantity)", "prohibition",
         "“shared accounts must not be used”"],
        ["Any other requirement", "obligation", "“MFA must be enabled for all remote access”"],
    ], [70 * mm, 34 * mm, CONTENT_W - 104 * mm]))
    story.append(Spacer(1, 4))
    story.append(P(
        "A frequency is a maximum gap, so “at least every 6 months” is a <i>max</i> rule even though it contains "
        f"“at least”. On the NovaTech handbook the detected type equals our hand-written type for "
        f"<b>{plat['type_match']} of {len(plat['rows'])}</b> rules."))

    story.append(P("11.2 Hybrid retrieval: words + meaning", "h2"))
    story.append(P(
        "Without the hand-written keywords, TF-IDF alone picks the wrong sentence whenever the policy and the "
        "procedure use different words. We added the mid-term plan’s fix: <b>Sentence-BERT</b> "
        "(<font face='Mono'>all-MiniLM-L6-v2</font>). The model turns a sentence of <i>T</i> tokens into contextual "
        "token vectors <b>h</b><sub>1</sub>, …, <b>h</b><sub>T</sub>, averages them (mean pooling) and normalises the "
        "result to a 384-dimensional sentence vector:"))
    story.append(eq(r"\bar{\mathbf{h}}=\frac{1}{T}\sum_{i=1}^{T}\mathbf{h}_i, \qquad\qquad "
                    r"\mathbf{e}(s)=\frac{\bar{\mathbf{h}}}{\Vert\bar{\mathbf{h}}\Vert_2}\ \in\mathbb{R}^{384}"))
    story.append(P(
        "Sentences with similar meaning get vectors that point in similar directions, even with no words in common. "
        "The retrieval score of rule <i>r</i> and clause <i>c</i> is the average of the word-level similarity of "
        "Section 6.2 (character n-gram TF-IDF) and the meaning-level similarity:"))
    story.append(eq(r"s(r,c)=\frac{1}{2}\left[\ \cos_{\mathrm{tfidf}}(r,c)\ +\ \mathbf{e}(r)^{\top}\mathbf{e}(c)\ \right],"
                    r"\qquad c^{*}=\arg\max_{c}\ s(r,c)"))
    story.append(P(
        "<b>Worked example (ACC-01).</b> Policy: “Access rights of departing employees must be <i>revoked</i> "
        "within 24 hours of exit.” With TF-IDF only, the best match was “Exit interviews are conducted by the HR "
        "business partner for every departing employee” (0.28), which is the wrong sentence. With the hybrid score "
        "the best match is “When an employee leaves, all their system accounts and access rights are "
        "<i>disabled</i> on their last working day” (0.44), which is the right one.", "callout"))

    story.append(P("11.3 Decision rule: weak matches cannot prove compliance", "h2"))
    story.append(table([
        ["Best score s", "Decision"],
        ["s &lt; 0.22", "<b>Not addressed.</b> No sentence talks about the rule."],
        ["0.22 ≤ s &lt; 0.40", "<b>Weak match, marked “review”.</b> The rule engine of Section 6.5 still runs. "
                                 "A possible violation is reported. A “Compliant” verdict is not trusted and becomes "
                                 "Not addressed."],
        ["s ≥ 0.40", "<b>Verdict of the rule engine</b> (numbers and units, weak wording, restrictions)."],
    ], [34 * mm, CONTENT_W - 34 * mm]))
    story.append(Spacer(1, 4))
    story.append(P(
        "The rule is deliberately one-sided. For an auditor, a wrong “Compliant” hides a problem, while a wrong "
        "“possible violation, review” costs one manual look. This fixes the mid-term RET-03 error (a backup sentence "
        "that never mentions testing is no longer accepted as proof)."))

    story.append(P("11.4 Results of the platform", "h2"))
    story.append(table([
        ["", "Mid-term demo (Section 8)", "Web platform"],
        ["Keywords per rule", "written by hand", "none"],
        ["Rule type", "written by hand", "detected from the wording"],
        ["Retrieval", "TF-IDF (character n-grams)", "TF-IDF + Sentence-BERT, averaged"],
        ["Weak matches", "accepted", "never accepted as Compliant; marked review"],
        ["NovaTech result", f"{int(round(demo_acc * len(demo)))} / {len(demo)} correct",
         f"<b>{plat['correct']} / {len(plat['rows'])} correct</b>"],
    ], [36 * mm, (CONTENT_W - 36 * mm) / 2, (CONTENT_W - 36 * mm) / 2]))
    story.append(Spacer(1, 6))
    rows = [["Rule", "Detected type", "Best sentence found", "Score", "Result", "Truth"]]
    wrong_rows = []
    for i, r in enumerate(plat["rows"], start=1):
        c = r["check"]
        kind = c["type"] + (f" {c['value']} {c['unit']}{'' if c['value'] == 1 else 's'}" if "value" in c else "")
        result = (f"<font color='{LABEL_COL[r['status']].hexval().replace('0x', '#')}'><b>{PRETTY[r['status']]}</b>"
                  f"</font>" + (" (review)" if r["weak"] else ""))
        ok = r["status"] == r["truth"]
        if not ok:
            wrong_rows.append(i)
        rows.append([f"<b>{r['rule_id']}</b>", kind, escape(r["evidence"]) or "—", f"{r['score']:.2f}", result,
                     PRETTY[r["truth"]] + ("" if ok else " <font color='#b3402f'><b>✗</b></font>")])
    t = table(rows, [17 * mm, 27 * mm, CONTENT_W - 17 * mm - 27 * mm - 13 * mm - 30 * mm - 26 * mm, 13 * mm,
                     30 * mm, 26 * mm], style="small")
    for i in wrong_rows:
        highlight_row(t, i, colors.HexColor("#fbecea"))
    story.append(t)
    story.append(P("Table: the web platform on the 19 NovaTech rules, using only the policy text. "
                   "Rows shaded red are errors.", "caption"))
    story += bullets([
        "<b>The remaining error is ACC-02.</b> The right sentence is found (quarterly access reviews), but its score "
        "is below 0.40, so the platform reports “Not addressed, review” where the truth is Compliant. This is the "
        "price of the one-sided rule in 11.3.",
        "<b>ACC-03 is right for the wrong reason.</b> The label is correct, but the evidence is a password sentence, "
        "not the “common login” sentence. It is marked review.",
        "<b>The two thresholds (0.22 and 0.40) were chosen on these same 19 rules</b>, so this result is a "
        "demonstration, not a benchmark.",
        f"<b>Contract check on raw text.</b> In the app the contract is split into sentences by our own splitter, "
        f"not by the dataset’s clause boundaries. Measured this way on the 123 test contracts the trained model "
        f"reaches <b>{plat['contract_acc']:.1%}</b> accuracy ({b2['accuracy']:.1%} with the dataset’s boundaries).",
    ])
    story.append(P("11.5 Limits of the platform", "h2"))
    story += bullets([
        "Only time periods and character counts are compared. Money amounts and percentages are not.",
        "One rule is compared with one sentence. A requirement spread over several sentences can be missed.",
        "A prohibition can be flagged from a sentence that is only on a related topic.",
        "Scanned PDFs (images) are not read, and only English is supported.",
        "Reports are kept in memory and disappear when the app stops.",
    ])

    # ---------------- 12. how to run
    story.append(P("12. How to run &amp; project structure", "h1"))
    story.append(P("pip install -r requirements.txt<br/>"
                   "python app.py &nbsp;&nbsp;&nbsp;# web app at http://127.0.0.1:5000<br/>"
                   "python run_all.py &nbsp;&nbsp;&nbsp;# mid-term pipeline, all 4 steps, about 3–5 minutes<br/>"
                   "python docs/build_readme_pdf.py &nbsp;&nbsp;&nbsp;# rebuilds this PDF", "code"))
    story.append(P("The app page opens straight away. The Sentence-BERT model loads in the background for about a "
                   "minute, so the first policy check waits for it; later checks take under a second."))
    story.append(table([
        ["Path", "Contents"],
        ["app.py", "Web app: forms, running a check, report page, CSV download"],
        ["checker/readers.py", "Reads .txt, .pdf and .docx uploads into plain text"],
        ["checker/policy_parser.py", "Policy document → rules with a detected type (Section 11.1)"],
        ["checker/engine.py", "Policy check: sentence splitting, hybrid retrieval, rule engine (11.2–11.3)"],
        ["checker/contract_model.py", "Contract check: runs the trained Stage A + Stage B model"],
        ["templates/, static/style.css", "The pages (plain HTML and CSS)"],
        ["data/samples/", "Sample policy handbook and sample NDA for the “Run the sample” buttons"],
        ["models/all-MiniLM-L6-v2/", "Sentence-BERT model, downloaded once on first run"],
        ["src/prepare_data.py", "Step 1: raw ContractNLI JSON → clause and pair CSV tables"],
        ["src/eda.py", "Step 2: statistics and figures 01–03"],
        ["src/train_models.py", "Step 3: Stage A + Stage B training, evaluation, figures 04–06, saved model"],
        ["src/compliance_checker.py", "Step 4: NovaTech end-to-end check and HTML report"],
        ["src/style.py", "Shared chart style and label colours"],
        ["data/raw/contract-nli/", "Original dataset (train/dev/test JSON)"],
        ["data/processed/", "rules.csv, clauses_{split}.csv, pairs_{split}.csv"],
        ["data/custom/", "policies.json, procedures/*.txt, ground_truth.csv (NovaTech)"],
        ["models/contractnli_pipeline.joblib", "Both vectorisers + 17 retrievers + 17 judges"],
        ["outputs/", "metrics.json, result CSVs, compliance_report.html, figures/"],
        ["midterm_demo.ipynb", "Executed walkthrough of every step, used for the presentation"],
    ], [58 * mm, CONTENT_W - 58 * mm]))
    story.append(Spacer(1, 4))
    story.append(P("Libraries: pandas, NumPy, scikit-learn, SciPy, Matplotlib, joblib, Jupyter for the pipeline; "
                   "Flask, sentence-transformers, pypdf and python-docx for the web app. "
                   "No GPU is needed; everything runs on a laptop CPU."))

    # ---------------- appendix
    story.append(P("Appendix A. Hyper-parameters", "h1"))
    story.append(table([
        ["Component", "Setting"],
        ["Stage A vectoriser", "TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000, sublinear_tf=True, "
                               "stop_words='english')"],
        ["Stage A classifier", "LogisticRegression(C=4.0, class_weight='balanced', max_iter=2000), one per rule"],
        ["Evidence passed on", "top k = 3 clauses per (document, rule)"],
        ["Stage B vectoriser", "TfidfVectorizer(ngram_range=(1,3), min_df=2, sublinear_tf=True) + 3 × best "
                               "Stage A score"],
        ["Stage B classifier", "LogisticRegression(C=2.0, class_weight='balanced', max_iter=3000), one per rule"],
        ["Demo retriever", "TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), sublinear_tf=True); "
                           "threshold τ = 0.20"],
        ["Platform retriever", "0.5 × the demo retriever’s TF-IDF cosine + 0.5 × Sentence-BERT cosine "
                               "(all-MiniLM-L6-v2, 384 dimensions, mean pooling, CPU)"],
        ["Platform thresholds", "not addressed below 0.22; weak match (review) below 0.40"],
        ["Random seed", "42"],
    ], [38 * mm, CONTENT_W - 38 * mm]))

    story.append(P("Appendix B. ContractNLI rules and Stage B test accuracy", "h1"))
    pr = per_rule.copy()
    pr["n"] = pr.rule_id.str.split("-").str[1].astype(int)
    pr = pr.sort_values("n")
    rows = [["Rule", "Name", "Rule text (hypothesis)", "Accuracy"]]
    for r in pr.itertuples():
        rows.append([r.rule_id, escape(r.rule_name), escape(r.rule_text), f"{r.accuracy:.1%}"])
    story.append(table(rows, [16 * mm, 40 * mm, CONTENT_W - 16 * mm - 40 * mm - 20 * mm, 20 * mm], style="small"))

    doc = SimpleDocTemplate(PDF_PATH, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=19 * mm, bottomMargin=18 * mm,
                            title="Intelligent Policy Compliance Checker — Project Report",
                            subject="NLP mid-term project: README and technical report")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    build()
