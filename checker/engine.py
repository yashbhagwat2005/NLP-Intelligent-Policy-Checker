"""
POLICY CHECK - compares procedure documents with a list of policy rules.

For every rule:
  1. RETRIEVE  the procedure clause most similar to the rule. The score is the average of
                 - word similarity    : TF-IDF on character n-grams ("report" ~ "reported")
                 - meaning similarity : Sentence-BERT embeddings ("revoked" ~ "disabled")
               -> if nothing is similar enough            => NOT_ADDRESSED
  2. JUDGE     that clause according to the type of rule
               - min / max rule : compare the number in the clause with the number in the rule
               - obligation     : weak wording ("optional", "encouraged")  => NON_COMPLIANT
               - prohibition    : clause must restrict the activity, else  => NON_COMPLIANT
               A weak match (score below 0.40) is never accepted as proof of compliance.
  3. EXPLAIN  keep the clause, the document it came from and a one-line reason
"""
import os
import re
import threading

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_THRESHOLD = 0.22   # below this, no clause talks about the rule
WEAK_MATCH = 0.40             # below this, the match is shown as "weak - please review"

# ----------------------------------------------------------------- sentence embeddings
MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", MODEL_NAME)
_embedder = None


def _load_embedder():
    """Importing the library is slow, so app.py runs this in the background at start-up."""
    global _embedder
    from sentence_transformers import SentenceTransformer
    if not os.path.isdir(MODEL_DIR):                       # first run only: download once (about 90 MB)
        SentenceTransformer(MODEL_NAME).save(MODEL_DIR)
    _embedder = SentenceTransformer(MODEL_DIR, device="cpu")


_loader = threading.Thread(target=_load_embedder, daemon=True)


def start_loading_embedder():
    if not _loader.is_alive() and _embedder is None:
        _loader.start()


def embedder_ready():
    return _embedder is not None


def embed(texts):
    """Sentence-BERT vectors, one per text, scaled to length 1."""
    if _embedder is None:
        start_loading_embedder()
        _loader.join()                                     # wait until the model is ready
    return _embedder.encode(texts, normalize_embeddings=True)

# ----------------------------------------------------------------- splitting text
ITEM_START = re.compile(r"^(?:[-*•]|\(?\d+(?:\.\d+)*[.)]|\(?[a-z][.)]|[A-Z]{2,6}-\d+[.:)]?)\s+")
RULE_ID = re.compile(r"^([A-Z]{2,6}-\d+)[.:)]?\s+")
SENTENCE_END = re.compile(r'(?<=[.!?;])\s+(?=[A-Z0-9"(])')


def is_heading(line):
    words = line.split()
    letters = [ch for ch in line if ch.isalpha()]
    if not letters:
        return True
    if all(ch.isupper() for ch in letters):
        return True                                   # "INFORMATION SECURITY"
    return len(words) <= 8 and line[-1] not in ".;!?" # "Access Control" / "Scope:"


def split_blocks(text):
    """
    Turn raw text into blocks: [{"text", "heading", "rule_id"}].
    A new block starts at a blank line, a heading, or a numbered / bulleted item.
    Lines inside a block are joined, which repairs the line breaks PDFs add mid-sentence.
    """
    blocks, current, rule_id = [], [], None

    def flush():
        nonlocal current, rule_id
        if current:
            blocks.append({"text": " ".join(current), "heading": False, "rule_id": rule_id})
        current, rule_id = [], None

    for raw in text.splitlines():
        line = " ".join(raw.split())
        if not line:
            flush()
            continue
        item = ITEM_START.match(line)
        if item:
            flush()
            found = RULE_ID.match(line)
            rule_id = found.group(1) if found else None
            line = line[item.end():].strip()
            if not line:
                continue
        elif is_heading(line):
            flush()
            blocks.append({"text": line.rstrip(":"), "heading": True, "rule_id": None})
            continue
        current.append(line)
    flush()
    return blocks


def split_sentences(text):
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def split_clauses(text, document):
    """One clause = one sentence of a procedure document (headings and tiny fragments dropped)."""
    clauses = []
    for block in split_blocks(text):
        if block["heading"]:
            continue
        for sentence in split_sentences(block["text"]):
            if len(sentence.split()) >= 4:
                clauses.append({"document": document, "clause": sentence})
    return clauses


# ----------------------------------------------------------------- numbers and units
TO_DAYS = {"minute": 1 / 1440, "hour": 1 / 24, "day": 1, "week": 7, "month": 30, "year": 365}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
FREQUENCY_WORDS = {  # phrase -> (value, unit)
    r"once a year|annually|yearly|every year": (1, "year"),
    r"quarterly|every quarter": (3, "month"),
    r"monthly|every month": (1, "month"),
    r"weekly|every week": (1, "week"),
    r"daily|every day": (1, "day"),
}
SAME_DAY = r"same day|last working day|immediately"
WEAK_WORDS = (r"\b(optional|not mandatory|not required|not compulsory|encouraged|"
              r"at (their|the) discretion|may choose|if possible|where possible|recommended)\b")
RESTRICTION_WORDS = (r"\b(not|never|no|prohibited|forbidden|banned|disallowed|without|only|restricted|"
                     r"requires? (prior )?(approval|authori[sz]ation|consent))\b")


def extract_quantity(text, want_characters=False, allow_same_day=True):
    """Find a quantity in the text. Returns (value, unit, matched_text) or None."""
    text = text.lower()
    text = re.sub(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", lambda m: str(NUMBER_WORDS[m.group(1)]), text)
    if want_characters:
        m = re.search(r"(\d+)[\s-]*(characters|chars)", text)
        return (int(m.group(1)), "character", m.group(0)) if m else None
    m = re.search(r"(\d[\d,]*)[\s-]*(minute|hour|day|week|month|year)s?\b", text)
    if m:
        return int(m.group(1).replace(",", "")), m.group(2), m.group(0)
    for pattern, (value, unit) in FREQUENCY_WORDS.items():
        m = re.search(pattern, text)
        if m:
            return value, unit, m.group(0)
    if allow_same_day:
        m = re.search(SAME_DAY, text)
        if m:
            return 0, "day", m.group(0)
    return None


def in_days(value, unit):
    return value * TO_DAYS[unit]


# ----------------------------------------------------------------- judging one clause
def judge(check, clause):
    """Return (label, reason) for one rule and its evidence clause."""
    kind = check["type"]
    text = clause.lower()

    if kind in ("min", "max"):
        is_chars = check["unit"] == "character"
        found = extract_quantity(clause, want_characters=is_chars)
        if found:
            value, unit, matched = found
            have = value if is_chars else in_days(value, unit)
            need = check["value"] if is_chars else in_days(check["value"], check["unit"])
            ok = have >= need if kind == "min" else have <= need
            limit = "at least" if kind == "min" else "at most"
            plural = "" if check["value"] == 1 else "s"
            reason = (f'Procedure says "{matched}"; policy requires {limit} '
                      f'{check["value"]} {check["unit"]}{plural}.')
            return ("COMPLIANT" if ok else "NON_COMPLIANT"), reason
        kind = "obligation"   # the clause has no number, so check its wording instead

    if kind == "obligation":
        weak = re.search(WEAK_WORDS, text)
        if weak:
            return "NON_COMPLIANT", f'The rule is mandatory, but the procedure says "{weak.group(0)}".'
        return "COMPLIANT", "The procedure states this requirement without weakening it."

    # prohibition
    restriction = re.search(RESTRICTION_WORDS, text)
    if restriction:
        return "COMPLIANT", f'The procedure restricts this activity ("{restriction.group(0)}").'
    return "NON_COMPLIANT", "The policy forbids this, but the procedure describes doing it."


# ----------------------------------------------------------------- the whole check
def check_policies(rules, clauses):
    """rules: from policy_parser.parse_policy().  clauses: from split_clauses().  Returns one row per rule."""
    queries = [r["text"] for r in rules]
    texts = [c["clause"] for c in clauses]

    # word similarity: TF-IDF over character n-grams
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    vectorizer.fit(texts + queries)
    word_sim = cosine_similarity(vectorizer.transform(queries), vectorizer.transform(texts))
    # meaning similarity: sentence embeddings (already length 1, so the dot product is the cosine)
    meaning_sim = embed(queries) @ embed(texts).T
    similarity = (word_sim + meaning_sim) / 2

    rows = []
    for rule, scores in zip(rules, similarity):
        best = int(scores.argmax())
        score = float(scores[best])
        if score < SIMILARITY_THRESHOLD:
            status, evidence, source = "NOT_ADDRESSED", "", ""
            reason = "No clause in the uploaded documents covers this rule."
        else:
            evidence, source = clauses[best]["clause"], clauses[best]["document"]
            status, reason = judge(rule["check"], evidence)
            # A loosely related sentence is never accepted as proof of compliance.
            # (A possible violation on a weak match is still shown, marked for review.)
            if score < WEAK_MATCH and status == "COMPLIANT":
                status = "NOT_ADDRESSED"
                reason = "A related sentence was found, but the match is too weak to confirm compliance."
        rows.append({
            "rule_id": rule["id"], "area": rule["area"], "rule": rule["text"],
            "rule_type": rule["check"]["type"], "status": status, "evidence": evidence,
            "source": source, "score": round(score, 2), "reason": reason,
            "weak": SIMILARITY_THRESHOLD <= score < WEAK_MATCH,
        })
    return rows
