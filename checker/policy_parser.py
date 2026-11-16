"""
Turn a policy document into a list of rules the engine can check.

A sentence is a rule if it contains a requirement word ("must", "shall", "prohibited", ...).
Each rule gets a type, worked out from its wording:

    "at least 12 characters", "for at least 8 years"         -> min   (value, unit)
    "within 24 hours", "every 6 months", "not more than 30"  -> max   (value, unit)
    "must not", "prohibited", "never"                        -> prohibition
    anything else                                            -> obligation
"""
import re

from .engine import extract_quantity, split_blocks, split_sentences

REQUIREMENT = re.compile(r"\b(must|shall|should|required|requires|need to|needs to|has to|have to|"
                         r"may not|prohibited|forbidden|not allowed|not permitted|mandatory)\b", re.I)
PROHIBITION = re.compile(r"\b(must not|shall not|should not|may not|cannot|prohibited|forbidden|"
                         r"not allowed|not permitted|never)\b", re.I)
# "at least every 6 months" and "at least once a year" are frequencies: the gap may not be longer -> max
MAX_CUES = re.compile(r"\b(every|once|annually|yearly|quarterly|monthly|weekly|daily|within|at most|no more than|not more than|no longer than|not longer than|"
                      r"maximum|not exceed|no later than|not later than|up to)\b|\bnot\b.{0,25}\bmore than\b", re.I)
MIN_CUES = re.compile(r"\b(at least|minimum|no less than|not less than|no fewer than|or more|or longer)\b", re.I)


def detect_check(rule_text):
    quantity = extract_quantity(rule_text, want_characters=bool(re.search(r"\d[\s-]*char", rule_text, re.I)),
                                allow_same_day=False)
    if quantity:
        value, unit, _ = quantity
        if MAX_CUES.search(rule_text):
            return {"type": "max", "value": value, "unit": unit}
        if MIN_CUES.search(rule_text):
            return {"type": "min", "value": value, "unit": unit}
    if PROHIBITION.search(rule_text):
        return {"type": "prohibition"}
    return {"type": "obligation"}


def parse_policy(text):
    """Return [{"id", "area", "text", "check"}]. Headings in the document become the policy area."""
    rules, area = [], "General"
    for block in split_blocks(text):
        if block["heading"]:
            if len(block["text"].split()) <= 6:          # skip long document titles
                area = block["text"].title()
            continue
        sentences = [s for s in split_sentences(block["text"]) if REQUIREMENT.search(s)]
        for i, sentence in enumerate(sentences):
            rule_id = block["rule_id"] if (block["rule_id"] and i == 0) else f"R{len(rules) + 1}"
            rules.append({"id": rule_id, "area": area, "text": sentence, "check": detect_check(sentence)})
    return rules
