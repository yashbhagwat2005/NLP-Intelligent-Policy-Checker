"""
Policy Compliance Checker - web app.

    python app.py        then open http://127.0.0.1:5000

Pages
    /            Policy check   : a policy document + procedure documents  -> report
    /contract    Contract check : one non-disclosure agreement             -> report
    /method      How the checks work and how accurate they are
"""
import csv
import glob
import io
import json
import os
import uuid
from datetime import datetime

from flask import Flask, Response, abort, redirect, render_template, request, url_for

from checker import contract_model, engine
from checker.policy_parser import parse_policy
from checker.readers import read_upload

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(ROOT, "data", "samples")
SAMPLE_PROCEDURES = os.path.join(ROOT, "data", "custom", "procedures")
STATUSES = ["NON_COMPLIANT", "NOT_ADDRESSED", "COMPLIANT"]
STATUS_NAMES = {"COMPLIANT": "Compliant", "NON_COMPLIANT": "Non-compliant", "NOT_ADDRESSED": "Not addressed"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024    # uploads up to 10 MB
REPORTS = {}                                           # report id -> report, kept while the app runs


def read_text_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def save_report(kind, policy_name, documents, rows, n_clauses):
    counts = {s: sum(r["status"] == s for r in rows) for s in STATUSES}
    areas = []
    for name in dict.fromkeys(r["area"] for r in rows if r["area"]):      # keeps document order
        area_rows = [r for r in rows if r["area"] == name]
        areas.append({"name": name, "total": len(area_rows),
                      **{s: sum(r["status"] == s for r in area_rows) for s in STATUSES}})
    report_id = uuid.uuid4().hex[:8]
    REPORTS[report_id] = {
        "id": report_id, "kind": kind, "policy_name": policy_name, "documents": documents,
        "created": datetime.now().strftime("%d %b %Y, %H:%M"), "rows": rows, "counts": counts,
        "areas": areas, "n_clauses": n_clauses, "n_weak": sum(r["weak"] for r in rows),
        "score": round(100 * counts["COMPLIANT"] / len(rows)),
    }
    return report_id


# ----------------------------------------------------------------- policy check
@app.get("/")
def policy_form(error=None):
    return render_template("policy.html", page="policy", error=error, model_ready=engine.embedder_ready())


@app.post("/check/policy")
def run_policy_check():
    try:
        if request.form.get("sample"):
            policy_name = "NovaTech Policy Handbook (sample)"
            policy_text = read_text_file(os.path.join(SAMPLES, "NovaTech_Policy_Handbook.txt"))
            documents = [(os.path.basename(p)[:-4].replace("_", " "), read_text_file(p))
                         for p in sorted(glob.glob(os.path.join(SAMPLE_PROCEDURES, "*.txt")))]
        else:
            policy_file = request.files.get("policy_file")
            if policy_file and policy_file.filename:
                policy_name, policy_text = read_upload(policy_file)
            else:
                policy_name, policy_text = "Pasted policy", request.form.get("policy_text", "")
            documents = [read_upload(f) for f in request.files.getlist("documents") if f.filename]

        if not policy_text.strip():
            raise ValueError("Add a policy first: upload a file or paste the rules.")
        if not documents:
            raise ValueError("Upload at least one document to check.")
        rules = parse_policy(policy_text)
        if not rules:
            raise ValueError('No rules were found in the policy. A rule is a sentence that uses a word '
                             'such as "must", "shall", "required" or "prohibited".')
        clauses = []
        for name, text in documents:
            clauses += engine.split_clauses(text, name)
        if not clauses:
            raise ValueError("The uploaded documents contain no readable sentences.")
    except ValueError as problem:
        return policy_form(error=str(problem)), 400

    rows = engine.check_policies(rules, clauses)
    report_id = save_report("policy", policy_name, [name for name, _ in documents], rows, len(clauses))
    return redirect(url_for("show_report", report_id=report_id))


# ----------------------------------------------------------------- contract check
@app.get("/contract")
def contract_form(error=None):
    return render_template("contract.html", page="contract", error=error)


@app.post("/check/contract")
def run_contract_check():
    try:
        if request.form.get("sample"):
            name, text = "Sample NDA", read_text_file(os.path.join(SAMPLES, "Sample_NDA.txt"))
        else:
            upload = request.files.get("contract_file")
            if upload and upload.filename:
                name, text = read_upload(upload)
            else:
                name, text = "Pasted contract", request.form.get("contract_text", "")
        if not text.strip():
            raise ValueError("Add a contract first: upload a file or paste its text.")
        rows, n_clauses = contract_model.check_contract(text, name)
        if n_clauses < 5:
            raise ValueError("The contract is too short to check. Please provide the full agreement.")
    except ValueError as problem:
        return contract_form(error=str(problem)), 400

    report_id = save_report("contract", "17 standard NDA rules (ContractNLI)", [name], rows, n_clauses)
    return redirect(url_for("show_report", report_id=report_id))


# ----------------------------------------------------------------- report
@app.get("/report/<report_id>")
def show_report(report_id):
    report = REPORTS.get(report_id) or abort(404)
    return render_template("report.html", page=report["kind"], report=report,
                           statuses=STATUSES, names=STATUS_NAMES)


@app.get("/report/<report_id>/csv")
def download_csv(report_id):
    report = REPORTS.get(report_id) or abort(404)
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["Rule", "Area", "Requirement", "Status", "Evidence", "Source document", "Reason",
                     "Score", "Needs review"])
    for r in report["rows"]:
        writer.writerow([r["rule_id"], r["area"], r["rule"], STATUS_NAMES[r["status"]], r["evidence"],
                         r["source"], r["reason"], r["score"], "yes" if r["weak"] else ""])
    return Response("﻿" + out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=compliance_report_{report_id}.csv"})


# ----------------------------------------------------------------- method page
@app.get("/method")
def method():
    with open(os.path.join(ROOT, "outputs", "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    return render_template("method.html", page="method", metrics=metrics)


@app.errorhandler(404)
def not_found(_):
    return render_template("message.html", page="", title="Report not found",
                           text="Reports are kept only while the app is running. Please run the check again."), 404


@app.errorhandler(413)
def too_large(_):
    return render_template("message.html", page="", title="File too large",
                           text="Uploads are limited to 10 MB in total."), 413


if __name__ == "__main__":
    engine.start_loading_embedder()      # loads in the background; the first policy check waits for it
    contract_model.load()
    print("\n  Policy Compliance Checker is running at http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
