"""
make_supplement.py — build the anonymous supplementary archive for TMLR (review round 3, finding F10).

What it contains (paper/_submission/supplement.zip):
  README.md            what is inside, how to check it, and what it cannot show
  freeze_log.csv       every freeze event (UTC time from the commit) with the SHA-256 of each frozen file as it was in
                       that commit; commit identifiers are withheld because the project repository is public
  scoring/             the scoring scripts of Studies 1-3 and of the supplementary analysis, final versions, with
                       identifying strings replaced (listed in README); SHA-256 of the unmodified files in freeze_log.csv
  results_sha256.csv   SHA-256 of every raw result file (per-cell CSVs) the paper's numbers are computed from

The frozen protocols themselves (written in Vietnamese, with internal notes) are not included; their hashes are, so
that the files released with the paper can be checked against them.
Run (from paper/): python3 scripts/make_supplement.py
"""
import csv
import glob
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)                      # de-tai-moi
REPO = os.path.dirname(ROOT)
OUT = os.path.join(PAPER, "_submission")
PFX = os.path.basename(ROOT) + "/"

# (event, commit, files, note); commits are resolved here and never written to the archive
EVENTS = [
    ("Study 1 protocol frozen (hypotheses, tests, datasets, baselines)", "c4da1f4", ["G5/EXPERIMENT_PROTOCOL.md"], ""),
    ("Study 1 scoring script first committed", "71eab4b", ["G8/analysis/g8_confirmatory.py"],
     "committed together with the first evaluation results; implements the frozen statistical plan"),
    ("Study 2 plan frozen, before any Study 2 table was downloaded", "aaee9fd", ["S2/S2_PLAN.md"], ""),
    ("Study 2 scoring script first committed", "40456af", ["S2/analysis/s2_confirmatory.py"],
     "committed together with the first results; one later fix of a floating-point tie is reported in the paper"),
    ("Study 3 protocol frozen, before its code, scoring script or data download", "c9009e8",
     ["S3/STUDY3_PLAN.md", "S3/s3_review.csv"], ""),
    ("Study 3 code and scoring script committed, before any Study 3 result", "7cc9679", ["S3/s3_report.py"], ""),
    ("Study 3 scored", "aa0775d", ["S3/results/s3_summary.json"], ""),
    ("Supplementary analysis S3B: plan, scoring script and prediction frozen, before any run", "d9e6062",
     ["S3/S3B_PLAN.md", "S3/s3b_report.py"], ""),
    ("S3B code committed, before any run", "b59ef14", [], "the hashes of the 87 code files of every run match this commit"),
    ("S3B session 1 results and scoring (B3 provisional)", "392b8de", ["S3/results/s3b_summary.json"], ""),
    ("S3B session 2 results and final scoring", "ce0fe93", ["S3/results/s3b_summary.json"], ""),
]
SCORING = ["G8/analysis/g8_confirmatory.py", "S2/analysis/s2_confirmatory.py", "S3/s3_report.py", "S3/s3b_report.py"]
RESULTS = ["G7/raw/**/*.csv", "S2/raw/**/*.csv", "S3/raw/S3/*.csv", "S3/raw/S3B/*.csv"]
REPLACE = [("icgibbs", "icg"), ("de-tai-moi", "project"), ("DRuanli", "anon")]
FORBID = ["icgibbs", "de-tai-moi", "DRuanli", "/home/", "neur-v3", "github.com/"]


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, check=True).stdout


def sha(b):
    return hashlib.sha256(b).hexdigest()


def sanitize(text):
    for a, b in REPLACE:
        text = text.replace(a, b)
    return text


def main():
    if git("rev-parse", "--is-shallow-repository").strip() == b"true":
        print("the clone is shallow: run `git fetch --unshallow` first")
        return 1
    stage = os.path.join(OUT, "supplement")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(os.path.join(stage, "scoring"))
    rows = []
    for ev, c, files, note in EVENTS:
        t = git("show", "-s", "--format=%cI", c).decode().strip()
        if not files:
            rows.append(dict(event=ev, utc_time=t, file="", sha256="", note=note))
        for f in files:
            b = git("show", f"{c}:{PFX}{f}")
            rows.append(dict(event=ev, utc_time=t, file=sanitize(f), sha256=sha(b), note=note))
    for f in SCORING:                                              # final versions, as used for the paper
        b = open(os.path.join(ROOT, f), "rb").read()
        rows.append(dict(event="final version used for the paper's numbers", utc_time="", file=sanitize(f),
                         sha256=sha(b), note="unmodified file; the copy in scoring/ has identifiers replaced"))
        dst = os.path.join(stage, "scoring", os.path.basename(f))
        open(dst, "w").write(sanitize(b.decode()))
    with open(os.path.join(stage, "freeze_log.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["event", "utc_time", "file", "sha256", "note"])
        w.writeheader()
        w.writerows(rows)
    res = []
    for pat in RESULTS:
        for f in sorted(glob.glob(os.path.join(ROOT, pat), recursive=True)):
            res.append((sanitize(os.path.relpath(f, ROOT)), sha(open(f, "rb").read())))
    with open(os.path.join(stage, "results_sha256.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "sha256"])
        w.writerows(res)
    readme = f"""# Supplementary material: pre-registration record

This archive documents the pre-registration of the three studies and of the supplementary analysis.

* `freeze_log.csv`: each freeze event with the UTC time of its commit and the SHA-256 of every frozen file as it was in
  that commit. Commit identifiers are withheld for anonymity.
* `scoring/`: the scoring scripts that produced the registered results (final versions). Identifying strings (the
  names of the repository, of the project directory and of the code package) were replaced; `freeze_log.csv` gives the
  SHA-256 of the unmodified files. Comments are partly in Vietnamese, the working language of the project. They import project modules that will
  be released with the code.
* `results_sha256.csv`: SHA-256 of the {len(res)} raw per-cell result files from which every number of the paper is
  computed.

What this shows and what it does not. Commit times are set by us, so this record documents the order of events but
does not prove it. For Study 3 and the supplementary analysis, the scoring scripts were committed before any result;
for Studies 1 and 2 they were first committed together with the first results, so their timing is not documented by a
commit (the frozen protocols, with the hypotheses, tests and decision rules, were committed before any evaluation data
were used). The frozen protocols and the code will be released with the paper; the hashes here let anyone check that
the released files are the ones referred to.
"""
    open(os.path.join(stage, "README.md"), "w").write(readme)
    bad = []
    for r, _, fs in os.walk(stage):
        for f in fs:
            txt = open(os.path.join(r, f), errors="replace").read()
            bad += [f"{f}: {s}" for s in FORBID if s in txt]
            bad += [f"{f}: email" for _ in re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", txt)]
    if bad:
        print("Anonymity check FAILED:\n  " + "\n  ".join(sorted(set(bad))))
        return 2
    zp = os.path.join(OUT, "supplement.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for r, _, fs in os.walk(stage):
            for f in fs:
                p = os.path.join(r, f)
                z.write(p, os.path.relpath(p, OUT))
    print(f"ok: {zp}; {len(rows)} freeze-log rows, {len(res)} result files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
