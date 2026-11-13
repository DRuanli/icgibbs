"""
make_submission_source.py — build the anonymous LaTeX source package for TMLR from the paper sources.

What it does (on a copy; the working sources are not changed):
  * copies main.tex, sections/, tables/, figures/, numbers*.tex, macros.tex, refs.bib, tmlr.sty, tmlr.bst, fancyhdr.sty;
  * removes every LaTeX comment (text after an unescaped %), which in the working sources point to internal documents;
  * keeps only bibliography entries that are cited;
  * checks the result for strings that could break anonymity (repository names, local paths, author or institution
    names given with --forbid) and stops if one is found;
  * compiles the copy and writes paper/_submission/source.zip and paper/_submission/main.pdf.
Run (from paper/): python3 scripts/make_submission_source.py [--forbid NAME ...]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
OUT = os.path.join(PAPER, "_submission")
COPY = ["main.tex", "macros.tex", "numbers.tex", "numbers_appendix.tex", "numbers_v2.tex", "numbers_s3b.tex",
        "refs.bib", "tmlr.sty", "tmlr.bst", "fancyhdr.sty"]
DIRS = ["sections", "tables", "figures"]
ALWAYS_FORBID = ["icgibbs", "de-tai-moi", "DRuanli", "/home/", "neur-v3", "G4/", "G9/", "V2/", "S3/raw", "HANDOFF",
                 "Kaggle notebook", "github.com/DRuanli"]


def strip_comments(text):
    out = []
    for line in text.splitlines():
        m = re.search(r"(?<!\\)%", line)
        if m:
            kept = line[:m.start()].rstrip()
            if not kept:          # whole-line comment: drop the line, keep paragraph structure
                continue
            out.append(kept + "%")  # keep the % so that no space is introduced at the line end
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--forbid", nargs="*", default=[], help="further strings that must not appear (author names, …)")
    a = ap.parse_args(argv)
    shutil.rmtree(OUT, ignore_errors=True)
    src = os.path.join(OUT, "src")
    os.makedirs(src)
    for f in COPY:
        shutil.copy(os.path.join(PAPER, f), src)
    for d in DIRS:
        shutil.copytree(os.path.join(PAPER, d), os.path.join(src, d))
    for root, _, files in os.walk(src):
        for f in files:
            if f.endswith((".tex", ".sty")) and f not in ("tmlr.sty", "fancyhdr.sty"):
                p = os.path.join(root, f)
                text = open(p).read()          # read before opening for writing (which truncates)
                open(p, "w").write(strip_comments(text))
    # compile once to find the cited keys, then prune the bibliography
    run = lambda: subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                                 cwd=src, capture_output=True, text=True)
    r = run()
    if r.returncode:
        print(r.stdout[-3000:])
        return 1
    cited = set()
    for line in open(os.path.join(src, "main.aux")):
        for m in re.finditer(r"\\(?:citation|abx@aux@cite)\{([^}]*)\}", line):
            cited |= set(k.strip() for k in m.group(1).split(","))
    bib = "\n".join(l for l in open(os.path.join(src, "refs.bib")).read().splitlines() if not l.lstrip().startswith("%"))
    entries = re.split(r"(?=^@)", bib, flags=re.M)
    keep = [e for e in entries if (m := re.match(r"@\w+\{([^,]+),", e)) and m.group(1).strip() in cited]
    open(os.path.join(src, "refs.bib"), "w").write("\n".join(e.strip() for e in keep) + "\n")
    for ext in (".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".pdf"):
        p = os.path.join(src, "main" + ext)
        if os.path.exists(p):
            os.remove(p)
    r = run()
    if r.returncode:
        print(r.stdout[-3000:])
        return 1
    bad = []
    for root, _, files in os.walk(src):
        for f in files:
            if f.endswith((".tex", ".bib", ".bbl")):
                t = open(os.path.join(root, f), errors="replace").read()
                for s in ALWAYS_FORBID + a.forbid:
                    if s in t:
                        bad.append(f"{os.path.relpath(os.path.join(root, f), src)}: '{s}'")
    pdftext = subprocess.run(["pdftotext", os.path.join(src, "main.pdf"), "-"], capture_output=True, text=True).stdout
    for s in ALWAYS_FORBID + a.forbid:
        if s in pdftext:
            bad.append(f"main.pdf: '{s}'")
    if bad:
        print("Anonymity check FAILED:\n  " + "\n  ".join(bad))
        return 2
    shutil.copy(os.path.join(src, "main.pdf"), os.path.join(OUT, "main.pdf"))
    with zipfile.ZipFile(os.path.join(OUT, "source.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(src):
            for f in files:
                if f.endswith((".aux", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".pdf")) and f.startswith("main"):
                    continue
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, src))
    print(f"ok: {OUT}/source.zip, {OUT}/main.pdf; {len(keep)} cited references kept of {len(entries) - 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
