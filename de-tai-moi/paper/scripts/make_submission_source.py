"""
make_submission_source.py — build the LaTeX source package for the TPAMI submission from the paper sources.

What it does (on a copy; the working sources are not changed):
  * copies main.tex, supplement.tex, sections/, tables/, figures/, numbers*.tex, macros.tex, refs.bib, IEEEtran.cls,
    IEEEtranN.bst and scripts/build_pdf.sh;
  * removes every LaTeX comment (text after an unescaped %), which in the working sources point to internal documents;
  * keeps only bibliography entries cited by the paper or the supplement;
  * compiles the copy and writes paper/_submission/source.zip, main.pdf and supplement.pdf;
  * with --anonymous (only if double-anonymous review is requested), checks the result for strings that could break
    anonymity (repository names, local paths, names given with --forbid) and stops if one is found.
  It always stops if a TODO marker is left in the PDFs and --allow-todo is not given.
Run (from paper/): python3 scripts/make_submission_source.py [--allow-todo] [--anonymous --forbid NAME ...]
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
COPY = ["main.tex", "supplement.tex", "macros.tex", "numbers.tex", "numbers_appendix.tex", "numbers_v2.tex",
        "numbers_s3b.tex", "refs.bib", "IEEEtran.cls", "IEEEtranN.bst"]
DIRS = ["sections", "tables", "figures"]
ALWAYS_FORBID = ["icgibbs", "de-tai-moi", "DRuanli", "/home/", "neur-v3", "G4/", "G9/", "V2/", "S3/raw", "HANDOFF",
                 "Kaggle notebook", "github.com/DRuanli"]
DOCS = ("main", "supplement")


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
    ap.add_argument("--anonymous", action="store_true", help="check anonymity (double-anonymous review only)")
    ap.add_argument("--forbid", nargs="*", default=[], help="further strings that must not appear (with --anonymous)")
    ap.add_argument("--allow-todo", action="store_true", help="package a draft that still has TODO markers")
    a = ap.parse_args(argv)
    shutil.rmtree(OUT, ignore_errors=True)
    src = os.path.join(OUT, "src")
    os.makedirs(os.path.join(src, "scripts"))
    for f in COPY:
        shutil.copy(os.path.join(PAPER, f), src)
    shutil.copy(os.path.join(HERE, "build_pdf.sh"), os.path.join(src, "scripts"))
    for d in DIRS:
        shutil.copytree(os.path.join(PAPER, d), os.path.join(src, d))
    for root, _, files in os.walk(src):
        for f in files:
            if f.endswith(".tex"):
                p = os.path.join(root, f)
                text = open(p).read()          # read before opening for writing (which truncates)
                open(p, "w").write(strip_comments(text))
    build = lambda: subprocess.run(["sh", "scripts/build_pdf.sh"], cwd=src, capture_output=True, text=True)
    r = build()
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-2000:])
        return 1
    cited = set()
    for doc in DOCS:
        for line in open(os.path.join(src, doc + ".aux")):
            for m in re.finditer(r"\\citation\{([^}]*)\}", line):
                cited |= set(k.strip() for k in m.group(1).split(","))
    bib = "\n".join(l for l in open(os.path.join(src, "refs.bib")).read().splitlines() if not l.lstrip().startswith("%"))
    entries = re.split(r"(?=^@)", bib, flags=re.M)
    keep = [e for e in entries if (m := re.match(r"@\w+\{([^,]+),", e)) and m.group(1).strip() in cited]
    open(os.path.join(src, "refs.bib"), "w").write("\n".join(e.strip() for e in keep) + "\n")
    for doc in DOCS:
        for ext in (".aux", ".bbl", ".blg", ".log", ".out", ".pdf"):
            p = os.path.join(src, doc + ext)
            if os.path.exists(p):
                os.remove(p)
    r = build()
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-2000:])
        return 1
    print(r.stdout.strip())
    bad = []
    texts = {doc: subprocess.run(["pdftotext", os.path.join(src, doc + ".pdf"), "-"], capture_output=True,
                                 text=True).stdout for doc in DOCS}
    if not a.allow_todo:
        bad += [f"{doc}.pdf: TODO marker left" for doc, t in texts.items() if "[TODO:" in t]
    if a.anonymous:
        for root, _, files in os.walk(src):
            for f in files:
                if f.endswith((".tex", ".bib", ".bbl")):
                    t = open(os.path.join(root, f), errors="replace").read()
                    bad += [f"{os.path.relpath(os.path.join(root, f), src)}: '{s}'" for s in ALWAYS_FORBID + a.forbid
                            if s in t]
        for doc, t in texts.items():
            bad += [f"{doc}.pdf: '{s}'" for s in ALWAYS_FORBID + a.forbid if s in t]
    if bad:
        print("Check FAILED:\n  " + "\n  ".join(bad))
        return 2
    for doc in DOCS:
        shutil.copy(os.path.join(src, doc + ".pdf"), os.path.join(OUT, doc + ".pdf"))
    with zipfile.ZipFile(os.path.join(OUT, "source.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(src):
            for f in files:
                if f.endswith((".aux", ".blg", ".log", ".out", ".pdf")) and f.split(".")[0] in DOCS:
                    continue
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, src))
    print(f"ok: {OUT}/source.zip, main.pdf, supplement.pdf; {len(keep)} cited references kept of {len(entries) - 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
