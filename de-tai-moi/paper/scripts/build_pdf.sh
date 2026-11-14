#!/bin/sh
# Build main.pdf (the paper) and supplement.pdf (all appendices) for TPAMI. The two files cross-reference each other
# through xr-hyper, so they are compiled alternately; run from paper/.
set -e
run() { pdflatex -interaction=nonstopmode -halt-on-error "$1" > /dev/null || { tail -40 "${1%.tex}.log"; exit 1; }; }
run main.tex; run supplement.tex
bibtex main > /dev/null; bibtex supplement > /dev/null
run main.tex; run supplement.tex; run main.tex; run supplement.tex
for f in main supplement; do
  if grep -q "LaTeX Warning: Reference\|LaTeX Warning: There were undefined\|Citation .* undefined" "$f.log"; then
    echo "$f: undefined references or citations"; exit 1; fi
  if grep -q "Overfull" "$f.log"; then echo "$f: overfull boxes:"; grep -c "Overfull" "$f.log"; fi
done
echo "built main.pdf ($(pdfinfo main.pdf | awk '/Pages/{print $2}') pages) and supplement.pdf ($(pdfinfo supplement.pdf | awk '/Pages/{print $2}') pages)"
