# Build the draft. `make data` regenerates every number, table and figure from the analysis outputs first.
.PHONY: all data pdf submission clean
all: data pdf
data:
	python3 scripts/make_tables.py
	python3 scripts/make_appendix.py
	python3 scripts/make_figures.py
	python3 scripts/make_v2.py
	python3 scripts/make_s3b.py
pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
submission:
	python3 scripts/make_submission_source.py
	python3 scripts/make_supplement.py
clean:
	latexmk -C main.tex
