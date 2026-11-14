# Build the TPAMI version (main.pdf + supplement.pdf). `make data` regenerates every number, table and figure from the analysis outputs first.
.PHONY: all data pdf submission clean
all: data pdf
data:
	python3 scripts/make_tables.py
	python3 scripts/make_appendix.py
	python3 scripts/make_figures.py
	python3 scripts/make_v2.py
	python3 scripts/make_s3b.py
pdf:
	sh scripts/build_pdf.sh
submission:
	python3 scripts/make_submission_source.py
	python3 scripts/make_prereg_archive.py
clean:
	rm -f main.{aux,bbl,blg,log,out,pdf} supplement.{aux,bbl,blg,log,out,pdf}
