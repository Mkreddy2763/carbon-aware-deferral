# make all: regenerate every table, figure and the paper from results/raw/.
PY ?= python
EXPS = E1 E2 E3 E4 E5 E6a E6b E6c E6d E6e E6f FIG
EXPS_V2 = V2U V2DL V2DA

.PHONY: all tables figs paper selftest data forecasts sweep clean-analysis
all: tables figs paper

tables:
	$(PY) analysis/make_tables.py --raw results/raw results/raw_v2 --out analysis/out --macros paper/numbers.tex --tex-tables paper/tables

figs: tables
	$(PY) analysis/make_figs_v2.py --tables analysis/out --out paper/figs

paper: tables figs
	cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null; \
	pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex | grep -E "Warning|Error" | grep -v "Font" || true

selftest:
	bash scripts/selftest.sh

# --- workstation path (the cluster path is cluster/setup.sh + cluster/submit.sh)
data:
	$(PY) -m carbonsched.data.download --what alibaba --out data/raw/alibaba
	$(PY) -m carbonsched.data.fetch_gb --out data/raw/gb --start 2021-01-01 --end 2026-01-01
	bash cluster/fetch_gb_archive.sh
	$(PY) -m carbonsched.data.download --what eia --out data/raw/eia
	$(PY) -m carbonsched.data.prep && $(PY) -m carbonsched.data.gb_archive && $(PY) -m carbonsched.data.trace
	$(PY) -m carbonsched.data.verify

forecasts:
	for r in GB GBop CISO ERCO PJM MISO; do $(PY) -m carbonsched.forecast.prepare --region $$r; done

sweep:
	$(PY) -m carbonsched.run --exp $(EXPS) --n-jobs 20
	$(PY) -m carbonsched.run --exp $(EXPS_V2) --config configs/experiments_v2.yaml --out results/raw_v2 --n-jobs 20

clean-analysis:
	rm -rf analysis/out paper/figs/*.pdf paper/numbers.tex paper/tables
