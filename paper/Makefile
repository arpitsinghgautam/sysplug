# SysPlug Paper Makefile

.PHONY: paper figures experiments clean

paper: paper.pdf

paper.pdf: paper.tex refs.bib figures/comparison_table.tex figures/system_overview.tex
	pdflatex -interaction=nonstopmode paper.tex
	bibtex paper
	pdflatex -interaction=nonstopmode paper.tex
	pdflatex -interaction=nonstopmode paper.tex

MEASURE_JSON = data/gpu_measurements.json

# Real measured figures (no mock data). The committed measurement JSON and PNGs
# are reused; run `make experiments` on a CUDA GPU to regenerate the data.
figures:
	python figures/throughput_comparison.py --results $(MEASURE_JSON) --output figures/throughput_comparison.png
	python figures/memory_model_error.py    --results $(MEASURE_JSON) --output figures/memory_model_error.png

# Run the real validation sweep (forward+backward+optimizer on GPT-2) and write
# the measurements the paper's tables and figures are built from.
experiments:
	cd .. && python -m paper.experiments.measure_gpu \
		--configs gpt2-small gpt2-medium gpt2-large \
		--out paper/data/gpu_measurements.json

clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out
	rm -f figures/*.pdf
