# SysPlug Paper Makefile

.PHONY: paper figures experiments clean

paper: paper.pdf

paper.pdf: paper.tex refs.bib figures/comparison_table.tex figures/system_overview.tex
	pdflatex -interaction=nonstopmode paper.tex
	bibtex paper
	pdflatex -interaction=nonstopmode paper.tex
	pdflatex -interaction=nonstopmode paper.tex

figures:
	python figures/memory_model_error.py --mock --output figures/memory_model_error.pdf
	python figures/throughput_comparison.py --mock --output figures/throughput_comparison.pdf

experiments:
	bash experiments/run_sft_eval.sh
	bash experiments/run_dpo_eval.sh
	bash experiments/run_vision_eval.sh
	bash experiments/run_prediction_bench.sh
	python experiments/collect_results.py

clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out
	rm -f figures/*.pdf
