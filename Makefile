# Local helpers. Training and rendering run on the cluster (see docs/hpc.md).
.PHONY: figures workflow report overleaf clean

figures: workflow     ## rebuild the final report figures from outputs/ (needs Pillow + ffmpeg)
	cd pipeline/figures && python3 final_figures.py

workflow:             ## the workflow diagram
	cd pipeline/figures && pdflatex -interaction=nonstopmode workflow.tex >/dev/null && cp workflow.pdf ../../report/final/figures/workflow.pdf

report:               ## build report/final/main.pdf
	cd report/final && latexmk -pdf -interaction=nonstopmode -quiet main.tex

overleaf:             ## report/final/final_report_overleaf.zip (upload to Overleaf as a new project)
	cd report/final && rm -f final_report_overleaf.zip && zip -qr final_report_overleaf.zip main.tex main.bib cvpr.sty ieeenat_fullname.bst figures

clean:                ## remove LaTeX build files and Python caches
	cd report/final && latexmk -c >/dev/null 2>&1 || true
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
