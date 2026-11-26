# Local helpers (the training and rendering run on the cluster, see docs/hpc.md)
.PHONY: figures report bundle

figures:            ## rebuild report figures from media/ (needs Pillow + ffmpeg)
	python3 tools/compose_figures.py
	cd tools && pdflatex -interaction=nonstopmode workflow.tex >/dev/null && cp workflow.pdf ../report/midterm/figures/workflow.pdf

report:             ## build report/midterm/main.pdf
	cd report/midterm && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null \
	  && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex

bundle:             ## (no git on the cluster) zip the repo for upload, with MANIFEST.sha256
	python3 tools/make_bundle.py
