
.PHONY: datasets stereoset truthfulqa ambigprompt

stereoset: # Make Stereoset
	python generation/make_stereoset.py

truthfulqa: # Make TruthfulQA
	python generation/make_truthfulqa.py

ambigprompt: # Make Ambigprompt
	python download/download_ambigprompt.py

datasets: # Make all datasets
	make stereoset
	make truthfulqa
	make ambigprompt

.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done
