
.PHONY: datasets stereoset truthfulqa ambigprompt

stereoset: # Make Stereoset
	python3 -m repepo.data.generation.make_stereoset

truthfulqa: # Make TruthfulQA
	python3 -m repepo.data.generation.make_truthfulqa

ambigprompt: # Make Ambigprompt
	python3 -m repepo.data.download.download_ambigprompt

datasets: # Make all datasets
	make stereoset
	make truthfulqa
	make ambigprompt

.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done
