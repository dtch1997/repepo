
.PHONY: datasets stereoset truthfulqa ambigprompt

stereoset: # Make Stereoset
	python generation/make_stereoset.py

truthfulqa: # Make TruthfulQA
	python multiple_choice/make_truthfulqa.py

mwe: # Make Anthropic MWE datasets
	python multiple_choice/make_mwe.py
# ambigprompt: # Make Ambigprompt
# 	python download/download_ambigprompt.py

sycophancy: # Make Sycophancy
	python multiple_choice/make_sycophancy.py

datasets: # Make all datasets
	make stereoset
	make truthfulqa
	make mwe
	make sycophancy

.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done
