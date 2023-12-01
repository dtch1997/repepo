IMAGE_NAME ?= repepo
IMAGE_TAG ?= latest

default: help

update-deps: # update dependency lock files
	python -m pip install --upgrade pip-tools pip wheel
	python -m piptools compile --upgrade --resolver backtracking -o requirements/main.txt requirements/main.in pyproject.toml
	python -m piptools compile --upgrade --resolver backtracking -o requirements/dev.txt requirements/dev.in

init: # initialize project in a virtualenv
	rm -rf .tox
	python -m pip install --upgrade pip wheel
	python -m pip install --upgrade -r requirements/main.txt -r requirements/dev.txt -e .
	python -m pip check

update: update-deps init # update dependencies and reinitialize project

.PHONY: update-deps init update

style: # check code style
	ruff .
	black --check --diff .

fmt: # run code formatter
	black .
	ruff --fix .

typecheck: # run pyright
	pyright

test: # run tests
	pytest

.PHONY: style fmt test

.PHONY: build-docker
build-docker: # build docker container
	docker buildx build --tag $(IMAGE_NAME):$(IMAGE_TAG) -f cluster/Dockerfile .

.PHONY: build-singularity
build-singularity: build-docker # build singularity container
	singularity build cluster/$(IMAGE_NAME)-$(IMAGE_TAG).sif docker-daemon://$(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: shell-singularity
shell-singularity:
	singularity shell --env PYTHONPATH=$(shell pwd) --nv cluster/$(IMAGE_NAME)-$(IMAGE_TAG).sif

.PHONY: clean-singularity
clean-singularity: # remove singularity container
	rm cluster/*.sif *.sif

.PHONY: clean
clean: # clean up
	rm -rf build .cache

.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done
