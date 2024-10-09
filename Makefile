IMAGE_NAME ?= repepo
IMAGE_TAG ?= latest

PROJECT_URL ?= ghcr.io/alignmentresearch/${IMAGE_NAME}
DOCKERFILE ?= Dockerfile

export IMAGE_NAME
export PROJECT_URL
export DOCKERFILE

COMMIT_FULL ?= $(shell git rev-parse HEAD)
BRANCH_NAME ?= $(shell git branch --show-current)

# Release tag or latest if not a release
RELEASE_PREFIX ?= latest
BUILD_PREFIX ?= $(shell git rev-parse --short HEAD)

DEVBOX_UID ?= 1001
CPU ?= 1
MEMORY ?= 60G
GPU ?= 0
DEVBOX_NAME ?= ${IMAGE_NAME}-devbox-2

default: help

update-deps: # update dependency lock files
	pdm run python -m pip install --upgrade pip-tools pip wheel
	pdm run python -m piptools compile --upgrade --resolver backtracking -o requirements/main.txt requirements/main.in pyproject.toml
	pdm run python -m piptools compile --upgrade --resolver backtracking -o requirements/dev.txt requirements/dev.in

init: # initialize project in a virtualenv
	rm -rf .tox
	pdm run python -m pip install --upgrade pip wheel
	pdm run python -m pip install --upgrade -r requirements/main.txt -r requirements/dev.txt -e .
	pdm run python -m pip check

update: update-deps init # update dependencies and reinitialize project

.PHONY: update-deps init update

style: # check code style
	pdm run ruff .
	pdm run black --check --diff .

fmt: # run code formatter
	pdm run black .
	pdm run ruff --fix .

typecheck: # run pyright
	pdm run pyright

test: # run tests
	pdm run pytest

.PHONY: style fmt test typecheck

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


# Section 1: Build Dockerfiles

# Before running `make` to build the Dockerfile, you should delete the `requirements.txt` target then run `make
# release/main-pip-tools`
.build/${BUILD_PREFIX}/%: requirements.txt
	mkdir -p .build/${BUILD_PREFIX}
	docker pull "${PROJECT_URL}:${BUILD_PREFIX}-$*" || true
	docker build --platform "linux/amd64" \
		--tag "${PROJECT_URL}:${BUILD_PREFIX}-$*" \
		--target "$*" \
		-f "${DOCKERFILE}" .
	touch ".build/${BUILD_PREFIX}/$*"

# We use RELEASE_PREFIX as the image so we don't have to re-build it constantly. Once we have bootstrapped
# `requirements.txt`, we can push the image with `make release/main-pip-tools`
requirements.txt.new: pyproject.toml
	docker run -v "${HOME}/.cache:/home/dev/.cache" -v "$(shell pwd):/workspace" "${PROJECT_URL}:${RELEASE_PREFIX}-main-pip-tools" \
	pip-compile --verbose -o requirements.txt.new --extra=dev pyproject.toml

requirements.txt: requirements.txt.new
	sed -E "s/^(nvidia-.*|torchvision==.*|torch==.*|triton==.*)$$/# DISABLED \\1/g" requirements.txt.new > requirements.txt

.PHONY: local-install
local-install: requirements.txt
	pip install --no-deps -r requirements.txt
	pip install --config-settings editable_mode=compat -e ".[dev-local,launch-jobs]"


.PHONY: build build/%
build/%: .build/${BUILD_PREFIX}/%  # Build Docker image at some tag (e.g. main, main-pip-tools)
	true
build: build/main
	true

.PHONY: push push/%
push/%: .build/${BUILD_PREFIX}/%
	docker push "${PROJECT_URL}:${BUILD_PREFIX}-$*"
push: push/main
	true

.PHONY: release release/%
release/%: push/%  # Build and upload Docker image at some tag (e.g. main, main-pip-tools)
	docker tag "${PROJECT_URL}:${BUILD_PREFIX}-$*" "${PROJECT_URL}:${RELEASE_PREFIX}-$*"
	docker push "${PROJECT_URL}:${RELEASE_PREFIX}-$*"
release: release/main
	true


# Section 2: Make Devboxes and local devboxes (with Docker)

.PHONY: devbox devbox/%
devbox/%:
	git push
	CURRENT_NAMESPACE="$$(kubectl config view --minify -o jsonpath='{..namespace}' | cut -d'-' -f2)"
	python -c "print(open('cluster/devbox.yaml').read().format(NAME='${DEVBOX_NAME}', IMAGE='${PROJECT_URL}:${RELEASE_PREFIX}-$*', COMMIT_HASH='${COMMIT_FULL}', CPU='${CPU}', MEMORY='${MEMORY}', GPU='${GPU}', USER_ID=${DEVBOX_UID}, GROUP_ID=${DEVBOX_UID}, WANDB_PROJECT='${IMAGE_NAME}'))" | kubectl create -f -
devbox: devbox/main
	true

clean-devbox:
	kubectl delete job "${DEVBOX_NAME}" || true

.PHONY: cuda-devbox cuda-devbox/%
cuda-devbox/%: devbox/%
	true  # Do nothing, the body has to have something otherwise make complains

cuda-devbox: cuda-devbox/main
	true

.PHONY: docker docker/%
docker/%:
	docker run -v "$(shell pwd):/workspace" -it "${PROJECT_URL}:${RELEASE_PREFIX}-$*" /bin/bash
docker: docker/main
