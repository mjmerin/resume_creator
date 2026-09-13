.DEFAULT_GOAL := all

UV ?= uv
PYTHON ?= .venv/bin/python
MATCH_PYTHON ?= python3
BUILD := $(PYTHON) scripts/build.py

.PHONY: setup init all sources variant match clean

setup:
	$(UV) venv --python 3.12
	$(UV) pip sync --python $(PYTHON) requirements.lock
	$(MAKE) init

init:
	@if [ -e data/profile.yaml ]; then \
		echo "data/profile.yaml already exists"; \
	else \
		cp data/profile.example.yaml data/profile.yaml; \
		echo "Created data/profile.yaml; replace the sample details before use"; \
	fi

all: init
	$(BUILD)

sources: init
	$(BUILD) --no-render

variant: init
	@test -n "$(VARIANT)" || (echo "Usage: make variant VARIANT=general"; exit 2)
	$(BUILD) $(VARIANT)

# Paste a job posting, then press Ctrl-D to evaluate the default resume variant locally.
match: init
	$(MATCH_PYTHON) scripts/evaluate_match.py $(if $(VARIANT),--variant $(VARIANT),) $(if $(JOB),--job-file $(JOB),)

clean:
	$(BUILD) --clean-only
