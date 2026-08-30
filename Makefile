.DEFAULT_GOAL := all

UV ?= uv
PYTHON ?= .venv/bin/python
BUILD := $(PYTHON) scripts/build.py

.PHONY: setup init all sources variant clean

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

clean:
	$(BUILD) --clean-only
