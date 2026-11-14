.PHONY: run test seed

UV   ?= uv
HOST ?= 127.0.0.1
PORT ?= 8000

run:
	$(UV) run uvicorn app.main:create_app --factory --host $(HOST) --port $(PORT)

test:
	$(UV) run pytest

seed:
	$(UV) run python -m app.seed --base-url http://$(HOST):$(PORT)
