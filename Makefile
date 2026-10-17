.PHONY: test backend-test frontend-check build

backend-test:
	python -m pytest -q backend

frontend-check:
	cd frontend && npm run lint && npm run typecheck

test: backend-test frontend-check

build:
	cd frontend && npm run build
