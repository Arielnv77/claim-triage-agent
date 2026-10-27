.PHONY: setup start stop test clean logs health

setup:
	docker compose up --build -d --wait --wait-timeout 120
	@$(MAKE) health
start:
	docker compose up -d --wait --wait-timeout 120
stop:
	docker compose down
# Application tests: run pytest inside the running api container. No LLM calls.
test:
	docker compose exec -T api python -m pytest -q tests
health:
	curl --fail --silent --show-error --retry 10 --retry-connrefused --retry-delay 2 http://localhost:8000/api/v1/health
	curl --fail --silent --show-error http://localhost:3000 > /dev/null
logs:
	docker compose logs -f
# Stops only this project. Persistent volumes are preserved.
clean:
	docker compose down --remove-orphans
