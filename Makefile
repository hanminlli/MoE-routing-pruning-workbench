.PHONY: test audit check plans

test:
	PYTHONPATH=src:. pytest -q

audit:
	python scripts/validation/audit_public_tree.py --root .

check:
	bash scripts/validation/run_repository_checks.sh

plans:
	@test -n "$(ACCOUNTING)" || (echo "Set ACCOUNTING=/path/to/ordinary_response_routing_by_task.csv.gz" && exit 1)
	ACCOUNTING="$(ACCOUNTING)" TASK_METADATA="$(TASK_METADATA)" bash scripts/experiments/prepare_all_phase1_plans.sh
