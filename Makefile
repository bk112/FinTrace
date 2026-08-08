.PHONY: test compile

compile:
	python -m compileall -q src tests

test:
	PYTHONPATH=src python tests/unit/test_rewards.py
