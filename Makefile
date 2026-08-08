.PHONY: test compile

compile:
	python -m compileall -q src tests

test:
	PYTHONPATH=src python tests/unit/test_rewards.py
	PYTHONPATH=src python -m unittest discover -s tests/unit -p 'test_*.py'
