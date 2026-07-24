.PHONY: install run test clean

install:
	pip install -r requirements.txt

run:
	python -m pipeline

test:
	pytest -q

clean:
	rm -rf data/raw data/mentions.db reports/summary.md reports/evaluation.md
