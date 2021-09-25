.PHONY: install format check

install:
	@pip install poetry
	@poetry install

format:
	@black main.py
	@isort main.py
	@black ./src
	@isort ./src

run: format
	@python main.py

check:
	@black --check -l 79 src/
	@isort --check src/
	@flake8 src/
