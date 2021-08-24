install:
	@pip install poetry
	@poetry install

format:
	@black -l 79 src/
	@isort src/

# run: format
# 	@python main.py

check:
	@black --check -l 79 src/
	@isort --check src/
	@flake8 src/