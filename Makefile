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