install:
	@pip install poetry
	@poetry install

format:
	@black main.py
	@black ./src

run: format
	@python main.py