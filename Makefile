install:
	@pip install poetry
	@poetry install

format:
	@black ./src
