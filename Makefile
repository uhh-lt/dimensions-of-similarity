export TOP_LEVEL_PYTHON_FILES=main.py
export SRC_DIR="dos"

format:
	python -c 'import black; print(black.__version__)'
	python -m black $(SRC_DIR) $(TOP_LEVEL_PYTHON_FILES)
	python -m isort . --profile black

types:
	python -m mypy $(SRC_DIR) $(TOP_LEVEL_PYTHON_FILES)

install-hooks:
	printf "#!/bin/sh\npython -m black --check $(TOP_LEVEL_PYTHON_FILES) $(SRC_DIR) && venv/bin/python -m isort --profile black --check .\n" > .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit
