# Indication Extraction Agent

A Python project for indication extraction.

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

## Usage

```bash
# Run with Poetry
poetry run python src/main.py

# Or activate the shell first
poetry shell
python src/main.py
```

## Adding Dependencies

To add a new package (e.g., langchain-openai):
```bash
poetry add langchain-openai
```

To add a development dependency:
```bash
poetry add --group dev pytest
```

To update dependencies:
```bash
poetry update
```

## License

MIT

