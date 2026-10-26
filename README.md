# House Price Predictor

A small machine learning project for predicting California housing values.

## Features

- Dataset loading and validation
- Exploratory data analysis
- Missing-value handling
- Train/test split
- Linear regression
- Model evaluation
- Automated tests
- Pre-commit code quality checks

## Requirements

- Python 3.11–3.13
- Poetry 2.x

## Installation

```bash
poetry install
```

## Run the project

```bash
poetry run python run.py
```

The first run may download the California Housing dataset.

## Run tests

```bash
poetry run pytest
```

## Run code quality checks

```bash
poetry run black --check src tests run.py
poetry run flake8 src tests run.py
poetry run mypy src
poetry run pytest
```

## Install Git hooks

```bash
poetry run pre-commit install
```

## Outputs

Generated plots and the trained model are saved to `reports/`.
