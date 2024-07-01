pretty:
	isort --profile black src/AIAH_utility
	black --line-length 120 src/AIAH_utility

test:
	isort --check --profile black src/AIAH_utility
	black --line-length 120 --check src/AIAH_utility
	flake8 src/AIAH_utility