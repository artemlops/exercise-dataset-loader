.PHONY: init
init:
	python -m pip install -U pip
	python -m pip install -r requirements.txt -c constraints.txt


lint:
	black --check dataset_loader tests setup.py
	flake8 dataset_loader tests setup.py
	mypy dataset_loader tests setup.py


format:
	isort -rc dataset_loader tests setup.py
	black dataset_loader tests setup.py


.PHONY: test_unit
test_unit:
	pytest -vv --cov=dataset_loader --cov-config=setup.cfg --cov-report \
		xml:.coverage-unit.xml tests/unit


.PHONY: upload_coverage
upload_coverage:
	bash -c 'bash <(curl -s https://codecov.io/bash)'
