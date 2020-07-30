
.PHONY: init
init:
	python -m pip install -U pip
	python -m pip install -r requirements.txt -c constraints.txt


lint:
	black --check giant_exercise tests setup.py
	flake8 giant_exercise tests setup.py
	mypy giant_exercise tests setup.py

format:
	isort -rc giant_exercise tests setup.py
	black giant_exercise tests setup.py



.PHONY: test_unit
test_unit:
	pytest -vv --cov giant_exercise --cov-config=setup.cfg --cov-report xml:.coverage-unit.xml tests/unit
