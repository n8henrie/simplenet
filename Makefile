SHELL := /bin/bash
PYTHON = /usr/bin/env python3
PWD = $(shell pwd)
GREP := $(shell command -v ggrep || command -v grep)
SED := $(shell command -v gsed || command -v sed)

.PHONY: clean-pyc clean-build docs clean release clean-docs help github

help:
	@$(GREP) --only-matching --word-regexp '^[^[:space:].]*:' Makefile | $(SED) 's|:[:space:]*||'

clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info
	rm -fr src/*.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/

lint:
	flake8 src/simplenet tests

test: .venv
	.venv/bin/python -m pip install -e .[test]
	py.test tests

test-all:
	tox

coverage:
	coverage run --source simplenet setup.py test
	coverage report -m
	coverage html
	open htmlcov/index.html

clean-docs:
	rm -f docs/simplenet*.rst
	rm -f docs/modules.rst

docs: clean-docs .venv
	.venv/bin/python -m pip install -e .[dev]
	source .venv/bin/activate && sphinx-apidoc -o docs/ src/simplenet
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	-open docs/_build/html/index.html

release: dist
	twine upload dist/*

dist: clean docs
	$(PYTHON) setup.py --long-description | rst2html.py --halt=2
	$(PYTHON) setup.py sdist
	$(PYTHON) setup.py bdist_wheel
	ls -l dist

.venv:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e .

update-reqs: requirements.txt
	@$(GREP) --invert-match --no-filename '^#' requirements*.txt | \
		$(SED) 's|==.*$$||g' | \
		xargs venv/bin/python -m pip install --upgrade; \
	for reqfile in requirements*.txt; do \
		echo "Updating $${reqfile}..."; \
		venv/bin/python -c 'print("\n{:#^80}".format("  Updated reqs below  "))' >> "$${reqfile}"; \
		for lib in $$(venv/bin/pip freeze --all --isolated --quiet | $(GREP) '=='); do \
			if $(GREP) "^$${lib%%=*}==" "$${reqfile}" >/dev/null; then \
				echo "$${lib}" >> "$${reqfile}"; \
			fi; \
		done; \
	done;

github:
	hub create
