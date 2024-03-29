.DEFAULT_GOAL := help

TEST_COMMAND = pytest
RUN_COMMAND = sh
help:
	@echo 'Use "make test" to run all the unit tests and docstring tests.'
	@echo 'Use "make pep8" to validate PEP8 compliance.'
	@echo 'Use "make html" to create html documentation with sphinx'
	@echo 'Use "make all" to run all the targets listed above.'
test:
	$(TEST_COMMAND) q2_matchmaker
pep8:
	pycodestyle q2_matchmaker setup.py --ignore=E731,E722,W503
	flake8 q2_matchmaker setup.py scripts  --ignore=E731,E722,W503

all: pep8 test
