.PHONY: docs

install:
	pip install -e ".[dev]"

clean:
	rm -rf .pytest_cache
	rm -rf build
	rm -rf dist
	rm -rf docs
	rm -rf scikit_lego.egg-info
	rm -rf .ipynb_checkpoints
	rm -rf .coverage*

pypi: clean
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*