#Run unit tests
test:
		CUDA_VISIBLE_DEVICES= pytest

#Compute testing coverage
coverage:
		CUDA_VISIBLE_DEVICES= coverage run -m pytest
		coverage report -i
		coverage html -i
		coverage xml -i

#Update code coverage report to codecov
coverage_upload:
		bin/codecov -f coverage.xml -t 29196d6a-3dc7-4efd-8aa5-3795c7eafaec

#Compute coverage and upload
coverageall: coverage coverage_upload

#Build python package
build:
		python -m build

#Upload built package to testpypi repository
deploytest:
		python -m twine upload --repository testpypi --skip-existing dist/* 
#Then you can test install with:
#python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps ethome
#or with version number
#python3 -m pip install --index-url https://test.pypi.org/simple/ ethome-ml==0.3.0

#Upload built package to pypi repository (publish)
deploy:
		python -m twine upload dist/*
#Then can install simply with:
#pip install ethome 

#Pointer to demo script for testing/experimenting with functionality
demo:
		CUDA_VISIBLE_DEVICES= python examples/sample_workflow.py

#See:
# https://www.mkdocs.org/getting-started/
# https://github.com/squidfunk/mkdocs-material
# https://github.com/lukasgeiter/mkdocs-awesome-pages-plugin
# https://pythonrepo.com/repo/ml-tooling-lazydocs-python-documentation#mkdocs-integration

#Generate html docs
doc:
		cp README.md docs/index.md
		lazydocs \
		    --output-path="./docs/api-docs" \
		    --overview-file="README.md" \
    		--src-base-url="https://github.com/benlansdell/ethome/blob/master/" \
			--ignored-modules="config" \
			--ignored-modules="version" \
			--ignored-modules="models" \
			--ignored-modules="plot" \
			--ignored-modules="unsupervised" \
			--ignored-modules="features" \
			--ignored-modules="features.cnn1d" \
			--ignored-modules="features.features" \
			--ignored-modules="features.generic_features" \
			--ignored-modules="features.dl_features" \
			--ignored-modules="features.mars_features" \
    		ethome
		mkdocs build

#Post html docs to gh-pages branch (publish)
doc-deploy:
		mkdocs gh-deploy
