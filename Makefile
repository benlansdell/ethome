test:
		CUDA_VISIBLE_DEVICES= pytest

coverage:
		CUDA_VISIBLE_DEVICES= coverage run -m pytest
		coverage report -i
		coverage html -i
		coverage xml -i

coverage_upload:
		bin/codecov -f coverage.xml -t 29196d6a-3dc7-4efd-8aa5-3795c7eafaec

coverageall: coverage coverage_upload

build:
		python -m build

deploytest:
		python -m twine upload --repository testpypi --skip-existing dist/* 

#Then test install w:
#python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps ethome
#or with version number
#python3 -m pip install --index-url https://test.pypi.org/simple/ ethome-ml==0.3.0

deploy:
		python -m twine upload dist/*

#Then install w:
#pip install ethome 

demo:
		CUDA_VISIBLE_DEVICES= python examples/sample_workflow.py

#See:
# https://www.mkdocs.org/getting-started/
# https://github.com/squidfunk/mkdocs-material
# https://github.com/lukasgeiter/mkdocs-awesome-pages-plugin
# https://pythonrepo.com/repo/ml-tooling-lazydocs-python-documentation#mkdocs-integration

doc:
		cp README.md docs/index.md
		lazydocs \
		    --output-path="./docs/api-docs" \
		    --overview-file="README.md" \
    		--src-base-url="https://github.com/benlansdell/ethome/blob/master/" \
			--ignored-modules="config" \
			--ignored-modules="version" \
			--ignored-modules="models" \
			--ignored-modules="features" \
			--ignored-modules="features.cnn1d" \
			--ignored-modules="features.features" \
			--ignored-modules="features.generic_features" \
			--ignored-modules="features.dl_features" \
			--ignored-modules="features.mars_features" \
    		ethome
		mkdocs build

doc-deploy:
		mkdocs gh-deploy
