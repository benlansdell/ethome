test:
		CUDA_VISIBLE_DEVICES= pytest

coverage:
		CUDA_VISIBLE_DEVICES= coverage run -m pytest
		coverage report -i
		coverage html -i

build:
		python -m build

deploytest:
		python -m twine upload --repository testpypi --skip-existing dist/* 

#Then test install w:
#python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps ethome
#or with version number
#python3 -m pip install --index-url https://test.pypi.org/simple/ ethome==0.2.1

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
		lazydocs \
		    --output-path="./docs/api-docs" \
		    --overview-file="README.md" \
    		--src-base-url="https://github.com/benlansdell/ethome/blob/master/" \
    		ethome
		mkdocs build
