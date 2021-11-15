test:
		pytest

coverage:
		coverage run -m pytest
		coverage report -i
		coverage html -i

demo:
		CUDA_VISIBLE_DEVICES= python examples/sample_workflow.py

doc:
		lazydocs \
		    --output-path="./docs/api-docs" \
		    --overview-file="README.md" \
    		--src-base-url="https://github.com/benlansdell/behaveml/tree/master/" \
    		behaveml	
