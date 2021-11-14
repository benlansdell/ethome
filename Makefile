test:
		pytest

coverage:
		coverage run -m pytest
		coverage report -i
		coverage html -i

demo:
		CUDA_VISIBLE_DEVICES= python examples/sample_workflow.p
