test:
		pytest

coverage:
		coverage run -m pytest
		coverage report

demo:
		CUDA_VISIBLE_DEVICES= python examples/sample_workflow.p
