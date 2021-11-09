test:
		pytest

coverage:
		coverage run -m pytest
		coverage report