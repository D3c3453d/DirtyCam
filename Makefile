lint:
	poetry run pre-commit run --all-files

train:
	poetry run python src/train.py

predict:
	poetry run python src/predict.py
