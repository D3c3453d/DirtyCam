lint:
	poetry run pre-commit run --all-files

train:
	poetry run python src/train.py --all-frames frames/all --focus-frames frames/focus --model-dir ./models

predict:
	poetry run python src/predict.py --model-dir ./models maxim-bogdanov-wjAR4jo979Y-unsplash.jpg
