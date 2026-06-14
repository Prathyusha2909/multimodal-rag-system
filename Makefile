.PHONY: install test api web docker assets samples evaluate

install:
	python -m pip install -r backend/requirements.txt
	cd frontend && npm install

test:
	cd backend && python -m unittest discover -s tests -v
	cd frontend && npm run build

api:
	cd backend && uvicorn app.main:app --reload

web:
	cd frontend && npm run dev

docker:
	docker compose up --build

assets:
	python scripts/generate_assets.py

samples:
	python scripts/generate_sample_data.py

evaluate:
	python evaluation/run_deepeval.py
