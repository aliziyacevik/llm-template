.PHONY: start test

start:
	uvicorn main:app --reload

test:
	curl -X POST http://127.0.0.1:8000/chat/ \
		-H "Content-Type: application/json" \
		-d '{"message": "Hello there"}'

