FROM python:3.10-slim

COPY hello.py .

RUN hello.py .

CMD ["python3", "-c", "print('Hello, World!')"]
