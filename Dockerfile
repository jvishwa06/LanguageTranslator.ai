FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY architecture.py mainonnx.py ./
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "mainonnx:app", "--host", "0.0.0.0", "--port", "8000"]