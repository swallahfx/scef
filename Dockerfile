# Create a Dockerfile in your project root
echo "FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD uvicorn web.app:app --host 0.0.0.0 --port \$PORT" > Dockerfilecf