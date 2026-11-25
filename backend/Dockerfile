FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project.
COPY . .

# Hugging Face Spaces provides $PORT=7860.
EXPOSE 7860

# Shell-form CMD so $PORT is expanded at runtime (falls back to 7860 locally).
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
