FROM python:3.11-slim

# System deps for scipy / numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (so layer caches across code edits)
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source
COPY bioperator_env /app/bioperator_env
COPY server /app/server
COPY openenv.yaml /app/openenv.yaml

ENV PYTHONPATH=/app
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
