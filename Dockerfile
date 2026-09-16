FROM python:3.11-slim

WORKDIR /app

# OpenCV headless dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code and templates
LABEL build_bust="v4"
COPY mockup.py quads.json ./
COPY server.py ./
COPY templates/ ./templates/

EXPOSE 8080
ENV PORT=8080

CMD uvicorn server:app --host 0.0.0.0 --port ${PORT}
