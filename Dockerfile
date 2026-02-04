FROM python:3.10-slim

# --------------------
# Environment
# --------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# --------------------
# System dependencies (MINIMAL)
# --------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --------------------
# App setup
# --------------------
WORKDIR /app

# Install Python deps first (cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (.dockerignore excludes .env, venv, __pycache__, etc.)
COPY . .

# Runtime directories
RUN mkdir -p uploads college_faces debug_faces

EXPOSE 8000

# --------------------
# Healthcheck
# --------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

# --------------------
# Start app (app.py at root → module app, variable app)
# Set SKIP_MIGRATIONS=true to start without running migrations (e.g. when DB is unreachable locally).
# --------------------
CMD ["sh", "-c", "if [ \"$SKIP_MIGRATIONS\" != \"true\" ]; then alembic upgrade head; fi && uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1"]
