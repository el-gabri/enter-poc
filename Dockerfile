# Multi-stage build: one source of truth for both services.
#   docker build --target api -t litigation-api .
#   docker build --target frontend -t litigation-frontend .

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./

# ---------------------------------------------------------------- api
FROM base AS api
# Tesseract (por) enables OCR for scanned lawsuits inside the container
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-por \
    && rm -rf /var/lib/apt/lists/*
COPY app ./app
RUN pip install ".[ocr]"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------- frontend
FROM base AS frontend
COPY app ./app
RUN pip install ".[frontend]"
COPY frontend ./frontend
EXPOSE 8501
CMD ["streamlit", "run", "frontend/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
