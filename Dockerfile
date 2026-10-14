# ==============================================================================
# Customer Churn Prediction API — Dockerfile
# Multi-stage production container with healthcheck and non-root user
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing bytecode and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Install system dependencies for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set container working directory
WORKDIR /app

# Copy dependency definition and install packages
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy source code and serialized model artifacts
COPY src/ /app/src/
COPY models/ /app/models/
COPY README.md /app/README.md

# Create dedicated non-root application user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose API service port
EXPOSE 8000

# Docker healthcheck querying GET /health
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI service with Uvicorn
CMD ["uvicorn", "src.churn_prediction.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
