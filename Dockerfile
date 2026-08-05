# ── Build stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install only server dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy only what the server needs at runtime
COPY api/main.py                           api/main.py
COPY frontend/                             frontend/
COPY linelogic_output/20260729_220240/linelogic_cnn.keras  linelogic_output/20260729_220240/linelogic_cnn.keras
COPY linelogic_output/20260729_220240/label_map.json       linelogic_output/20260729_220240/label_map.json

EXPOSE 8000

# ALLOWED_ORIGINS can be overridden at runtime:
#   docker run -e ALLOWED_ORIGINS="https://yourdomain.com" ...
ENV ALLOWED_ORIGINS="*"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
