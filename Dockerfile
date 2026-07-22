# syntax=docker/dockerfile:1.2
# Runtime image for the flight delay prediction API.
#
# Only the serving dependencies (requirements.txt) and the modules the API actually
# imports make it into the image: no notebook, no training pipeline, no MLflow, no
# dataset. The result is a small image that starts fast on Cloud Run.
FROM python:3.10-slim

# Pinned interpreter: numpy 1.22.4 / pandas 1.3.5 publish no wheels for 3.11+.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The serving artifact travels with the image, so a cold start needs no network.
COPY challenge/__init__.py challenge/api.py challenge/model.py challenge/model.joblib ./challenge/

# Least privilege: the service never writes to disk, so it runs as a non-root user.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','8080') + '/health', timeout=4).status == 200 else 1)"

# Cloud Run injects $PORT and expects the container to listen on it.
CMD ["sh", "-c", "exec uvicorn challenge:application --host 0.0.0.0 --port ${PORT:-8080}"]
