FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# libgomp1: OpenMP runtime that slim images omit; limen's lightgbm dlopens
# libgomp.so.1 at import, so the app dies at boot without it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .

ARG GIT_SHA=""
ENV GIT_SHA=$GIT_SHA
ARG BUILD_DATE=""
ENV BUILD_DATE=$BUILD_DATE
ARG GIT_SUBJECT=""
ENV GIT_SUBJECT=$GIT_SUBJECT

EXPOSE 8765

CMD ["python", "-m", "mill"]
