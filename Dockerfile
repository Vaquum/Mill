FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
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
