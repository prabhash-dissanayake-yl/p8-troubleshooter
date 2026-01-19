FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build deps and git
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git

COPY . /app

# Upgrade pip, setuptools, and wheel, then install uv
RUN pip install --upgrade pip setuptools wheel && pip install uv

ARG GITHUB_TOKEN
ENV GITHUB_TOKEN=${GITHUB_TOKEN}

# Run the build script to set up the application
RUN bash /app/build.sh

VOLUME ["/app/lightrag_storage"]

EXPOSE 8084

CMD ["/app/.venv/bin/python", "demo.py"]
