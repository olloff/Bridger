# Build wheels separately so the compiler toolchain never reaches runtime.
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# Minimal non-root runtime image.
FROM python:3.11-slim AS runtime

# Set working directory
WORKDIR /app

# Install system dependencies
#  - tzdata: lets the bridge resolve names like "Europe/Berlin" so the
#    `timezone` setting in settings.yaml renders digest timestamps in
#    your local time instead of the container default (UTC).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    ffmpeg \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install only pre-built wheels from the builder stage.
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Copy application code
COPY main.py .
COPY bridge/ bridge/
COPY example.settings.yaml .

# Optional build-time stamp — pass via --build-arg GIT_SHA=$(git rev-parse --short HEAD)
ARG GIT_SHA=""
ENV BRIDGER_GIT_SHA=$GIT_SHA
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create directories for persistent data
RUN mkdir -p /app/messages/telegram /app/messages/discord /app/messages/whatsapp /app/logs

# Create non-root user for security
RUN useradd --create-home --shell /usr/sbin/nologin app && chown -R app:app /app /home/app
USER app

# Run the application
CMD ["python", "main.py"]
