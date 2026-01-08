# MedLit Agent Docker Image
# Multi-stage build for smaller final image

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir build && \
    pip wheel --no-cache-dir --wheel-dir /wheels .

# Runtime stage
FROM python:3.11-slim as runtime

WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash medlit
USER medlit

# Copy wheels and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --user /wheels/*.whl && \
    rm -rf /wheels

# Copy application code
COPY --chown=medlit:medlit . .

# Set environment
ENV PATH="/home/medlit/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from medlit import __version__; print(__version__)" || exit 1

# Default command
ENTRYPOINT ["medlit"]
CMD ["--help"]
