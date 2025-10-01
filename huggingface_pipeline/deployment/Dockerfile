# Multi-stage Dockerfile for BuilderBrain inference API
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 builderbrain

# Install uv package manager
RUN pip install uv

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN uv sync --frozen

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 builderbrain

# Copy virtual environment from base stage
COPY --from=base --chown=builderbrain:builderbrain /app/.venv /app/.venv

# Set work directory
WORKDIR /app

# Copy application code
COPY --chown=builderbrain:builderbrain huggingface_pipeline/ ./huggingface_pipeline/
COPY --chown=builderbrain:builderbrain bb_core/ ./bb_core/
COPY --chown=builderbrain:builderbrain bb_nn/ ./bb_nn/
COPY --chown=builderbrain:builderbrain bb_losses/ ./bb_losses/
COPY --chown=builderbrain:builderbrain bb_priors/ ./bb_priors/
COPY --chown=builderbrain:builderbrain bb_runtime/ ./bb_runtime/
COPY --chown=builderbrain:builderbrain bb_train/ ./bb_train/
COPY --chown=builderbrain:builderbrain configs/ ./configs/

# Switch to non-root user
USER builderbrain

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["python", "-m", "huggingface_pipeline.inference_api.app"]
