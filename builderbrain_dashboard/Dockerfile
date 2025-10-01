# Dockerfile for BuilderBrain Dashboard
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install uv

# Create non-root user
RUN useradd -m -u 1000 builderbrain

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Set environment variables for runtime
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application code
COPY . ./

# Clean up any existing virtual environment with wrong permissions
RUN rm -rf .venv || true

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start the dashboard as root (needed for uv to create virtual environment)
CMD ["uv", "run", "streamlit", "run", "builderbrain_dashboard/dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
