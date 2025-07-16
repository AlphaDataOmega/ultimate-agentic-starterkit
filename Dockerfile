FROM python:3.11-slim

# Set build arguments
ARG PYTHON_VERSION=3.11

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Basic build tools
    build-essential \
    gcc \
    g++ \
    make \
    # Audio libraries for pyttsx3
    espeak \
    espeak-data \
    libespeak-dev \
    pulseaudio \
    alsa-utils \
    # System utilities
    curl \
    wget \
    git \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r starterkit && useradd -r -g starterkit starterkit

# Create necessary directories
RUN mkdir -p /app/logs /app/workspace /app/temp /app/output \
    && chown -R starterkit:starterkit /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Fix permissions
RUN chown -R starterkit:starterkit /app

# Switch to non-root user
USER starterkit

# Create directories that might be needed
RUN mkdir -p logs workspace temp output

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "from StarterKit.core.config import get_config; print('healthy')" || exit 1

# Default command
CMD ["python", "-c", "from StarterKit.core.config import get_config; from StarterKit.core.logger import setup_global_logging; setup_global_logging(); print('StarterKit ready')"]