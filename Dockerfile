# Production Dockerfile for Aerial Image Change Detection Studio
# Multi-stage lightweight Python container with GPU / CPU auto-selection

FROM python:3.11-slim

# Prevent python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU or CUDA via build argument)
ARG CUDA_VERSION=cu121
RUN if [ "$CUDA_VERSION" = "cpu" ]; then \
        pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/${CUDA_VERSION}; \
    fi

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY . .

# Expose web server port
EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Launch production server
CMD ["python", "app.py"]
