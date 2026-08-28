FROM python:3.10-slim

WORKDIR /app

# Install system runtime dependencies for OpenCV and GUI-less image manipulation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and configuration files
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Install aeroeval package in editable mode
RUN pip install --no-cache-dir -e . --no-deps

EXPOSE 8000 8501

# Default command starts FastAPI API
CMD ["uvicorn", "aeroeval.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
