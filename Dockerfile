FROM python:3.11-slim

# System dependencies for video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser
RUN playwright install chromium && playwright install-deps chromium

# Copy application code
COPY . .

# Create runtime directories
RUN mkdir -p clips/raw clips/square clips/captioned clips/final \
    logs output temp

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/status')" || exit 1

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8000"]
