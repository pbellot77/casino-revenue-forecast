# Dockerfile

# Use slim Python image to keep container size small
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency manifest first (Docker layer caching)
# If requirements.txt hasn't changed, this layer is cached
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# Create data directory
RUN mkdir -p data

# Expose port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
