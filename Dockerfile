# Base image
FROM python:3.12-slim

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy app files
COPY ./app /app/app

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Create writable directories with proper permissions
RUN mkdir -p /app/model_cache /app/temp && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV MODEL_CACHE="/app/model_cache"
ENV TEMP_DIR="/app/temp"
ENV MODEL_NAME="facebook/wav2vec2-base"
ENV MODEL_REPO="Esahe/Urdu_Pashto_Model"
ENV SCALER_FILE="scaler.pkl"
ENV CLASSIFIER_FILE="embedding_classifier.pth"
ENV DEVICE="cpu"

# Expose port
EXPOSE 7860

# Run API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
