# Use Python 3.11 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the application code
COPY . .

# Create temp directory
RUN mkdir -p /tmp

# Define build arguments for model configuration
ARG MODEL_NAME=facebook/wav2vec2-base
ARG MODEL_REPO=Esahe/Urdu_Pashto_Model
ARG SCALER_FILE=scaler.pkl
ARG CLASSIFIER_FILE=embedding_classifier.pth
ARG DEVICE=cpu

# Set environment variables
ENV PYTHONPATH=/app
ENV TEMP_DIR=/tmp
ENV MODEL_NAME=${MODEL_NAME}
ENV MODEL_REPO=${MODEL_REPO}
ENV SCALER_FILE=${SCALER_FILE}
ENV CLASSIFIER_FILE=${CLASSIFIER_FILE}
ENV DEVICE=${DEVICE}

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
