# Hugging Face Hub Deployment Guide

## Fixed Issues

The permission error you encountered has been resolved by making the following changes:

### 1. Docker Container Permissions
- **Problem**: The container was running as root and couldn't write to model cache directories
- **Solution**: Created a non-root user (`appuser`) with proper permissions for all directories

### 2. Model Cache Directory
- **Problem**: Permission denied when downloading model files
- **Solution**: 
  - Created dedicated `/app/model_cache` directory with proper ownership
  - Added permission checks in the model loading function
  - Improved error handling for download failures

### 3. Environment Configuration
- **Problem**: Missing environment variables in deployment
- **Solution**: Added all required environment variables directly in the Dockerfile

## Key Changes Made

### Dockerfile Updates
```dockerfile
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
```

### Model Loading Improvements
- Added permission checks before creating cache directories
- Improved error handling for Hugging Face Hub downloads
- Better error messages for debugging

### Temp Directory Handling
- Updated to use configurable temp directory instead of hardcoded `/tmp`
- Added directory creation with proper error handling

## Deployment Steps

1. **Build and push your Docker image**:
   ```bash
   docker build -t your-username/urdu-pashto-classifier .
   docker push your-username/urdu-pashto-classifier
   ```

2. **Deploy to Hugging Face Spaces**:
   - Go to your Hugging Face Space
   - Update the Docker image reference
   - The environment variables are now set in the Dockerfile

3. **Verify deployment**:
   - Check the logs for successful model loading
   - Test the `/predict` endpoint with a sample audio file

## Environment Variables

All required environment variables are now set in the Dockerfile:
- `MODEL_CACHE`: `/app/model_cache` (writable directory)
- `TEMP_DIR`: `/app/temp` (writable directory)
- `MODEL_NAME`: `facebook/wav2vec2-base`
- `MODEL_REPO`: `Esahe/Urdu_Pashto_Model`
- `SCALER_FILE`: `scaler.pkl`
- `CLASSIFIER_FILE`: `embedding_classifier.pth`
- `DEVICE`: `cpu`

## Troubleshooting

If you still encounter issues:

1. **Check container logs** for specific error messages
2. **Verify model repository access** - ensure the model files are publicly accessible
3. **Check disk space** - ensure sufficient space for model downloads
4. **Network connectivity** - ensure the container can access Hugging Face Hub

The improved error handling will now provide more specific error messages to help diagnose any remaining issues.
