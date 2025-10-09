#!/bin/bash
# Build script for Vercel deployment

echo "Setting up optimized build environment..."

# Set environment variables for build optimization
export TORCH_HOME=/tmp/torch_cache
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_HOME=/tmp/huggingface_cache
export MODEL_CACHE=./model_files
export DEVICE=cpu

# Install dependencies with optimizations
echo "Installing dependencies..."
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r requirements-vercel.txt

echo "Build setup complete!"
