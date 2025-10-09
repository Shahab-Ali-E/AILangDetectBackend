# Memory-optimized model loading for Vercel deployment
import torch
import joblib
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from huggingface_hub import hf_hub_download
from .config import settings
import os
import gc

class EmbeddingClassifier(torch.nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x)


def load_model_vercel():
    """Memory-optimized model loading for Vercel deployment"""
    device = torch.device("cpu")  # Force CPU for Vercel
    
    # Set cache directories to /tmp for Vercel
    cache_dir = "/tmp/model_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set environment variables for Hugging Face cache
    os.environ["HF_HOME"] = "/tmp/huggingface_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
    os.environ["TORCH_HOME"] = "/tmp/torch_cache"
    
    # Create cache directories
    for cache_path in ["/tmp/huggingface_cache", "/tmp/transformers_cache", "/tmp/torch_cache"]:
        os.makedirs(cache_path, exist_ok=True)
    
    print(f"Using cache directory: {cache_dir}")
    
    # Download model files with minimal memory usage
    classifier_path = os.path.join(cache_dir, settings.CLASSIFIER_FILE)
    scaler_path = os.path.join(cache_dir, settings.SCALER_FILE)
    
    if not os.path.exists(classifier_path) or not os.path.exists(scaler_path):
        print("Downloading model files...")
        try:
            # Download with minimal memory footprint
            classifier_path = hf_hub_download(
                repo_id=settings.MODEL_REPO,
                filename=settings.CLASSIFIER_FILE,
                cache_dir=cache_dir,
                local_files_only=False
            )
            scaler_path = hf_hub_download(
                repo_id=settings.MODEL_REPO,
                filename=settings.SCALER_FILE,
                cache_dir=cache_dir,
                local_files_only=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download model files: {e}")
    
    # Load model with memory optimization
    print("Loading classifier model...")
    model = EmbeddingClassifier(input_dim=768)
    model.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
    model.eval()
    model = model.to(device)
    
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("Loading scaler...")
    scaler = joblib.load(scaler_path)
    
    # Load Wav2Vec2 with memory optimization
    print("Loading Wav2Vec2 model...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        settings.MODEL_NAME,
        cache_dir="/tmp/transformers_cache",
        local_files_only=False
    )
    
    # Load model with reduced precision to save memory
    wav2vec = Wav2Vec2Model.from_pretrained(
        settings.MODEL_NAME,
        cache_dir="/tmp/transformers_cache",
        local_files_only=False,
        torch_dtype=torch.float32  # Use float32 instead of float64
    ).to(device).eval()
    
    # Clear memory again
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("Model loading complete!")
    return model, scaler, feature_extractor, wav2vec, device
