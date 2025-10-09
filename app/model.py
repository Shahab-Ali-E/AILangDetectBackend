import torch
import joblib
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from huggingface_hub import hf_hub_download
from .config import settings
import os

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


def load_model():
    device = torch.device(settings.DEVICE if torch.cuda.is_available() else "cpu")
    
    # Create cache directory with proper permissions
    try:
        os.makedirs(settings.MODEL_CACHE, exist_ok=True)
        # Ensure the directory is writable
        if not os.access(settings.MODEL_CACHE, os.W_OK):
            raise PermissionError(f"Cannot write to model cache directory: {settings.MODEL_CACHE}")
    except Exception as e:
        raise RuntimeError(f"Failed to create or access model cache directory: {e}")

    # defensive checks (config.py already validates, but double-check)
    if not settings.MODEL_REPO or not settings.CLASSIFIER_FILE or not settings.SCALER_FILE:
        raise RuntimeError(
            "MODEL_REPO, CLASSIFIER_FILE and SCALER_FILE must be set. "
            "Check your .env or Docker environment variables."
        )

    # Use local model files if they exist, otherwise download from Hugging Face Hub
    classifier_path = os.path.join(settings.MODEL_CACHE, "models--Esahe--Urdu_Pashto_Model", "snapshots", "94aa4d83598350edc1c5f2fd853346795e454ba5", settings.CLASSIFIER_FILE)
    scaler_path = os.path.join(settings.MODEL_CACHE, "models--Esahe--Urdu_Pashto_Model", "snapshots", "94aa4d83598350edc1c5f2fd853346795e454ba5", settings.SCALER_FILE)
    
    # Check if local files exist
    if not os.path.exists(classifier_path) or not os.path.exists(scaler_path):
        print("Local model files not found, downloading from Hugging Face Hub...")
        try:
            classifier_path = hf_hub_download(
                repo_id=settings.MODEL_REPO,
                filename=settings.CLASSIFIER_FILE,
                cache_dir=settings.MODEL_CACHE
            )
            scaler_path = hf_hub_download(
                repo_id=settings.MODEL_REPO,
                filename=settings.SCALER_FILE,
                cache_dir=settings.MODEL_CACHE
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model files from Hugging Face Hub: {e}. "
                f"Check your internet connection and model repository access."
            )
    else:
        print(f"Using local model files: {classifier_path}, {scaler_path}")

    # load model/scaler/backbone as before...
    model = EmbeddingClassifier(input_dim=768).to(device)
    model.load_state_dict(torch.load(classifier_path, map_location=device))
    model.eval()
    
    # Ensure model is in eval mode and disable dropout
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()

    scaler = joblib.load(scaler_path)

    # For Wav2Vec2, we need to download from Hugging Face Hub as it's not included in local files
    print("Loading Wav2Vec2 model from Hugging Face Hub...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(settings.MODEL_NAME)
    wav2vec = Wav2Vec2Model.from_pretrained(settings.MODEL_NAME).to(device).eval()

    # Debug information
    print(f"Debug - Model loaded on device: {device}")
    print(f"Debug - Model classifier path: {classifier_path}")
    print(f"Debug - Scaler path: {scaler_path}")
    print(f"Debug - Model cache dir: {settings.MODEL_CACHE}")
    
    # Test model with dummy input to see if it's working correctly
    dummy_input = torch.randn(1, 768).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)
        dummy_prob = torch.sigmoid(dummy_output).cpu().numpy()[0][0]
    print(f"Debug - Dummy test - Raw output: {dummy_output.cpu().numpy()[0][0]:.4f}, Sigmoid: {dummy_prob:.4f}")

    return model, scaler, feature_extractor, wav2vec, device