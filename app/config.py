# app/config.py
from dotenv import load_dotenv, find_dotenv
import os

# ensure .env is found if present
load_dotenv(find_dotenv())

class Settings:
    MODEL_NAME = os.getenv("MODEL_NAME")
    MODEL_REPO = os.getenv("MODEL_REPO")
    SCALER_FILE = os.getenv("SCALER_FILE")
    CLASSIFIER_FILE = os.getenv("CLASSIFIER_FILE")
    MODEL_CACHE = os.getenv("MODEL_CACHE", "./model_files")
    DEVICE = os.getenv("DEVICE", "cpu")

settings = Settings()

# Validate required settings early and raise clear error if missing
_required = {
    "MODEL_NAME": settings.MODEL_NAME,
    "MODEL_REPO": settings.MODEL_REPO,
    "SCALER_FILE": settings.SCALER_FILE,
    "CLASSIFIER_FILE": settings.CLASSIFIER_FILE,
}
_missing = [k for k, v in _required.items() if not v]
if _missing:
    raise RuntimeError(
        f"Missing environment variables: {', '.join(_missing)}. "
        "Set them in your .env or in the environment. Example: MODEL_REPO=Esahe/Urdu_Pashto_Model"
    )
