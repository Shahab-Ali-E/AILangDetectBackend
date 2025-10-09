# app/main.py
from fastapi import FastAPI, UploadFile, File
import os
import tempfile
from .model import load_model
from .utils import predict_audio_or_video

app = FastAPI(title="Urdu Pashto Language Classifier")

# Load model once (this will now fail early with a helpful error if env is missing)
model, scaler, feature_extractor, wav2vec, device = load_model()

@app.get("/")
def home():
    return {"message": "Welcome to Urdu-Pashto Language Classifier API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    # create a temp file in the configured temp directory
    temp_dir = os.getenv("TEMP_DIR", "/app/temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir) as tmp:
        tmp.write(await file.read())
        upload_path = tmp.name

    try:
        pred_label, prob, audio_path = predict_audio_or_video(
            upload_path, model, scaler, feature_extractor, wav2vec, device
        )
        return {
            "prediction": pred_label,
            "probability": prob,
            "video_path": upload_path,
            "extracted_audio": audio_path
        }
    finally:
        # cleanup uploaded file
        try:
            os.remove(upload_path)
        except Exception:
            pass

# This is important for Vercel
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)