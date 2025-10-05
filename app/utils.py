import torch
import numpy as np
import torchaudio
import soundfile as sf
from moviepy import VideoFileClip
import os


def extract_audio_from_video(video_path, output_dir=".", sample_rate=16000):
    """
    Extracts audio from a given video and saves it as a .wav file.
    Compatible with MoviePy v2+.
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(
        output_dir,
        os.path.splitext(os.path.basename(video_path))[0] + ".wav"
    )

    try:
        clip = VideoFileClip(video_path)
        if clip.audio:
            clip.audio.write_audiofile(audio_path, fps=sample_rate, logger=None)
        else:
            raise ValueError(f"No audio track found in {video_path}")
        clip.close()
    except Exception as e:
        raise RuntimeError(f"Failed to extract audio from {video_path}: {e}")

    return audio_path


def extract_embedding(audio_path, feature_extractor, wav2vec, device, sample_rate=16000):
    print(f"Debug - Processing audio: {audio_path}")
    try:
        waveform, sr = torchaudio.load(audio_path)
        print(f"Debug - Loaded with torchaudio, sr: {sr}, shape: {waveform.shape}")
    except Exception as e:
        print(f"Debug - torchaudio failed: {e}, trying soundfile")
        # fallback for mp3 or unsupported formats
        waveform, sr = sf.read(audio_path)
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        print(f"Debug - Loaded with soundfile, sr: {sr}, shape: {waveform.shape}")

    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    if sr != sample_rate:
        print(f"Debug - Resampling from {sr} to {sample_rate}")
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

    waveform = waveform.numpy().astype(np.float32)
    print(f"Debug - Final waveform shape: {waveform.shape}, mean: {waveform.mean():.4f}, std: {waveform.std():.4f}")

    inputs = feature_extractor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True
    )
    input_values = inputs["input_values"].to(device)
    print(f"Debug - Feature extractor input shape: {input_values.shape}")

    with torch.no_grad():
        outputs = wav2vec(input_values)
        hidden = outputs.last_hidden_state
        print(f"Debug - Wav2Vec output shape: {hidden.shape}")
        emb = hidden.mean(dim=1).cpu().numpy().astype(np.float32)
        print(f"Debug - Final embedding shape: {emb.shape}")

    return emb


def predict_audio_or_video(file_path, model, scaler, feature_extractor, wav2vec, device, label_map={0: "ur", 1: "ps"}):
    """
    Handles both audio (.wav, .mp3) and video (.mp4, .mkv) files.
    Extracts audio from video before generating embeddings.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv"]:
        # Extract audio first
        audio_path = extract_audio_from_video(file_path)
    else:
        audio_path = file_path

    emb = extract_embedding(audio_path, feature_extractor, wav2vec, device)
    print(f"Debug - Embedding shape: {emb.shape}, mean: {emb.mean():.4f}, std: {emb.std():.4f}")
    
    emb_scaled = scaler.transform(emb)
    print(f"Debug - Scaled embedding mean: {emb_scaled.mean():.4f}, std: {emb_scaled.std():.4f}")

    x_tensor = torch.tensor(emb_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        logit = model(x_tensor)
        prob = torch.sigmoid(logit).cpu().numpy()[0][0]

    # Debug information (can be removed in production)
    print(f"Debug - Raw logit: {logit.cpu().numpy()[0][0]:.4f}, Sigmoid prob: {prob:.4f}")
    print(f"Debug - Model device: {next(model.parameters()).device}, Input device: {x_tensor.device}")
    
    # Clamp probability to prevent extreme values
    prob = np.clip(prob, 0.001, 0.999)
    print(f"Debug - Clamped prob: {prob:.4f}")

    pred_idx = int(prob >= 0.5)
    pred_label = label_map.get(pred_idx, str(pred_idx))

    return pred_label, float(prob), audio_path
