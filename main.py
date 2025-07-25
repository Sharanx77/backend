from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from deepface import DeepFace
import tempfile
import os

app = FastAPI()

# Placeholder Deepfake detection logic:
# Normally you would load a model or API that detects deepfakes,
# Here, we mock with random for demonstration.

import random

def detect_deepfake(image):
    # Placeholder: returns a fake deepfake score between 0 (real) and 1 (fake)
    return random.uniform(0, 1)

def analyze_emotions(image):
    try:
        analysis = DeepFace.analyze(image, actions=["emotion"], enforce_detection=False)
        emotion = analysis["dominant_emotion"]
        confidence = max(analysis["emotion"].values())
        return emotion, confidence
    except Exception:
        return None, 0.0

def extract_frames_from_video(video_path, max_frames=30):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        count += 1
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames

def check_emotion_consistency(emotions, threshold=0.5):
    # Check if dominant emotions vary too much or too frequently
    # Let's measure the ratio of mode emotion count / total frames
    if not emotions:
        return True
    dominant_emotions = [e['dominant_emotion'] for e in emotions]
    mode_emotion = max(set(dominant_emotions), key=dominant_emotions.count)
    consistency_ratio = dominant_emotions.count(mode_emotion) / len(dominant_emotions)
    return consistency_ratio > threshold

@app.post("/analyze")
async def analyze_file(media: UploadFile = File(...)):
    suffix = os.path.splitext(media.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await media.read()
        tmp.write(contents)
        tmp_path = tmp.name

    response = {}
    try:
        if suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
            # Image processing
            img = cv2.imread(tmp_path)
            if img is None:
                return JSONResponse(status_code=400, content={"error": "Invalid image file"})
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            deepfake_score = detect_deepfake(img_rgb)
            dominant_emotion, confidence = analyze_emotions(img_rgb)

            response = {
                "deepfake_score": deepfake_score,
                "emotion_consistency": True,
                "emotions": [{
                    "frame": 0,
                    "dominant_emotion": dominant_emotion,
                    "confidence": confidence
                }]
            }
        elif suffix in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
            # Video processing
            frames = extract_frames_from_video(tmp_path, max_frames=30)
            deepfake_scores = []
            emotion_results = []

            for i, frame in enumerate(frames):
                df_score = detect_deepfake(frame)
                deepfake_scores.append(df_score)

                emo, conf = analyze_emotions(frame)
                if emo is None:
                    emo = "unknown"
                emotion_results.append({
                    "frame": i,
                    "dominant_emotion": emo,
                    "confidence": conf
                })

            avg_deepfake_score = sum(deepfake_scores) / len(deepfake_scores) if deepfake_scores else 0
            emotion_consistency = check_emotion_consistency(emotion_results, threshold=0.5)

            response = {
                "deepfake_score": avg_deepfake_score,
                "emotion_consistency": emotion_consistency,
                "emotions": emotion_results
            }
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file format"})

    finally:
        os.unlink(tmp_path)

    return response
