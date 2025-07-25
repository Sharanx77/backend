import cv2
import numpy as np
import librosa
import os

def extract_frames(video_path, model, labels):
    cap = cv2.VideoCapture(video_path)
    frame_emotions = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or count > 30:
            break
        gray = cv2.cvtColor(cv2.resize(frame, (48, 48)), cv2.COLOR_BGR2GRAY)
        face = np.expand_dims(np.expand_dims(gray, axis=-1), axis=0) / 255.0
        pred = model.predict(face)[0]
        label = labels[np.argmax(pred)]
        frame_emotions.append(label)
        count += 5
        cap.set(1, count)
    cap.release()
    return frame_emotions

def extract_audio_emotions(video_path):
    y, sr = librosa.load(video_path)
    features = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)
    label = 'Neutral'  # Mock — replace with real model
    return [label] * 3
