from keras.models import load_model
import numpy as np
import cv2
from utils.video_utils import extract_frames, extract_audio_emotions

emotion_model = load_model('model/emotion_model.h5')
emotion_labels = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral']

def check_emotional_consistency(video_path):
    frame_emotions = extract_frames(video_path, emotion_model, emotion_labels)
    audio_emotions = extract_audio_emotions(video_path)

    consistent = frame_emotions[:3] == audio_emotions[:3]
    return {
        'video_emotion': frame_emotions,
        'audio_emotion': audio_emotions,
        'consistent': consistent
    }
