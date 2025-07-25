from keras.models import load_model
import numpy as np
import cv2
from PIL import Image

model = load_model('model/deepfake_model.h5')

def detect_deepfake_image(file):
    img = Image.open(file).resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    return {'label': 'fake' if prediction > 0.5 else 'real', 'confidence': float(prediction)}
