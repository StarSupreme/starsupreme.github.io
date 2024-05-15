import os
import re
import string
import random
import pandas as pd
import numpy as np
import sounddevice as sd
import librosa
from collections import Counter
from flask import Flask, render_template, request, session
from gtts import gTTS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fer import FER
import cv2
import requests
from kivy.core.audio import SoundLoader
import speech_recognition as sr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pyautogui
import pygetwindow as gw
from datetime import datetime
import time

app = Flask(__name__)
app.secret_key = 'MANBEARPIG_MUDMAN888'

tokenizer = AutoTokenizer.from_pretrained("./model/")
model = AutoModelForCausalLM.from_pretrained("./model/")

df = pd.read_csv('./train.csv')
df['Concept'] = df['Concept'].str.lower().str.strip()
df['Description'] = df['Description'].str.lower().str.strip()
df['context'] = df['Concept'] + " " + df['Description']
df = df[['context', 'Description']]
df.columns = ['context', 'response']

fallback_responses = [
    "I'm here to help. Could you give me more details?",
    "Can you elaborate on that?",
    "Let's dive deeper into this topic. What exactly would you like to know?",
    "I'm not sure I understand fully. Could you clarify?",
    "Interesting point! Could you tell me a bit more about what you're thinking?",
    "I'm here to assist! How can I help you further?",
    "That's an interesting topic! Do you have any specific questions about it?",
    "Could you provide more context? I'd love to help out!",
    "I'm eager to assist! What more can I do for you?",
    "Let's explore this topic together. What more can you tell me?"
]

casual_responses = {
    "how are you": "I'm doing well, thanks for asking!",
    "hello": "Hello there! It's great to chat with you.",
    "bye": "Goodbye! Have a wonderful day.",
    "hi": "Hi there! How can I assist you today?",
    "i am happy": "Do you know what I do when I'm in a good mood?",
    "what do you do when you are happy": "I like to read things",
    "so you are a lawyer": "No, I'm talking about the 'Laws of Motion'"
}

output_directory = "./screenshots/"
os.makedirs(output_directory, exist_ok=True)

target_window_title = "Camera"

def take_screenshots():
    window = gw.getWindowsWithTitle(target_window_title)
    if not window:
        print(f"Window titled '{target_window_title}' not found.")
        return

    window = window[0]
    try:
        window.activate()
    except Exception as e:
        print(f"Warning: Could not activate the window '{target_window_title}'. Error: {e}")

    for i in range(30):
        screenshot = pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        screenshot_path = os.path.join(output_directory, f"screenshot_{timestamp}_{i+1}.png")
        screenshot.save(screenshot_path)
        print(f"Saved screenshot {i + 1}/30 at {screenshot_path}")
        time.sleep(1)

def preprocess_text(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Error: {e}")
        return None

def retrieve_response(query, df, word_threshold=3):
    query_words = set(preprocess_text(query).split())
    best_match = None
    best_count = 0
    exact_match = None

    for _, row in df.iterrows():
        concept_words = set(preprocess_text(row['context']).split())
        context_words = set(preprocess_text(row['context']).split())

        concept_match = concept_words.issubset(query_words)
        if concept_match:
            exact_match = row['response']
            break

        common_words = query_words & context_words
        common_count = len(common_words)

        if common_count > best_count:
            best_count = common_count
            best_match = row['response']

    if exact_match:
        return exact_match
    elif best_count >= word_threshold:
        return best_match
    else:
        return generate_response_with_dialogpt(query)

def generate_response_with_dialogpt(query):
    inputs = tokenizer.encode(query + tokenizer.eos_token, return_tensors='pt')
    response_ids = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(response_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response

def perform_real_time_prediction():
    global final_emotion

    def extract_features(audio_data, sample_rate):
        mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs

    def generate_audio_data(emotion, duration, sample_rate):
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

        if emotion == 'happy':
            frequency = np.interp(np.random.random(), [0, 1], [220, 630])
            audio_data = np.sin(2 * np.pi * frequency * t + 5 * np.sin(2 * np.pi * 0.25 * t))
        elif emotion == 'sad':
            frequency = np.interp(np.random.random(), [0, 1], [100, 700])
            audio_data = np.sin(2 * np.pi * 220 * t) * np.interp(t, [0, duration], [1, 0])
        elif emotion == 'angry':
            frequency = np.interp(np.random.random(), [0, 1], [300, 700])
            audio_data = librosa.core.tone(frequency, sr=sample_rate, duration=duration) + 0.3 * np.sin(
                2 * np.pi * 0.5 * t)
        else:
            audio_data = np.interp(np.random.rand(int(duration * sample_rate)), [0, 1], [-1, 1])

        return audio_data

    sample_rate = 22050
    duration = 3

    emotions = ['happy', 'sad', 'angry', 'neutral']
    X = []
    y = []

    for emotion in emotions:
        for _ in range(50):
            audio_data = generate_audio_data(emotion, duration, sample_rate)
            features = extract_features(audio_data, sample_rate)
            X.append(features)
            y.append(emotion)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    def predict_emotion(audio_data, sample_rate):
        features = extract_features(audio_data, sample_rate)
        features = features.reshape(1, -1)
        predicted_emotion = rf_classifier.predict(features)
        return predicted_emotion[0]

    duration = 5
    sample_rate = 22050
    channels = 1

    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, blocking=True)
    dominant_emotion_voice = predict_emotion(audio_data[:, 0], sample_rate)
    emotions_voice = dominant_emotion_voice

    detector = FER(mtcnn=True)

    emotion_intensities = {
        'happy': 0,
        'sad': 0,
        'fear': 0,
        'disgust': 0,
        'angry': 0,
        'surprise': 0,
        'neutral': 0
    }

    emoji_urls = {
        'happy': 'https://ibb.co/YpDkdrY',
        'sad': 'https://ibb.co/Fh1DKLK',
        'fear': 'https://ibb.co/R052r20',
        'disgust': 'https://ibb.co/K21gxTB',
        'angry': 'https://ibb.co/wYmby3r',
        'surprise': 'https://ibb.co/W54VnqN',
        'neutral': 'https://ibb.co/wCZ8yq0',
    }

    def fetch_image_with_alpha(url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)
                if image is not None and image.shape[2] == 3:
                    b, g, r = cv2.split(image)
                    alpha = np.ones_like(b) * 255
                    image = cv2.merge((b, g, r, alpha))
                return image
            else:
                print(f"Failed to fetch image from {url}. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching image from {url}: {e}")
            return None

    emoji_images = {emotion: fetch_image_with_alpha(url) for emotion, url in emoji_urls.items()}

    def get_latest_screenshot():
        screenshots = [f for f in os.listdir('screenshots') if f.endswith('.png')]
        if screenshots:
            latest_file = max(screenshots, key=lambda x: os.path.getctime(os.path.join('screenshots', x)))
            return os.path.join('screenshots', latest_file)
        return None

    latest_screenshot = get_latest_screenshot()
    if not latest_screenshot:
        print("No screenshots available")
    else:
        frame = cv2.imread(latest_screenshot)
        if frame is None:
            print("Failed to read screenshot")
        else:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            result = detector.detect_emotions(small_frame)
            if result:
                for face in result:
                    emotions_face = face['emotions']
                    dominant_emotion_face = max(emotions_face, key=emotions_face.get)

                    def update_emotion_intensities_voice(emotion):
                        emotion_intensities[emotion] += 1

                    update_emotion_intensities_voice(emotions_voice)

                    weight_face = 0.6
                    weight_voice = 0.4

                    normalized_emotion_face = emotions_face[dominant_emotion_face] / sum(emotions_face.values())
                    normalized_emotion_voice = emotion_intensities.get(emotions_voice, 0) / sum(
                        emotion_intensities.values())

                    combined_emotion_score = (normalized_emotion_face * weight_face) + (
                                normalized_emotion_voice * weight_voice)

                    final_emotion = dominant_emotion_face if combined_emotion_score >= 0.5 else emotions_voice

                    emotion_voice = predict_emotion(audio_data[:, 0], sample_rate)

                    if dominant_emotion_face == emotions_voice:
                        final_emotion = dominant_emotion_face
                    elif dominant_emotion_face != emotions_voice and dominant_emotion_face in ["happy", "neutral", "surprise"] and emotions_voice in [ "happy", "neutral"]:
                        final_emotion = "happy"
                    elif dominant_emotion_face != emotions_voice and dominant_emotion_face in ["disgust", "fear", "sad"] and emotions_voice in ["sad", "angry"]:
                        final_emotion = "sad"
                    elif dominant_emotion_face != emotions_voice and dominant_emotion_face in ["fear", "surprise"] and emotions_voice in ["sad", "neutral"]:
                        final_emotion = "fear"
                    elif dominant_emotion_face != emotions_voice and dominant_emotion_face in ["neutral", "angry", "sad"] and emotions_voice in ["sad", "angry", "neutral"]:
                        final_emotion = "sad"
                    break

        print("Emotion: " + final_emotion)
        return final_emotion

def get_corpus_file_path():
    global text

    if final_emotion == 'neutral':
        text = "Hi, how are you?"

    elif final_emotion == 'happy':
        text = "Someone seems happy!"

    elif final_emotion == 'sad':
        text = "Someone is sad, what is the matter?"

    elif final_emotion == 'angry':
        text = "You have a bad temper now, what's the matter?"

    elif final_emotion == 'disgust':
        text = "What is that?!"

    elif final_emotion == 'surprise':
        text = "OMG! What just happened?"

    elif final_emotion == 'fear':
        text = "Woah! What happened?!"

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save('output.mp3')
    sound = SoundLoader.load('output.mp3')
    if sound:
        sound.play()

@app.route('/', methods=['GET', 'POST'])
def home():
    final_emotion = perform_real_time_prediction()
    get_corpus_file_path()
    speak(text)
    if request.method == 'POST':
        input_text = recognize_speech()
        if not input_text:
            return render_template('index.html', response="Sorry, I didn't understand that.", final_emotion='')

        if input_text in casual_responses:
            response = casual_responses[input_text]
        else:
            response = retrieve_response(input_text, df)
            if not response:
                response = random.choice(fallback_responses)

        speak(response)
        return render_template('index.html', response=response, final_emotion=final_emotion)

    return render_template('index.html', response='', final_emotion=final_emotion)

@app.route('/record', methods=['POST'])
def record_speech():
    input_text = recognize_speech()
    if not input_text:
        speak('Sorry, I didn\'t understand that.')
        return 'Sorry, I didn\'t understand that.'

    if input_text in casual_responses:
        response = casual_responses[input_text]
    else:
        response = retrieve_response(input_text, df)
        if not response:
            response = random.choice(fallback_responses)

    speak(response)
    return response

@app.route('/analyze_latest_screenshot', methods=['GET'])
def analyze_latest_screenshot():
    final_emotion = perform_real_time_prediction()
    if final_emotion:
        get_corpus_file_path()
        speak(text)
        return {
            "emotion": final_emotion,
            "message": text
        }
    else:
        return {"error": "No screenshots available or failed to process the screenshot."}, 404

if __name__ == '__main__':
    print("Capturing initial screenshots...")
    take_screenshots()
    print("Screenshots captured. Starting the application...")
    app.run()