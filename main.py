import streamlit as st
import cv2
from HandTrackingModule import HandDetector
from ClassificationModule import Classifier
import numpy as np
import math
import time
from PIL import Image
import pyttsx3
import threading
import queue

# Initialize variables
cap = cv2.VideoCapture(0)
detector1 = HandDetector(maxHands=1)
detector2 = HandDetector(maxHands=2)
offset = 20
imgSize = 300
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

classifier1 = Classifier("model_asl/keras_model.h5", "model_asl/labels.txt")
classifier2 = Classifier("ISl_Teachable/keras_model.h5", "TechableISL_upgrade/labels.txt")

# Initialize Text-to-Speech with a queue system
tts_available = False
speech_queue = queue.Queue()
tts_engine = None
speech_thread = None
speech_active = False

def speech_worker():
    global speech_active
    while True:
        text = speech_queue.get()
        if text is None:
            break
        speech_active = True
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            st.error(f"TTS Error: {e}")
        finally:
            speech_active = False
        speech_queue.task_done()

def speak_text(text):
    if tts_available and not speech_active:
        speech_queue.put(text)

try:
    tts_engine = pyttsx3.init()
    tts_available = True
    speech_thread = threading.Thread(target=speech_worker, daemon=True)
    speech_thread.start()
    voices = tts_engine.getProperty('voices')
    default_voice = 0
except Exception as e:
    tts_available = False
    st.warning(f"TTS unavailable: {e}")

# Session state initialization
state_keys = [
    "current_word", "words", "last_letter", "last_detection_time", "stable_letter",
    "stable_counter", "last_added_time", "auto_speak_letter"
]
for key in state_keys:
    if key not in st.session_state:
        st.session_state[key] = "" if "word" in key or "letter" in key else 0

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Multilingual Sign Language Recognizer")

try:
    logo = Image.open("logo.ico")
    st.sidebar.image(logo, width=80)
except:
    st.sidebar.write("Logo not found")

selected_model = st.sidebar.radio("Choose Sign Language Model:", ["American Sign Language", "Indian Sign Language"])
use_code1 = selected_model == "American Sign Language"
use_code2 = selected_model == "Indian Sign Language"
st.sidebar.write(f"Currently using: {selected_model}")

col_chart1, col_chart2 = st.sidebar.columns(2)
with col_chart1:
    if st.button("View ASL Chart"):
        st.sidebar.image("Charts/ASL_CHART.png", use_container_width=True)
with col_chart2:
    if st.button("View ISL Chart"):
        st.sidebar.image("Charts/ISL_CHART.jpg", use_container_width=True)

stability_threshold = st.sidebar.slider("Stability Threshold (frames)", 5, 30, 15)
cooldown_period = st.sidebar.slider("Letter Cooldown (seconds)", 0.5, 3.0, 1.0)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 1.0, 0.7)

st.sidebar.subheader("Text-to-Speech Settings")
if tts_available:
    if len(voices) > 0:
        voice_names = [voice.name for voice in voices]
        selected_voice = st.sidebar.selectbox("Select Voice", voice_names, index=default_voice)
        selected_voice_index = voice_names.index(selected_voice)
        tts_engine.setProperty('voice', voices[selected_voice_index].id)

    rate = st.sidebar.slider("Speech Rate", 100, 300, 150)
    tts_engine.setProperty('rate', rate)

    volume = st.sidebar.slider("Volume", 0.0, 1.0, 0.7)
    tts_engine.setProperty('volume', volume)

    st.session_state.auto_speak_letter = st.sidebar.checkbox("Speak Each Letter", value=False)

# Layout
cam_col, control_col = st.columns([3, 1])
frame_placeholder = cam_col.empty()
text_status_container = cam_col.container()
text_display = text_status_container.empty()
status_display = text_status_container.empty()
speech_status = cam_col.empty()

with control_col:
    st.subheader("Controls")
    if st.button("Add Space"):
        st.session_state.current_word += " "
    if st.button("Backspace"):
        st.session_state.current_word = st.session_state.current_word[:-1]
    if st.button("Clear All"):
        st.session_state.current_word = ""
        st.session_state.words = []

    if tts_available:
        if st.button("Speak Current Text"):
            if st.session_state.current_word:
                speak_text(st.session_state.current_word)
        if st.button("Add Word & Speak"):
            if st.session_state.current_word:
                speak_text(st.session_state.current_word)
                st.session_state.words.append(st.session_state.current_word)
                st.session_state.current_word = ""

# Start capturing video
while cap.isOpened():
    success, img = cap.read()
    if not success:
        st.error("Failed to capture video")
        break

    imgOutput = img.copy()
    hands, img = (detector1.findHands(img) if use_code1 else detector2.findHands(img))
    current_letter = None

    if speech_active:
        speech_status.info("Speaking...")
    else:
        speech_status.empty()

    try:
        if hands:
            if use_code1:
                hand = hands[0]
                x, y, w, h = hand['bbox']
            else:
                x1, y1, w1, h1 = hands[0]['bbox']
                if len(hands) == 2:
                    x2, y2, w2, h2 = hands[1]['bbox']
                    x, y = min(x1, x2), min(y1, y2)
                    w = max(x1 + w1, x2 + w2) - x
                    h = max(y1 + h1, y2 + h2) - y
                else:
                    x, y, w, h = x1, y1, w1, h1

            y_min = max(y - offset, 0)
            y_max = min(y + h + offset, img.shape[0])
            x_min = max(x - offset, 0)
            x_max = min(x + w + offset, img.shape[1])
            imgCrop = img[y_min:y_max, x_min:x_max]

            if imgCrop.size == 0:
                raise ValueError("Empty crop")

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            classifier = classifier1 if use_code1 else classifier2
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            if prediction[index] >= confidence_threshold:
                current_letter = labels[index]
                cv2.rectangle(imgOutput, (x_min, y_min - 50), (x_min + 90, y_min), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, current_letter, (x_min + 10, y_min - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x_min, y_min), (x_max, y_max), (255, 0, 255), 4)

                if current_letter == st.session_state.stable_letter:
                    st.session_state.stable_counter += 1
                else:
                    st.session_state.stable_letter = current_letter
                    st.session_state.stable_counter = 1

                current_time = time.time()
                if (st.session_state.stable_counter >= stability_threshold and
                    current_time - st.session_state.last_added_time >= cooldown_period):
                    st.session_state.current_word += current_letter
                    st.session_state.last_added_time = current_time

                    if tts_available and st.session_state.auto_speak_letter and not speech_active:
                        speak_text(current_letter)

                    st.session_state.stable_counter = 0

                progress = min(st.session_state.stable_counter / stability_threshold, 1.0)
                status_display.progress(progress)
                status_display.write(f"Detecting: {current_letter} ({int(progress * 100)}%)")
            else:
                status_display.write("Sign not recognized clearly")
                st.session_state.stable_counter = 0
        else:
            status_display.write("No hands detected")
            st.session_state.stable_counter = 0

    except Exception:
        status_display.write("Detection issue: Try repositioning your hand")
        st.session_state.stable_counter = 0

    text_display.markdown(f"## Recognized Text: {st.session_state.current_word}")
    imgRGB = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(imgRGB, channels="RGB")
    time.sleep(0.001)

if tts_available and speech_thread and speech_thread.is_alive():
    speech_queue.put(None)
    speech_thread.join(timeout=1)

cap.release()
