import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import tkinter as tk
from tkinter import Label, Button, Frame, messagebox
from PIL import Image, ImageTk
import os
from collections import deque
import re
from gtts import gTTS
import pygame
import tempfile
import time
import enchant  # Python spell-checking library

# Initialize pygame mixer
pygame.mixer.init()

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition")

        # Set background image
        try:
            bg_image = Image.open(
                "C:/Users/Xxx/xx/Sign-to-Speech-Real-Time-Gesture-Recognition-and-Voice-Generation/imges/sky.jpg")
            self.bg_photo = ImageTk.PhotoImage(bg_image)
            self.background_label = Label(root, image=self.bg_photo)
            self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception as e:
            self.root.configure(bg='black')
            print(f"Failed to load background image: {str(e)}")

        try:
            # Load model quietly
            self.model = tf.keras.models.load_model("Model/sign_language_model_optimized.h5")
            with open("Model/labels.txt", 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.root.destroy()
            return

        # Configuration parameters
        self.img_size = 224
        self.confidence_threshold = 0.75
        self.smoothing_window = 5
        self.prediction_buffer = deque(maxlen=self.smoothing_window)
        self.padding = 40
        self.last_gesture = None
        self.gesture_count = 0
        self.gesture_threshold = 3

        # Initialize English dictionary
        self.dictionary = enchant.Dict("en_US")

        # Main container frame
        self.main_frame = Frame(root, bg='black', bd=2, relief='groove')
        self.main_frame.pack(padx=10, pady=10)

        # Title label
        self.label_title = Label(self.main_frame, text="Sign Language Recognition",
                                 font=("Arial", 18, "bold"), bg='black', fg='white')
        self.label_title.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # Image display area
        try:
            self.asl_image = Image.open(
                "C:/Users/gokul/Downloads/Realtime-Sign-Language-Detection-Using-LSTM-Model-main/imges/asl.jpg")
            self.asl_image = self.asl_image.resize((400, 300), Image.LANCZOS)
            self.asl_photo = ImageTk.PhotoImage(self.asl_image)
            self.image_label = Label(self.main_frame, image=self.asl_photo, bg='black')
            self.image_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        except Exception as e:
            self.image_label = Label(self.main_frame, text="ASL Image Placeholder", bg='black', fg='white')
            self.image_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))

        # Video display area (initially hidden)
        self.video_label = Label(self.main_frame, bg='black')
        self.video_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        self.video_label.grid_remove()

        # Detection area frame
        self.detection_frame = Frame(self.main_frame, bg='black')
        self.detection_frame.grid(row=2, column=0, columnspan=2, pady=(0, 10), sticky="ew")

        # Detected gesture label
        self.detected_label = Label(self.detection_frame, text="Detected Gesture: ",
                                    font=("Arial", 14), bg='black', fg='white')
        self.detected_label.pack(fill='x', pady=(0, 5))

        # Confidence label
        self.confidence_label = Label(self.detection_frame, text="Confidence: ",
                                      font=("Arial", 12), bg='black', fg='white')
        self.confidence_label.pack(fill='x')

        # Generated sentence label
        self.sentence_label = Label(self.detection_frame, text="Sentence: ",
                                    font=("Arial", 14), bg='black', fg='white')
        self.sentence_label.pack(fill='x')

        # Next suggestion label
        self.suggestion_label = Label(self.detection_frame, text="Next Suggestion: ",
                                      font=("Arial", 12), bg='black', fg='cyan')
        self.suggestion_label.pack(fill='x')

        # Auto-corrected label
        self.corrected_label = Label(self.detection_frame, text="Auto-Corrected: ",
                                     font=("Arial", 12), bg='black', fg='yellow')
        self.corrected_label.pack(fill='x')

        # Button frame
        self.button_frame = Frame(self.main_frame, bg='black')
        self.button_frame.grid(row=3, column=0, columnspan=2, pady=(5, 5))

        # Buttons
        self.start_button = Button(self.button_frame, text="Start Detection", bg='#333333', fg='white',
                                   font=("Arial", 12), command=self.start_detection)
        self.start_button.pack(side='left', padx=5, ipadx=10)

        self.toggle_button = Button(self.button_frame, text="Toggle Mode: Single Gesture", bg='#333333', fg='white',
                                    font=("Arial", 12), command=self.toggle_mode)
        self.toggle_button.pack(side='left', padx=5, ipadx=10)

        self.speak_button = Button(self.button_frame, text="Speak Sentence", bg='#333333', fg='white',
                                   font=("Arial", 12), command=self.speak_sentence)
        self.speak_button.pack(side='left', padx=5, ipadx=10)

        self.clear_button = Button(self.button_frame, text="Clear Sentence", bg='#333333', fg='white',
                                   font=("Arial", 12), command=self.clear_sentence)
        self.clear_button.pack(side='left', padx=5, ipadx=10)

        # Close button
        self.close_frame = Frame(self.main_frame, bg='black')
        self.close_frame.grid(row=4, column=0, columnspan=2, pady=(5, 0), sticky='ew')
        self.close_button = Button(self.close_frame, text="Close", bg='#333333', fg='white',
                                   font=("Arial", 12), command=self.close_app)
        self.close_button.pack(fill='x')

        # Hand detector setup
        self.detector = HandDetector(maxHands=1, detectionCon=0.8)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            self.root.destroy()
            return

        # State variables
        self.running = False
        self.camera_active = False
        self.sentence_mode = False
        self.current_sentence = []
        self.alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                         'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                         'W', 'X', 'Y', 'Z']

    def auto_correct(self, text):
        """Auto-correct using enchant dictionary"""
        text = text.upper()

        # First remove consecutive duplicate letters
        corrected = re.sub(r'(.)\1+', r'\1', text)

        # Check if the word exists in dictionary
        if self.dictionary.check(corrected):
            return corrected

        # If not, try to suggest corrections
        suggestions = self.dictionary.suggest(corrected)
        if suggestions:
            return suggestions[0].upper()  # Return first suggestion

        return text  # Return original if no correction found

    def get_next_suggestion(self):
        if not self.current_sentence:
            return "Start with any letter"

        last_char = self.current_sentence[-1].upper()
        if last_char in self.alphabet:
            current_index = self.alphabet.index(last_char)
            next_index = (current_index + 1) % len(self.alphabet)
            return f"Next: {self.alphabet[next_index]}"
        return "Next letter suggestion"

    def toggle_mode(self):
        self.sentence_mode = not self.sentence_mode
        mode_text = "Sentence Mode" if self.sentence_mode else "Single Gesture Mode"
        self.toggle_button.config(text=f"Toggle Mode: {mode_text}")
        self.clear_sentence()

    def speak_sentence(self):
        if self.current_sentence:
            sentence = ' '.join(self.current_sentence)
            try:
                with tempfile.NamedTemporaryFile(delete=True) as fp:
                    tts = gTTS(text=sentence, lang='en')
                    tts.save(fp.name + '.mp3')
                    time.sleep(0.1)
                    pygame.mixer.music.load(fp.name + '.mp3')
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
            except Exception as e:
                messagebox.showerror("Speech Error", f"Could not speak sentence: {str(e)}")
        else:
            messagebox.showinfo("Info", "No sentence to speak")

    def close_app(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        pygame.mixer.quit()
        self.root.quit()

    def preprocess_image(self, img_crop):
        """Improved preprocessing"""
        if img_crop is None or img_crop.size == 0:
            return None

        img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape
        aspect_ratio = w / h

        if aspect_ratio > 1:
            new_w = self.img_size
            new_h = int(self.img_size / aspect_ratio)
        else:
            new_h = self.img_size
            new_w = int(self.img_size * aspect_ratio)

        img_resized = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        img_padded = np.full((self.img_size, self.img_size, 3), 255, dtype=np.uint8)
        y_offset = (self.img_size - new_h) // 2
        x_offset = (self.img_size - new_w) // 2
        img_padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized
        img_normalized = img_padded / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)

        return img_input

    def predict_gesture(self, img_input):
        try:
            predictions = self.model.predict(img_input, verbose=0)
            index = np.argmax(predictions)
            confidence = predictions[0][index]
            return self.labels[index], confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None

    def smooth_predictions(self, gesture, confidence):
        if confidence > self.confidence_threshold:
            if gesture == self.last_gesture:
                self.gesture_count += 1
            else:
                self.last_gesture = gesture
                self.gesture_count = 1

            self.prediction_buffer.append((gesture, confidence))

        if len(self.prediction_buffer) > 0 and self.gesture_count >= self.gesture_threshold:
            gestures, confidences = zip(*self.prediction_buffer)
            final_gesture = max(set(gestures), key=gestures.count)
            final_confidence = np.mean([conf for g, conf in self.prediction_buffer if g == final_gesture])
            return final_gesture, final_confidence
        return None, None

    def start_detection(self):
        if not self.camera_active:
            self.image_label.grid_remove()
            self.video_label.grid()
            self.camera_active = True
        self.running = True
        self.update_frame()

    def update_frame(self):
        if not self.running:
            return

        success, img = self.cap.read()
        if not success:
            print("Failed to capture frame")
            self.root.after(10, self.update_frame)
            return

        hands, img = self.detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            pad = self.padding
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
            img_crop = img[y1:y2, x1:x2]

            img_input = self.preprocess_image(img_crop)

            if img_input is not None:
                gesture, confidence = self.predict_gesture(img_input)
                final_gesture, final_confidence = self.smooth_predictions(gesture, confidence)

                if final_gesture and final_confidence > self.confidence_threshold:
                    self.detected_label.config(text=f"Detected Gesture: {final_gesture}")
                    self.confidence_label.config(text=f"Confidence: {final_confidence:.2f}")

                    if self.sentence_mode:
                        corrected = self.auto_correct(final_gesture)
                        self.current_sentence.append(corrected)
                        self.sentence_label.config(text=f"Sentence: {' '.join(self.current_sentence)}")
                        self.corrected_label.config(text=f"Auto-Corrected: {corrected}")
                        self.suggestion_label.config(text=f"Next Suggestion: {self.get_next_suggestion()}")
                        self.last_gesture = None
                        self.gesture_count = 0

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.video_label.config(image=img)
        self.video_label.image = img

        self.root.after(10, self.update_frame)

    def clear_sentence(self):
        self.current_sentence = []
        self.sentence_label.config(text="Sentence: ")
        self.suggestion_label.config(text="Next Suggestion: Start with any letter")
        self.corrected_label.config(text="Auto-Corrected: ")
        self.last_gesture = None
        self.gesture_count = 0


if __name__ == "__main__":
    root = tk.Tk()
    try:
        app = SignLanguageApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal Error", f"Application crashed: {str(e)}")
