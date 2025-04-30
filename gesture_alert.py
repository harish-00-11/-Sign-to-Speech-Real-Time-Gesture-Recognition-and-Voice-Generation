import os
import time
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import smtplib
import geocoder
from geopy.geocoders import Nominatim
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import pyttsx3
import threading

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Configuration constants
SENDER_EMAIL = "your_email@gmail.com"
RECEIVER_EMAIL = "receiver_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
INPUT_SIZE = 224
ALERT_EXPIRY_MINUTES = 15
MIN_CONFIDENCE = 0.6
CONSECUTIVE_FRAMES = 5

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Initialize MediaPipe hands with optimized parameters
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load model and labels
model = tf.keras.models.load_model('Model/sign_language_model_ER.keras')
with open('Model/Labels_ER.txt', 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]

GESTURE_MESSAGES = {
    "Pain": {
        "email_subject": "⚠️ PAIN DETECTED",
        "email_body": "The person is signaling pain. Please check on them.",
        "voice_message": "Pain detected. Someone needs assistance."
    },
    "Help me": {
        "email_subject": "🆘 HELP ME REQUESTED",
        "email_body": "The person is requesting help. Please check on them.",
        "voice_message": "Help me requested. Someone needs assistance."
    },
    "Er": {
        "email_subject": "⚠️ ALERT DETECTED",
        "email_body": "An alert signal was detected. Please check on the person.",
        "voice_message": "Alert detected. Someone needs assistance."
    }
}


class LocationTracker:
    def __init__(self):
        self.last_location = None
        self.location_time = None

    def get_current_location(self):
        """Get fresh location data every time (no caching)"""
        try:
            # Get precise location using IP
            g = geocoder.ip('me')
            if not g.latlng:
                return None

            lat, lng = g.latlng
            geolocator = Nominatim(user_agent="gesture_alert_system")
            location = geolocator.reverse(f"{lat}, {lng}", exactly_one=True)

            return {
                "address": location.address if location else "Location unavailable",
                "maps_link": f"https://www.google.com/maps?q={lat},{lng}",
                "coordinates": f"{lat},{lng}",
                "timestamp": time.time()
            }
        except Exception as e:
            print(f"Location error: {e}")
            return None


location_tracker = LocationTracker()


class GestureDetector:
    def __init__(self):
        self.detection_history = []
        self.last_detection_time = 0
        self.stable_gesture = None
        self.stable_confidence = 0
        self.last_spoken_time = 0
        self.speech_cooldown = 5
    def update_detection(self, gesture, confidence):
        current_time = time.time()
        if current_time - self.last_detection_time > 1.0:
            self.detection_history = []
        self.last_detection_time = current_time

        self.detection_history.append((gesture, confidence))
        if len(self.detection_history) > CONSECUTIVE_FRAMES:
            self.detection_history.pop(0)

        if len(self.detection_history) == CONSECUTIVE_FRAMES:
            gesture_counts = {}
            for g, c in self.detection_history:
                gesture_counts[g] = gesture_counts.get(g, 0) + 1
            most_common = max(gesture_counts.items(), key=lambda x: x[1])
            if most_common[1] >= CONSECUTIVE_FRAMES - 1:
                self.stable_gesture = most_common[0]
                confidences = [c for g, c in self.detection_history if g == self.stable_gesture]
                self.stable_confidence = sum(confidences) / len(confidences)

                if (self.stable_confidence > MIN_CONFIDENCE and
                        current_time - self.last_spoken_time > self.speech_cooldown):
                    self.speak_gesture(self.stable_gesture)
                    self.last_spoken_time = current_time

        return self.stable_gesture, self.stable_confidence

    def speak_gesture(self, gesture):
        """Speak the gesture without saving any audio files"""

        def speak():
            try:
                if gesture in GESTURE_MESSAGES:
                    engine.say(GESTURE_MESSAGES[gesture]["voice_message"])
                else:
                    engine.say(f"{gesture} detected")
                engine.runAndWait()
            except Exception as e:
                print(f"Speech error: {e}")

        # Run speech in a separate thread to avoid blocking
        threading.Thread(target=speak, daemon=True).start()


gesture_detector = GestureDetector()


def send_alert_email(gesture, frame):
    try:
        if gesture not in GESTURE_MESSAGES:
            print(f"No message configuration for gesture: {gesture}")
            return

        # Always get fresh location data
        location = location_tracker.get_current_location()
        if location is None:
            location = {
                "address": "Location unavailable",
                "maps_link": "https://www.google.com/maps",
                "coordinates": "0,0"
            }

        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = GESTURE_MESSAGES[gesture]['email_subject']

        body = f"""
        {GESTURE_MESSAGES[gesture]['email_body']}<br>
        <p><strong>Detected Gesture:</strong> {gesture}</p>
        <p><strong>Location:</strong> {location['address']}</p>
        <p><strong>Coordinates:</strong> {location['coordinates']}</p>
        <p><a href="{location['maps_link']}">View on Google Maps</a></p>
        <p><strong>Time:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        msg.attach(MIMEText(body, 'html'))

        # Attach the current frame
        resized_frame = cv2.resize(frame, (640, 480))
        _, img_encoded = cv2.imencode('.jpg', resized_frame)
        msg.attach(MIMEImage(img_encoded.tobytes(), name="gesture.jpg"))

        # Send email in a separate thread
        def send_email():
            try:
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                    server.login(SENDER_EMAIL, EMAIL_PASSWORD)
                    server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
                print(f"Alert sent for {gesture}")
            except Exception as e:
                print(f"Email error: {str(e)}")

        threading.Thread(target=send_email, daemon=True).start()

    except Exception as e:
        print(f"Alert creation error: {str(e)}")


def process_frame(frame):
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(rgb_frame)
    detection = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate hand bounding box
            x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf
            h, w = frame.shape[:2]

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min, x_max, y_max = min(x_min, x), min(y_min, y), max(x_max, x), max(y_max, y)

            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            hand_roi = frame[y_min:y_max, x_min:x_max]

            if hand_roi.size == 0:
                continue

            try:
                # Resize and normalize the ROI
                input_data = tf.image.resize(hand_roi / 255.0, [INPUT_SIZE, INPUT_SIZE])

                # Batch prediction for better performance
                predictions = model.predict(np.expand_dims(input_data, axis=0), verbose=0)
                gesture = class_labels[np.argmax(predictions)]
                confidence = np.max(predictions)

                stable_gesture, stable_confidence = gesture_detector.update_detection(gesture, confidence)

                if stable_gesture:
                    detection = (stable_gesture, stable_confidence)
                    cv2.putText(frame, f"{stable_gesture} ({stable_confidence:.2f})",
                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                print(f"Prediction error: {e}")

    return frame, detection


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    # Set camera resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_alert_time = 0
    alert_cooldown = 30  # seconds between alerts

    try:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            processed_frame, detection = process_frame(frame)

            cv2.imshow('Gesture Detection', processed_frame)

            current_time = time.time()
            if detection and detection[1] > MIN_CONFIDENCE:
                gesture, confidence = detection
                print(f"Detected: {gesture} with confidence {confidence:.2f}")
                if current_time - last_alert_time > alert_cooldown:
                    send_alert_email(gesture, frame)
                    last_alert_time = current_time

            # Calculate processing time and adjust waitKey accordingly
            processing_time = time.time() - start_time
            delay = max(1, int(30 - (processing_time * 1000)))

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
