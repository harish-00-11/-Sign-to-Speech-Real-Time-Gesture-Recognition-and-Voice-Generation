# ✋ Gesture Control System

A powerful and modular real-time gesture recognition system built using computer vision and machine learning. The system supports hand gesture recognition, sign language interpretation, air drawing, emergency alerts, and voice interaction — all through a simple and intuitive interface.

## 🔥 Features

- ✋ **Hand Gesture Recognition**  
  Real-time classification of static and dynamic gestures using a trained CNN model.

- 🧏 **Sign Language Detection**  
  Detects and interprets American Sign Language (ASL) alphabets and commonly used sign gestures.

- 🖌️ **Air Gesture Canvas**  
  Draw characters in the air using finger tracking and convert handwriting to text using OCR.

- 🚨 **Emergency Alert System**  
  Recognizes critical gestures (e.g., "Help Me") and sends an alert email with GPS location and an image snapshot.

- 🗣️ **Text-to-Speech Feedback**  
  Automatically converts recognized gestures or air-drawn text into voice responses using gTTS.

- 🧠 **Training Modules**  
  Includes tools to collect gesture images and train custom models for gesture and sign language recognition.

## 📁 Modules Overview

| File Name             | Description                                                         |
|-----------------------|---------------------------------------------------------------------|
| `app.py`              | Main GUI application to access all features                         |
| `air_ges.py`          | Air drawing canvas with OCR and TTS support                         |
| `gui.py`              | Real-time sign language detection interface                         |
| `gesture_alert.py`    | Emergency gesture alert system with email and location reporting    |
| `data_collection.py`  | Tool to collect new gesture images for training                     |
| `Training.py`         | Trains a CNN model for hand gesture recognition                     |
| `sign_training.py`    | Trains a CNN model for sign language gesture classification         |

## 🚀 Installation

1. **Clone the Repository**  
   Download or clone the project to your local system:
   ```bash
   git clone https://github.com/yourusername/gesture-control-system.git
   cd gesture-control-system


2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install additional components**:
   - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for air canvas)
   - [MediaPipe](https://google.github.io/mediapipe/)
   - [OpenCV](https://opencv.org/)
   - [TensorFlow](https://www.tensorflow.org/)

## 🧪 Usage

Run the main app:
```bash
python app.py
```
Then choose:
- **Hand Gesture Recognition**
- **Sign Language Detection**
- **Air Gesture Canvas**

## ⚙️ Configuration

**For email alerts** in `gesture_alert.py`, update:
```python
SENDER_EMAIL = "your_email@gmail.com"
RECEIVER_EMAIL = "receiver_email@gmail.com"
EMAIL_PASSWORD = "your_app_password" ## here get the passwords from the app password option in the google Account
```

**For Tesseract OCR** in `air_ges.py`, set the correct path:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## 📸 Data Collection

Collect training data for new gestures:
```bash
python data_collection.py
```
Follow on-screen prompts to save real and thresholded images.

## 🧠 Training Models

Train gesture models:
```bash
python Training.py          # For general hand gestures
python sign_training.py     # For sign language gestures
```

## 📂 Project Structure

```
gesture-control-system/
├── Data/                   # Hand gesture images
├── Data_sign/              # Sign language images
├── Model/                  # Trained models
├── app.py                  # GUI launcher
├── air_ges.py              # Air canvas + OCR
├── gui.py                  # Sign recognition interface
├── gesture_alert.py        # Email alert system
├── data_collection.py      # Image data collection
├── Training.py             # Gesture training script
├── sign_training.py        # Sign language training
└── README.md               # Project overview
```

## Dataset link 
kaggle- https://www.kaggle.com/datasets/kira0182/sign-language-a-z-and-three-gestures

## 🤝 Contributing

We welcome contributions! Fork the repo, create a new branch, make your changes, and submit a pull request.

## 📜 License

This project is licensed under the [MIT License](LICENSE).
```

