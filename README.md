# Gesture Control System

A comprehensive system for hand gesture recognition, sign language detection, and air gesture control using computer vision and machine learning.

## Features

- **Hand Gesture Recognition**: Real-time detection of hand gestures with a trained model.
- **Sign Language Detection**: Recognition of ASL (American Sign Language) alphabets and common signs.
- **Air Gesture Control**: Draw in the air and have your drawings recognized as text.
- **Alert System**: Send email alerts with location when specific emergency gestures are detected.
- **Text-to-Speech**: Convert detected gestures or drawn text to spoken words.
- **Training Modules**: Tools to train new gesture recognition models.

## Modules

1. **Main Application (`app.py`)**: Central hub to access all features.
2. **Air Gesture Canvas (`air_ges.py`)**: Draw in air and recognize text.
3. **Sign Language GUI (`gui.py`)**: Real-time sign language detection interface.
4. **Gesture Alert System (`gesture_alert.py`)**: Emergency gesture detection with alerts.
5. **Data Collection (`data_collection.py`)**: Capture hand gesture images for training.
6. **Model Training (`Training.py`, `sign_training.py`)**: Scripts to train gesture recognition models.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gesture-control-system.git
   cd gesture-control-system
Install required dependencies:

bash
pip install -r requirements.txt
Install additional components:

Tesseract OCR (for text recognition in air gestures)

MediaPipe (for hand tracking)

OpenCV (for computer vision)

TensorFlow (for machine learning)

Usage
Run the main application:

bash
python app.py
Select from the available modules:

Hand Gesture Recognition

Sign Language Detection

Air Gesture Control

Configuration
For email alerts in gesture_alert.py, update:

python
SENDER_EMAIL = "your_email@gmail.com"
RECEIVER_EMAIL = "recipient_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"  # Use app-specific password
For Tesseract OCR in air_ges.py, set the correct path:

python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
Data Collection
To collect new gesture data:

bash
python data_collection.py
Follow on-screen instructions to capture images for training.

Model Training
Train new models:

bash
python Training.py          # For hand gestures
python sign_training.py     # For sign language
Requirements
Python 3.8+

OpenCV

MediaPipe

TensorFlow 2.x

Tesseract OCR

Pygame

gTTS

Other dependencies in requirements.txt

File Structure
Gesture-Control-System/
├── Data/                   # Training data for hand gestures
├── Data_sign/              # Training data for sign language
├── Model/                  # Saved models and labels
├── air_ges.py              # Air gesture control module
├── app.py                  # Main application
├── data_collection.py      # Data collection tool
├── gesture_alert.py        # Emergency gesture alert system
├── gui.py                  # Sign language GUI
├── sign_training.py        # Sign language model training
├── Training.py             # Hand gesture model training
└── README.md               # This file
Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

License
This project is licensed under the MIT License - see the LICENSE file for details.
