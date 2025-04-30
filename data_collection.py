import cv2
import os
import time
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Constants
offset = 20
imgSize = 300
capture_images = False  # Flag to start capturing
images_to_capture = 1500  # Number of images to collect
counter = 0

# Set class name dynamically
class_name = input("Enter the gesture name: ").strip().capitalize()
folder = f"Data_sign/{class_name}"
os.makedirs(folder, exist_ok=True)

print("Press SPACE to start capturing 1500 images.")

while True:
    success, img = cap.read()
    if not success:
        print("Camera not detected!")
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[max(0, y - offset):y + h + offset, max(0, x - offset):x + w + offset]

        if imgCrop.size == 0:
            continue

        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize
        except Exception as e:
            print("Error in resizing:", e)
            continue

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        # Capture images if flag is enabled
        if capture_images and counter < images_to_capture:
            counter += 1
            img_path = f'{folder}/Image_{time.time()}.jpg'
            cv2.imwrite(img_path, imgWhite)
            print(f"Saved: {img_path} ({counter})")

            if counter >= images_to_capture:
                print("1500 images captured successfully!")
                capture_images = False  # Stop capturing

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)

    # Start capturing images when spacebar is pressed
    if key == ord(" "):
        print("📸 Capturing 1500 images...")
        capture_images = True
        counter = 0  # Reset counter before new capture session

    # Exit on 'q'
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
