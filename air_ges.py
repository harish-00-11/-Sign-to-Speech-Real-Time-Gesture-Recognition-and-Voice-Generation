import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import colorchooser, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import io
import pytesseract
import string
import pygame
from gtts import gTTS
import time
import os

# Initialize pygame mixer
pygame.mixer.init()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.7,
    static_image_mode=False
)
mp_draw = mp.solutions.drawing_utils

# Create main window
root = tk.Tk()
root.title("Air Canvas with Hand Tracking & Text Detection")
root.state('zoomed')

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Helper function to create fallback background
def create_fallback_bg(width, height, color):
    img = Image.new('RGB', (width, height), color)
    return ImageTk.PhotoImage(img)

# Load background images with fallbacks
try:
    main_bg = Image.open(r"C:\Users\gokul\Downloads\Realtime-Sign-Language-Detection-Using-LSTM-Model-main\imges\sky.jpg")
    main_bg = main_bg.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.Resampling.LANCZOS)
    main_bg_photo = ImageTk.PhotoImage(main_bg)
except:
    main_bg_photo = create_fallback_bg(root.winfo_screenwidth(), root.winfo_screenheight(), '#f0f0f0')

try:
    camera_bg = Image.open("camera_bg.png")
    camera_bg = camera_bg.resize((640, 480), Image.Resampling.LANCZOS)
    camera_bg_photo = ImageTk.PhotoImage(camera_bg)
except:
    camera_bg_photo = create_fallback_bg(640, 480, '#e0e0e0')

try:
    paper_bg = Image.open("paper_bg.png")
    paper_bg = paper_bg.resize((800, 600), Image.Resampling.LANCZOS)
    paper_bg_photo = ImageTk.PhotoImage(paper_bg)
except:
    paper_bg_photo = create_fallback_bg(800, 600, 'white')

# Create video frame
video_frame = tk.Frame(root, bd=2, relief=tk.GROOVE, bg="#333333")
video_frame.place(x=50, y=50, width=640, height=480)

# Create canvas frame
canvas_frame = tk.Frame(root, bd=2, relief=tk.GROOVE, bg="#333333")
canvas_frame.place(x=700, y=50, width=800, height=600)

canvas_width, canvas_height = 800, 600
canvas = tk.Canvas(canvas_frame, width=canvas_width, height=canvas_height, bg="white", highlightthickness=0)
canvas.create_image(0, 0, image=paper_bg_photo, anchor=tk.NW)
canvas.pack(padx=1, pady=1)

# Video label
video_label = tk.Label(video_frame, bd=0, highlightthickness=0)
video_label.config(image=camera_bg_photo)
video_label.pack(padx=1, pady=1)

# Title frame
title_frame = tk.Frame(root, bg="#333333")
title_frame.place(x=50, y=10, width=1450, height=30)

tk.Label(title_frame, text="Camera View", font=('Arial', 12, 'bold'),
         bg="#333333", fg="white").place(x=270, y=2)
tk.Label(title_frame, text="Air Canvas", font=('Arial', 12, 'bold'),
         bg="#333333", fg="white").place(x=1090, y=2)

# Global variables
brush_color = "#000000"
brush_size = 8
last_x, last_y = None, None
current_word = ""
detecting = False

def map_coordinates(x, y):
    scale_x = canvas_width / frame_width
    scale_y = canvas_height / frame_height
    return int(x * scale_x), int(y * scale_y)

def clear_canvas():
    canvas.delete("all")
    canvas.create_image(0, 0, image=paper_bg_photo, anchor=tk.NW)
    global current_word
    current_word = ""
    update_status()

def update_color_from_picker(event=None):
    global brush_color
    r = int(red_scale.get())
    g = int(green_scale.get())
    b = int(blue_scale.get())

    brush_color = f'#{r:02x}{g:02x}{b:02x}'
    color_preview.config(bg=brush_color)

    size_preview.delete("all")
    size_preview.create_oval(10 - brush_size / 2, 10 - brush_size / 2,
                             10 + brush_size / 2, 10 + brush_size / 2,
                             fill=brush_color, outline="")
    update_status()

def update_status():
    status_var.set(f"Color: {brush_color.upper()} | Size: {brush_size}px | Detected: {current_word}")

def save_canvas_image():
    img = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(img)

    for item in canvas.find_all():
        if canvas.type(item) == "image":
            continue

        coords = canvas.coords(item)
        color = canvas.itemcget(item, "fill")
        width = int(canvas.itemcget(item, "width"))

        if len(coords) >= 4:
            for i in range(0, len(coords) - 2, 2):
                draw.line([coords[i], coords[i + 1], coords[i + 2], coords[i + 3]],
                          fill=color, width=width)
    return img

def animate_detection():
    global detecting, current_word
    if detecting:
        return

    detecting = True
    detect_btn.config(state=tk.DISABLED)
    status_var.set("Detecting...")

    for i in range(1, 6):
        canvas.delete("scan_text")
        canvas.create_text(canvas_width / 2, canvas_height / 2,
                           text="Scanning" + "." * i,
                           font=("Arial", 24), fill=brush_color, tags="scan_text")
        root.update()
        time.sleep(0.15)

    current_word = detect_drawn_text()

    if current_word != "No text detected":
        canvas.delete("scan_text")
        canvas.create_text(canvas_width / 2, canvas_height / 2,
                           text=f"Detected: {current_word}",
                           font=("Arial", 24), fill=brush_color, tags="scan_text")
        root.update()
        time.sleep(1.5)

    canvas.delete("scan_text")
    detect_btn.config(state=tk.NORMAL)
    detecting = False
    update_status()

def detect_drawn_text():
    try:
        img = save_canvas_image()
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        text = pytesseract.image_to_string(processed,
                                           config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')

        cleaned_text = "".join([c for c in text if c in string.ascii_letters + " "]).strip()
        return cleaned_text if cleaned_text else "No text detected"
    except Exception as e:
        print(f"Error in text detection: {e}")
        return "Error"

def speak_text():
    global current_word
    text = current_word if current_word else detect_drawn_text()
    if text and text != "No text detected":
        try:
            tts = gTTS(text=text, lang='en')
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            pygame.mixer.music.load(audio_bytes)
            pygame.mixer.music.play()

            canvas.delete("speak_text")
            canvas.create_text(canvas_width / 2, canvas_height / 2,
                               text=f"Speaking: {text}",
                               font=("Arial", 24), fill=brush_color, tags="speak_text")
            root.update()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            canvas.delete("speak_text")

        except Exception as e:
            print(f"Error in speech generation: {e}")

def increase_brush_size():
    global brush_size
    brush_size = min(20, brush_size + 2)
    size_preview.delete("all")
    size_preview.create_oval(10 - brush_size / 2, 10 - brush_size / 2,
                             10 + brush_size / 2, 10 + brush_size / 2,
                             fill=brush_color, outline="")
    size_label.config(text=f"{brush_size}px")
    update_status()

def decrease_brush_size():
    global brush_size
    brush_size = max(2, brush_size - 2)
    size_preview.delete("all")
    size_preview.create_oval(10 - brush_size / 2, 10 - brush_size / 2,
                             10 + brush_size / 2, 10 + brush_size / 2,
                             fill=brush_color, outline="")
    size_label.config(text=f"{brush_size}px")
    update_status()

def choose_color():
    global brush_color
    color = colorchooser.askcolor(initialcolor=brush_color)
    if color[1]:
        brush_color = color[1]
        r, g, b = int(color[0][0]), int(color[0][1]), int(color[0][2])
        red_scale.set(r)
        green_scale.set(g)
        blue_scale.set(b)
        color_preview.config(bg=brush_color)
        size_preview.delete("all")
        size_preview.create_oval(10 - brush_size / 2, 10 - brush_size / 2,
                                 10 + brush_size / 2, 10 + brush_size / 2,
                                 fill=brush_color, outline="")
        update_status()

# Create control notebook
control_notebook = ttk.Notebook(root)
control_notebook.place(x=50, y=540, width=1450, height=120)

style = ttk.Style()
style.configure('TNotebook.Tab', font=('Arial', 11, 'bold'))
style.configure('TFrame', background='#f0f0f0')

# Drawing tools tab
drawing_tab = ttk.Frame(control_notebook)
control_notebook.add(drawing_tab, text="  Drawing Tools  ")

# Color picker frame
color_picker_frame = tk.LabelFrame(drawing_tab, text="Color Picker", font=('Arial', 10, 'bold'),
                                   bg='#f0f0f0', padx=5, pady=5)
color_picker_frame.pack(side=tk.LEFT, padx=10, pady=5, fill='y')

# Red scale
red_frame = tk.Frame(color_picker_frame, bg='#f0f0f0')
red_frame.pack(fill='x', pady=2)
tk.Label(red_frame, text="R:", font=('Arial', 10), bg='#f0f0f0', fg='red', width=2).pack(side=tk.LEFT)
red_scale = tk.Scale(red_frame, from_=0, to=255, orient='horizontal',
                     command=lambda x: update_color_from_picker(), length=150, bg='#f0f0f0')
red_scale.set(0)
red_scale.pack(side=tk.LEFT)

# Green scale
green_frame = tk.Frame(color_picker_frame, bg='#f0f0f0')
green_frame.pack(fill='x', pady=2)
tk.Label(green_frame, text="G:", font=('Arial', 10), bg='#f0f0f0', fg='green', width=2).pack(side=tk.LEFT)
green_scale = tk.Scale(green_frame, from_=0, to=255, orient='horizontal',
                       command=lambda x: update_color_from_picker(), length=150, bg='#f0f0f0')
green_scale.set(0)
green_scale.pack(side=tk.LEFT)

# Blue scale
blue_frame = tk.Frame(color_picker_frame, bg='#f0f0f0')
blue_frame.pack(fill='x', pady=2)
tk.Label(blue_frame, text="B:", font=('Arial', 10), bg='#f0f0f0', fg='blue', width=2).pack(side=tk.LEFT)
blue_scale = tk.Scale(blue_frame, from_=0, to=255, orient='horizontal',
                      command=lambda x: update_color_from_picker(), length=150, bg='#f0f0f0')
blue_scale.set(0)
blue_scale.pack(side=tk.LEFT)

# Color preview
color_preview_frame = tk.Frame(color_picker_frame, bg='#f0f0f0')
color_preview_frame.pack(pady=5)
tk.Label(color_preview_frame, text="Current:", font=('Arial', 10), bg='#f0f0f0').pack(side=tk.LEFT)
color_preview = tk.Label(color_preview_frame, width=6, height=1, bg=brush_color, relief='sunken', bd=2)
color_preview.pack(side=tk.LEFT, padx=5)

color_picker_btn = tk.Button(color_preview_frame, text="🎨", command=choose_color,
                             font=('Arial', 12), bg='#f0f0f0', padx=5, pady=0)
color_picker_btn.pack(side=tk.LEFT, padx=5)

# Brush size frame
size_frame = tk.LabelFrame(drawing_tab, text="Brush Size", font=('Arial', 10, 'bold'),
                           bg='#f0f0f0', padx=10, pady=5)
size_frame.pack(side=tk.LEFT, padx=10, pady=5, fill='y')

size_preview = tk.Canvas(size_frame, width=20, height=20, bg='white', highlightthickness=0)
size_preview.create_oval(10 - brush_size / 2, 10 - brush_size / 2,
                         10 + brush_size / 2, 10 + brush_size / 2,
                         fill=brush_color, outline="")
size_preview.pack(side=tk.TOP, pady=5)

size_buttons = tk.Frame(size_frame, bg='#f0f0f0')
size_buttons.pack(side=tk.TOP, pady=5)

tk.Button(size_buttons, text="−", command=decrease_brush_size,
          font=('Arial', 12, 'bold'), bg='#f0f0f0', fg='black',
          padx=5, pady=0, relief=tk.RAISED, bd=1, width=2).pack(side=tk.LEFT)

size_label = tk.Label(size_buttons, text=f"{brush_size}px",
                      font=('Arial', 12), bg='#f0f0f0', width=4)
size_label.pack(side=tk.LEFT, padx=5)

tk.Button(size_buttons, text="+", command=increase_brush_size,
          font=('Arial', 12, 'bold'), bg='#f0f0f0', fg='black',
          padx=5, pady=0, relief=tk.RAISED, bd=1, width=2).pack(side=tk.LEFT)

# Actions frame
actions_frame = tk.LabelFrame(drawing_tab, text="Actions", font=('Arial', 10, 'bold'),
                              bg='#f0f0f0', padx=10, pady=5)
actions_frame.pack(side=tk.LEFT, padx=10, pady=5, fill='both')

detect_btn = tk.Button(actions_frame, text="🔍 Detect Text", command=animate_detection,
                       font=('Arial', 11, 'bold'), bg='#4CAF50', fg='white',
                       padx=10, pady=5, relief=tk.RAISED, bd=2, width=12)
detect_btn.pack(side=tk.LEFT, padx=10, pady=5)

speak_btn = tk.Button(actions_frame, text="🔊 Speak Text", command=speak_text,
                      font=('Arial', 11, 'bold'), bg='#2196F3', fg='white',
                      padx=10, pady=5, relief=tk.RAISED, bd=2, width=12)
speak_btn.pack(side=tk.LEFT, padx=10, pady=5)

clear_btn = tk.Button(actions_frame, text="❌ Clear Canvas", command=clear_canvas,
                      font=('Arial', 11, 'bold'), bg='#f44336', fg='white',
                      padx=10, pady=5, relief=tk.RAISED, bd=2, width=12)
clear_btn.pack(side=tk.LEFT, padx=10, pady=5)

# Color presets with scrollbar
color_presets_frame = tk.LabelFrame(drawing_tab, text="Color Presets", font=('Arial', 10, 'bold'),
                                    bg='#f0f0f0', padx=10, pady=5)
color_presets_frame.pack(side=tk.LEFT, padx=10, pady=5, fill='both')

# Create canvas and scrollbar for color presets
color_canvas = tk.Canvas(color_presets_frame, bg='#f0f0f0', highlightthickness=0)
color_scrollbar = ttk.Scrollbar(color_presets_frame, orient="vertical", command=color_canvas.yview)
color_scrollable_frame = tk.Frame(color_canvas, bg='#f0f0f0')

color_scrollable_frame.bind(
    "<Configure>",
    lambda e: color_canvas.configure(
        scrollregion=color_canvas.bbox("all")
    )
)

color_canvas.create_window((0, 0), window=color_scrollable_frame, anchor="nw")
color_canvas.configure(yscrollcommand=color_scrollbar.set)

color_canvas.pack(side="left", fill="both", expand=True)
color_scrollbar.pack(side="right", fill="y")

preset_colors = [
    ("#000000", "Black"), ("#FF0000", "Red"), ("#00FF00", "Green"),
    ("#0000FF", "Blue"), ("#FFFF00", "Yellow"), ("#FF00FF", "Magenta"),
    ("#00FFFF", "Cyan"), ("#FFA500", "Orange"), ("#800080", "Purple"),
    ("#008000", "Dark Green"), ("#800000", "Maroon"), ("#008080", "Teal"),
    ("#FFC0CB", "Pink"), ("#A52A2A", "Brown"), ("#FFD700", "Gold")
]

def set_preset_color(hex_color):
    global brush_color
    brush_color = hex_color

    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    red_scale.set(r)
    green_scale.set(g)
    blue_scale.set(b)

    color_preview.config(bg=brush_color)
    size_preview.delete("all")
    size_preview.create_oval(10 - brush_size / 2, 10 - brush_size / 2,
                             10 + brush_size / 2, 10 + brush_size / 2,
                             fill=brush_color, outline="")
    update_status()

row, col = 0, 0
for hex_color, color_name in preset_colors:
    btn = tk.Button(color_scrollable_frame, bg=hex_color, width=3, height=1,
                    command=lambda c=hex_color: set_preset_color(c))
    btn.grid(row=row, column=col, padx=2, pady=2)
    col += 1
    if col > 2:
        col = 0
        row += 1

# Information tab with scrollbar
info_tab = ttk.Frame(control_notebook)
control_notebook.add(info_tab, text="  Information  ")

# Create canvas and scrollbar for info text
info_canvas = tk.Canvas(info_tab, bg='#f0f0f0', highlightthickness=0)
info_scrollbar = ttk.Scrollbar(info_tab, orient="vertical", command=info_canvas.yview)
info_scrollable_frame = tk.Frame(info_canvas, bg='#f0f0f0')

info_scrollable_frame.bind(
    "<Configure>",
    lambda e: info_canvas.configure(
        scrollregion=info_canvas.bbox("all")
    )
)

info_canvas.create_window((0, 0), window=info_scrollable_frame, anchor="nw")
info_canvas.configure(yscrollcommand=info_scrollbar.set)

info_canvas.pack(side="left", fill="both", expand=True)
info_scrollbar.pack(side="right", fill="y")

info_text = """
Air Canvas Instructions:
1. Point your index finger at the camera to draw
2. Use the color picker sliders to select your preferred color
3. Click the 🎨 button for direct color selection
4. Adjust brush size with + and - buttons
5. Click "Detect Text" to recognize written text
6. Click "Speak Text" to hear the detected text
7. Click "Clear Canvas" to start over

Tips:
- Keep your hand steady for better writing
- Draw large clear letters for better recognition
- Make sure there is good lighting
- Use color presets for quick color changes

Advanced Features:
- The system can detect both uppercase and lowercase letters
- Works best with printed-style letters rather than cursive
- Try different colors for better contrast with the background
- The speech synthesis supports multiple languages (change lang parameter in code)

Troubleshooting:
- If text detection isn't working, try drawing larger letters
- Make sure your hand is well-lit and visible to the camera
- Restart the application if the camera feed freezes
- Ensure Tesseract OCR is properly installed if getting detection errors
"""

info_label = tk.Label(info_scrollable_frame, text=info_text, font=('Arial', 11),
                      justify=tk.LEFT, bg='#f0f0f0', padx=20, pady=10)
info_label.pack(fill='both', expand=True)

# Status bar
status_var = tk.StringVar()
status_var.set("Color: #000000 | Size: 8px | Detected: ")
status_label = tk.Label(root, textvariable=status_var, bd=1, relief=tk.SUNKEN,
                        anchor=tk.W, font=('Arial', 10), bg='white', padx=10)
status_label.place(x=50, y=670, width=1450, height=25)

# Initialize video capture
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def update_video():
    global last_x, last_y

    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger.x * frame_width), int(index_finger.y * frame_height)
                current_x, current_y = map_coordinates(x, y)

                if last_x is not None and last_y is not None:
                    canvas.create_line(last_x, last_y, current_x, current_y,
                                       width=brush_size, fill=brush_color,
                                       capstyle=tk.ROUND, smooth=tk.TRUE)

                last_x, last_y = current_x, current_y
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            last_x, last_y = None, None

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        video_label.img = img
        video_label.config(image=img)

    root.after(10, update_video)

# Initialize and start the application
update_color_from_picker()
update_video()
root.mainloop()

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()