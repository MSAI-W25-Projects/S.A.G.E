import tkinter as tk
from PIL import Image, ImageTk, ImageEnhance
import cv2
import torch
from torchvision import transforms, models
import numpy as np
import random
import time
import math
import pygame
from collections import deque, Counter
from keras.models import load_model
import os
import sys

# -------------------------------
# CONFIGURATION
# -------------------------------
AGE_RANGES = ["0-10", "11-20", "21-30", "31-40", "41-50", "51+"]
GENDER_LIST = ["Male", "Female"]
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
BUCKET_MIDPOINTS = np.array([5, 15.5, 25.5, 35.5, 45.5, 55])
DIST_THRESH = 50
MAX_MISSING = 5
ALPHA_EMA = 0.1
T_AGE = 1.5
T_GENDER = 1.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller .exe """
    try:
        base_path = sys._MEIPASS  # Used by PyInstaller
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Load age model
age_model = models.resnet18(weights=None)
age_model.fc = torch.nn.Linear(age_model.fc.in_features, len(AGE_RANGES))
age_model.load_state_dict(torch.load(resource_path("resnet18_utk_agegroup_model.pth"), map_location=device))
age_model.to(device).eval()

# Load gender model
gender_model = models.resnet18(weights=None)
gender_model.fc = torch.nn.Linear(gender_model.fc.in_features, 2)
gender_model.load_state_dict(torch.load(resource_path("resnet18_utk_gender_model.pth"), map_location=device))
gender_model.to(device).eval()

# Load emotion model
emotion_model = load_model(resource_path("assets/mini_xception_weights.h5"),compile=False)

def preprocess_emotion_face(face_bgr):
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (64, 64))  
    face_normalized = face_resized.astype("float32") / 255.0
    face_reshaped = np.expand_dims(face_normalized, axis=(0, -1))  
    return face_reshaped

def predict_emotion(face_np):
    try:
        preprocessed = preprocess_emotion_face(face_np)
        preds = emotion_model.predict(preprocessed, verbose=0)
        return EMOTION_LABELS[np.argmax(preds)]
    except Exception as e:
        print("[ERROR] Emotion prediction failed:", e)
        return "Unknown"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def brightness_aug(img): return ImageEnhance.Brightness(img).enhance(1.2)
def contrast_aug(img): return ImageEnhance.Contrast(img).enhance(0.9)
AGE_AUGS = [lambda x: x, lambda x: x.transpose(Image.FLIP_LEFT_RIGHT), brightness_aug, contrast_aug]
GENDER_AUGS = AGE_AUGS

face_cascade = cv2.CascadeClassifier(resource_path("assets/haarcascade_frontalface_default.xml"))

def adjust_gamma(image):
    mean = np.mean(image) / 255.0
    gamma = np.log(0.5) / np.log(mean + 1e-6)
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def tta_predict(model, pil_img, transform, aug_fns, temp):
    logits = []
    for aug in aug_fns:
        img_aug = aug(pil_img)
        input_tensor = transform(img_aug).unsqueeze(0).to(device)
        with torch.no_grad():
            logits.append(model(input_tensor).squeeze(0))
    avg = torch.stack(logits).mean(0) / temp
    return torch.nn.functional.softmax(avg, dim=0).cpu().numpy()

class ParticleMixin:
    def create_particles(self, count):
        particles = []
        for _ in range(count):
            x, y = random.randint(0, 960), random.randint(0, 720)
            dx, dy = random.choice([-1, 1]), random.choice([-1, 1])
            p = self.canvas.create_oval(x, y, x + 5, y + 5, fill="white", outline="")
            particles.append((p, dx, dy))
        return particles

    def animate_particles(self):
        for i, (p, dx, dy) in enumerate(self.particles):
            x0, y0, x1, y1 = self.canvas.coords(p)
            if x0 <= 0 or x1 >= 960:
                dx *= -1
            if y0 <= 0 or y1 >= 720:
                dy *= -1
            self.canvas.move(p, dx, dy)
            self.particles[i] = (p, dx, dy)
        self.root.after(50, self.animate_particles)

    def animate_energy_ring(self):
        if hasattr(self, 'ring_arc'):
            self.canvas.delete(self.ring_arc)
        angle = int(time.time() * 30) % 360
        self.ring_arc = self.canvas.create_arc(330, 60, 630, 360, start=angle, extent=90, outline="#00ffee", style="arc", width=4)
        self.root.after(100, self.animate_energy_ring)

class SageUI(ParticleMixin):
    def __init__(self, root):
        self.root = root
        self.root.title("S.A.G.E — Seer of Age and Gender Essence")
        self.root.geometry("960x1000")
        self.root.configure(bg="black")

        pygame.mixer.init()
        pygame.mixer.music.load(resource_path("assets/intro_sound.mp3"))
        pygame.mixer.music.play(-1)
        self.cap = cv2.VideoCapture(0)

        self.canvas = tk.Canvas(self.root, width=960, height=480, bg="black", highlightthickness=0)
        self.canvas.pack(pady=(10, 0))

        self.header = self.canvas.create_text(480, 30, text="🧙‍♂️ Ask the SAGE", font=("Papyrus", 28, "bold"), fill="#00ffee")
        sage_img = Image.open(resource_path("assets/sage-focus.png")).resize((300, 300))
        self.sage_photo = ImageTk.PhotoImage(sage_img)

        self.glow = self.canvas.create_oval(330, 60, 630, 360, fill="#00ffee", outline="", stipple="gray25")
        self.sage_image_item = self.canvas.create_image(480, 210, image=self.sage_photo)
        self.animate_energy_ring()

        self.video_panel = tk.Label(self.root, bd=0, bg="black")
        self.video_panel.place(relx=0.5, rely=0.72, anchor='center', width=720, height=400)

        self.age_label = tk.Label(self.video_panel, text="", font=("Helvetica", 14, "bold"),
                                  fg="#00ffee", bg="#000000", anchor="w")
        self.age_label.place(relx=0.02, rely=0.02)

        self.gender_label = tk.Label(self.video_panel, text="", font=("Helvetica", 14, "bold"),
                                     fg="#ff66cc", bg="#000000", anchor="w")
        self.gender_label.place(relx=0.02, rely=0.10)

        self.emotion_label = tk.Label(self.video_panel, text="", font=("Helvetica", 14, "bold"),
                                      fg="#ffa500", bg="#000000", anchor="w")
        self.emotion_label.place(relx=0.02, rely=0.18)

        self.trackers = {}
        self.next_face_id = 0
        self.prev_time = time.time()

        self.particles = self.create_particles(30)
        self.animate_particles()
        self.update_frame()

    def update_frame(self):
        if hasattr(self, 'last_process_time') and time.time() - self.last_process_time < 0.1:
            self.root.after(10, self.update_frame)
            return
        self.last_process_time = time.time()

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.root.after(100, self.update_frame)
            return

        now = time.time()
        fps = 1.0 / (now - self.prev_time)
        self.prev_time = now

        frame_proc = adjust_gamma(frame)
        gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        seen_ids = set()

        for (x, y, w, h) in faces:
            cx, cy = x + w // 2, y + h // 2
            best_id, best_dist = None, float('inf')
            for fid, tr in self.trackers.items():
                d = math.hypot(cx - tr['centroid'][0], cy - tr['centroid'][1])
                if d < best_dist and d < DIST_THRESH:
                    best_id, best_dist = fid, d

            fid = best_id if best_id is not None else self.next_face_id
            if fid == self.next_face_id:
                self.next_face_id += 1
                self.trackers[fid] = {'centroid': (cx, cy), 'missing': 0, 'ema_age': None, 'ema_gender': None}

            self.trackers[fid]['centroid'] = (cx, cy)
            self.trackers[fid]['missing'] = 0
            seen_ids.add(fid)

            face = frame_proc[y:y + h, x:x + w]
            pil_face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            age_probs = tta_predict(age_model, pil_face, transform, AGE_AUGS, T_AGE)
            age_idx = age_probs.argmax()

            if 'age_history' not in self.trackers[fid]:
                self.trackers[fid]['age_history'] = deque(maxlen=10)
                self.trackers[fid]['cont_age'] = None

            if age_probs[age_idx] >= 0.5:
                self.trackers[fid]['ema_age'] = age_probs if self.trackers[fid]['ema_age'] is None else (
                    ALPHA_EMA * age_probs + (1 - ALPHA_EMA) * self.trackers[fid]['ema_age'])
                self.trackers[fid]['age_history'].append(age_idx)
                cont_age = (self.trackers[fid]['ema_age'] * BUCKET_MIDPOINTS).sum()
                self.trackers[fid]['cont_age'] = cont_age if self.trackers[fid]['cont_age'] is None else (
                    ALPHA_EMA * cont_age + (1 - ALPHA_EMA) * self.trackers[fid]['cont_age'])
            else:
                continue

            gender_probs = tta_predict(gender_model, pil_face, transform, GENDER_AUGS, T_GENDER)
            gender_idx = gender_probs.argmax()
            gender_conf = gender_probs[gender_idx] * 100
            gender_str = f"{GENDER_LIST[gender_idx]} — {gender_conf:.1f}%"
            self.current_gender_color = "0099ff" if gender_idx == 0 else "ff66cc"

            age_mode = Counter(self.trackers[fid]['age_history']).most_common(1)[0][0]
            age_str = AGE_RANGES[age_mode]
            age_conf = age_probs[age_mode] * 100
            smoothed_cont_age = int(self.trackers[fid]['cont_age']) if self.trackers[fid]['cont_age'] is not None else "?"
            age_label = f"{age_str}  — {age_conf:.1f}% ({smoothed_cont_age} yrs est.)"

            self.age_label.config(text=f"🧠 Age: {age_label}")
            self.gender_label.config(text=f"👤 Gender: {gender_str}")
            emotion_str = predict_emotion(face)
            self.emotion_label.config(text=f"🎭 Emotion: {emotion_str}")

        for fid in list(self.trackers):
            if fid not in seen_ids:
                self.trackers[fid]['missing'] += 1
                if self.trackers[fid]['missing'] > MAX_MISSING:
                    del self.trackers[fid]

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(img)
        self.video_panel.configure(image=photo)
        self.video_panel.image = photo
        self.root.after(60, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = SageUI(root)
    root.mainloop()
