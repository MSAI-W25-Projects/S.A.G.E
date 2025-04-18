import tkinter as tk
from PIL import Image, ImageTk
import cv2
import torch
from torchvision import transforms, models
import numpy as np
import random
import time
import math
import pygame

# -------------------------------
# CONFIGURATION
# -------------------------------
AGE_RANGES = ["0-10", "11-20", "21-30", "31-40", "41-50", "51+"]
GENDER_LIST = ["Male", "Female"]
BUCKET_MIDPOINTS = np.array([5, 15.5, 25.5, 35.5, 45.5, 55])
DIST_THRESH = 50
MAX_MISSING = 5
ALPHA_EMA = 0.3
T_AGE = 1.5
T_GENDER = 1.5

# -------------------------------
# DEVICE & MODEL SETUP
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

age_model = models.resnet18(weights=None)
age_model.fc = torch.nn.Linear(age_model.fc.in_features, len(AGE_RANGES))
age_model.load_state_dict(torch.load("resnet18_utk_agegroup_model.pth", map_location=device))
age_model.to(device).eval()

gender_model = models.resnet18(weights=None)
gender_model.fc = torch.nn.Linear(gender_model.fc.in_features, 2)
gender_model.load_state_dict(torch.load("resnet18_utk_gender_model.pth", map_location=device))
gender_model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

AGE_AUGS = [lambda x: x, lambda x: x.transpose(Image.FLIP_LEFT_RIGHT)]
GENDER_AUGS = AGE_AUGS
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def adjust_gamma(image):
    mean = np.mean(image)/255.0
    gamma = np.log(0.5) / np.log(mean + 1e-6)
    table = np.array([((i/255.0)**(1.0/gamma))*255 for i in range(256)]).astype("uint8")
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

# -------------------------------
# PARTICLE FUNCTIONS
# -------------------------------

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

# -------------------------------
    def animate_prediction_glow(self):
        if hasattr(self, 'current_gender_color'):
            base_color = self.current_gender_color
        else:
            base_color = "00ffee"

        color = f"#{base_color}{int(self.glow_alpha):02x}"
        try:
            self.canvas.itemconfig(self.prediction_label, fill=color)
        except:
            pass
        self.glow_alpha += 5 if self.glow_increasing else -5
        if self.glow_alpha >= 255:
            self.glow_alpha = 255
            self.glow_increasing = False
        elif self.glow_alpha <= 100:
            self.glow_alpha = 100
            self.glow_increasing = True
        self.root.after(60, self.animate_prediction_glow)

# -------------------------------
    def scroll_background(self):
        self.bg_scroll_offset = (self.bg_scroll_offset + 1) % 960
        self.canvas.configure(scrollregion=(self.bg_scroll_offset, 0, self.bg_scroll_offset + 960, 720))
        self.root.after(100, self.scroll_background)

    def animate_energy_ring(self):
        if hasattr(self, 'ring_arc'):
            self.canvas.delete(self.ring_arc)
        angle = int(time.time() * 30) % 360
        self.ring_arc = self.canvas.create_arc(330, 60, 630, 360, start=angle, extent=90, outline="#00ffee", style="arc", width=4)
        self.root.after(100, self.animate_energy_ring)

# -------------------------------
# SAGE UI CLASS
# -------------------------------
class SageUI(ParticleMixin):
    def __init__(self, root):
        self.root = root
        self.root.title("S.A.G.E — Seer of Age and Gender Essence")
        self.root.geometry("960x1000")
        self.root.configure(bg="black")

        pygame.mixer.init()
        pygame.mixer.music.load("assets/intro_sound.mp3")
        pygame.mixer.music.play(-1)
        self.cap = cv2.VideoCapture(0)

        self.bg_scroll_offset = 0
        self.canvas = tk.Canvas(self.root, width=960, height=960, bg="black", highlightthickness=0)
        self.canvas.pack()

        self.header = self.canvas.create_text(480, 30, text="🧙‍♂️ Ask the SAGE",
                                              font=("Papyrus", 28, "bold"), fill="#00ffee")

        sage_img = Image.open("assets/sage-focus.png")
        sage_img = sage_img.resize((300, 300))
        sage_img = Image.open("assets/sage-focus.png")
        sage_img = sage_img.resize((300, 300))
        self.sage_photo = ImageTk.PhotoImage(sage_img)

        self.glow = self.canvas.create_oval(330, 60, 630, 360, fill="#00ffee", outline="", stipple="gray25")
        self.sage_image_item = self.canvas.create_image(480, 210, image=self.sage_photo)
        self.animate_energy_ring()

        self.video_panel = tk.Label(self.canvas, bd=0, bg="black")
        self.video_window = self.canvas.create_window(480, 660, window=self.video_panel, width=720, height=400)

        self.trackers = {}
        self.next_face_id = 0
        self.prev_time = time.time()

        self.particles = self.create_particles(30)
        self.animate_particles()
        self.age_label_bg = self.canvas.create_rectangle(180, 370, 780, 400, fill="#222222", outline="", stipple="gray25")
        self.gender_label_bg = self.canvas.create_rectangle(180, 405, 780, 435, fill="#222222", outline="", stipple="gray25")
        self.age_label = self.canvas.create_text(480, 385, text="", font=("Helvetica", 18, "bold"), fill="#00ffee")
        self.gender_label = self.canvas.create_text(480, 420, text="", font=("Helvetica", 18, "bold"), fill="#ff66cc")
        self.glow_alpha = 0
        self.glow_increasing = True
        self.scroll_background()
        self.animate_prediction_glow()
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
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
            cx, cy = x + w//2, y + h//2
            best_id, best_dist = None, float('inf')
            for fid, tr in self.trackers.items():
                d = math.hypot(cx - tr['centroid'][0], cy - tr['centroid'][1])
                if d < best_dist and d < DIST_THRESH:
                    best_id, best_dist = fid, d

            if best_id is None:
                fid = self.next_face_id
                self.next_face_id += 1
                self.trackers[fid] = {'centroid': (cx, cy), 'missing': 0, 'ema_age': None, 'ema_gender': None}
            else:
                fid = best_id
                self.trackers[fid]['centroid'] = (cx, cy)
                self.trackers[fid]['missing'] = 0
            seen_ids.add(fid)

            face = frame_proc[y:y+h, x:x+w]
            pil_face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

            age_probs = tta_predict(age_model, pil_face, transform, AGE_AUGS, T_AGE)
            self.trackers[fid]['ema_age'] = age_probs if self.trackers[fid]['ema_age'] is None else (
                ALPHA_EMA * age_probs + (1 - ALPHA_EMA) * self.trackers[fid]['ema_age'])
            age_idx = self.trackers[fid]['ema_age'].argmax()
            age_str = AGE_RANGES[age_idx]
            cont_age = (self.trackers[fid]['ema_age'] * BUCKET_MIDPOINTS).sum()
            age_conf = age_probs[age_idx] * 100
            age_label = f"{age_str}  — {age_conf:.1f}%"

            gender_probs = tta_predict(gender_model, pil_face, transform, GENDER_AUGS, T_GENDER)
            self.trackers[fid]['ema_gender'] = gender_probs if self.trackers[fid]['ema_gender'] is None else (
                ALPHA_EMA * gender_probs + (1 - ALPHA_EMA) * self.trackers[fid]['ema_gender'])
            gender_idx = self.trackers[fid]['ema_gender'].argmax()
            gender_conf = gender_probs[gender_idx] * 100
            gender_str = f"{GENDER_LIST[gender_idx]} — {gender_conf:.1f}%"

            icon = "♂" if gender_str == "Male" else "♀"
            label = f"{icon} {gender_str}   {age_label}"
            self.canvas.itemconfig(self.age_label, text=f"🧠 Age: {age_label}")
            self.canvas.itemconfig(self.gender_label, text=f"👤 Gender: {gender_str}")
            
            self.current_gender_color = "0099ff" if gender_str == "Male" else "ff66cc"
            color = (0, 153, 255) if gender_str == "Male" else (255, 102, 204)

                        

        for fid in list(self.trackers):
            if fid not in seen_ids:
                self.trackers[fid]['missing'] += 1
                if self.trackers[fid]['missing'] > MAX_MISSING:
                    del self.trackers[fid]

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
        self.video_panel.imgtk = imgtk
        self.video_panel.configure(image=imgtk)
        # cap.release()  # Kept open for continuous feed
        self.root.after(60, self.update_frame)

# -------------------------------
# LAUNCH
# -------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SageUI(root)
    root.mainloop()
