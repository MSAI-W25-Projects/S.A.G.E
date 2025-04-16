import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import threading
import time
import math

# -------------------------------
# CONFIGURATION
# -------------------------------
CAM_INDEX = 1
AGE_RANGES = ["0-10", "11-20", "21-30", "31-40", "41-50", "51+"]
GENDER_LIST = ["Male","Female" ]
CENTER_DISTANCE_THRESHOLD = 50
MIN_INTERVAL = 3  # seconds between age predictions

# -------------------------------
# DEVICE SETUP
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# AGE MODEL SETUP
# -------------------------------
age_model = models.resnet18(pretrained=False)
age_model.fc = torch.nn.Linear(age_model.fc.in_features, len(AGE_RANGES))
age_model.load_state_dict(torch.load("resnet18_utk_agegroup.pth", map_location=device))
age_model.to(device)
age_model.eval()

age_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# GENDER MODEL SETUP
# -------------------------------
gender_model = models.resnet18(pretrained=False)
gender_model.fc = torch.nn.Linear(gender_model.fc.in_features, 2)
gender_model.load_state_dict(torch.load("resnet18_utk_gender_epoch10.pth", map_location=device))
gender_model.to(device)
gender_model.eval()

gender_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# FACE DETECTOR
# -------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------------------
# AGE THREADING CONTROL
# -------------------------------
global_age_prediction = None
age_prediction_running = False
age_prediction_lock = threading.Lock()
last_face_center = None
last_prediction_time = 0

# -------------------------------
# UTILS
# -------------------------------
def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

def predict_age_async(face_img):
    global global_age_prediction, age_prediction_running
    try:
        pil_face = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        input_tensor = age_transform(pil_face).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = age_model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            predicted_age = AGE_RANGES[pred.item()] if conf.item() > 0.6 else "Unsure"
    except Exception as e:
        print("[ERROR] Age prediction failed:", e)
        predicted_age = "Error"

    with age_prediction_lock:
        global_age_prediction = predicted_age
    age_prediction_running = False

# -------------------------------
# MAIN LOOP
# -------------------------------
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("📷 Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_img = frame[y:y+h, x:x+w].copy()
        current_face_center = (x + w / 2, y + h / 2)

        now = time.time()
        distance = euclidean_distance(current_face_center, last_face_center) if last_face_center else float('inf')

        should_trigger = False
        if last_face_center is None or distance > CENTER_DISTANCE_THRESHOLD:
            if now - last_prediction_time > MIN_INTERVAL:
                should_trigger = True
                last_prediction_time = now
                last_face_center = current_face_center

        if should_trigger and not age_prediction_running:
            age_prediction_running = True
            with age_prediction_lock:
                global_age_prediction = None
            threading.Thread(target=predict_age_async, args=(face_img.copy(),), daemon=True).start()

        # GENDER PREDICTION (EVERY FRAME)
        pil_gender = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        gender_input = gender_transform(pil_gender).unsqueeze(0).to(device)
        with torch.no_grad():
            gender_output = gender_model(gender_input)
            gender_probs = torch.nn.functional.softmax(gender_output, dim=1)
            gender_conf, gender_pred = torch.max(gender_probs, 1)
            gender_label = GENDER_LIST[gender_pred.item()] if gender_conf.item() > 0.6 else "Unsure"

        # AGE PREDICTION FROM THREAD
        with age_prediction_lock:
            age_label = global_age_prediction if global_age_prediction else "Predicting..."

        # DRAW UI
        label = f"Gender: {gender_label}, Age: {age_label}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

    else:
        last_face_center = None  # Reset tracking when no face is visible

    cv2.imshow("Age + Gender Prediction (PyTorch)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()