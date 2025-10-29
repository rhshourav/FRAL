import os
import time
import ctypes
import numpy as np
import cv2
import face_recognition
from ultralytics import YOLO

# ==== CONFIGURATION ====
FACES_DIR = "TranningData"
POSE_MODEL = "yolov8n-pose.pt"
CONF_THRESH = 0.5
LOCK_DELAY = 5
FACE_SKIP_FRAMES = 2

# ==== INITIALIZATION ====
print("[*] Loading models...")
pose_model = YOLO(POSE_MODEL)

# ==== LOAD KNOWN FACES ====
known_faces = []
known_names = []

if not os.path.isdir(FACES_DIR):
    raise RuntimeError(f"[!] Folder not found: {FACES_DIR}")

for filename in os.listdir(FACES_DIR):
    path = os.path.join(FACES_DIR, filename)
    try:
        img = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(img)
        if not encs:
            print(f"[!] No face found in {filename}, skipping.")
            continue
        known_faces.append(encs[0])
        known_names.append(os.path.splitext(filename)[0])
        print(f"[+] Loaded {filename}")
    except Exception as e:
        print(f"[!] Error loading {filename}: {e}")

if not known_faces:
    raise RuntimeError("[!] No valid faces loaded. Exiting...")

print(f"[**] Loaded {len(known_faces)} known face(s).")

# ==== CAMERA ====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not found.")

print("[*] Camera started. Press 'q' to quit.")

frame_count = 0
last_seen_known_time = time.time()
current_known_name = None
posture_state = "Unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    rgb_small = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.37, fy=0.37), cv2.COLOR_BGR2RGB)

    detected_known = False  # whether the known person is visible

    # ==== FACE RECOGNITION ====
    if frame_count % FACE_SKIP_FRAMES == 0:
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for enc in face_encodings:
            matches = face_recognition.compare_faces(known_faces, enc)
            face_distances = face_recognition.face_distance(known_faces, enc)
            if face_distances.size > 0:
                best_idx = np.argmin(face_distances)
                if matches[best_idx]:
                    detected_known = True
                    current_known_name = known_names[best_idx]
                    last_seen_known_time = time.time()

    # ==== POSE DETECTION ====
    results = pose_model(frame, verbose=False)
    persons = [r for r in results if len(r.keypoints) > 0]

    if not detected_known:
        # Known person not found — check if timeout passed
        if time.time() - last_seen_known_time > LOCK_DELAY:
            print(f"[🔒] {current_known_name or 'Known user'} left. Locking workstation...")
            ctypes.windll.user32.LockWorkStation()

        status_text = "Known person not detected"
        color = (0, 0, 255)
    else:
        # Known person found; optionally assess posture
        if persons:
            kps = persons[0].keypoints.xy[0]
            if len(kps) >= 13:
                shoulder_y = (kps[5, 1] + kps[6, 1]) / 2
                hip_y = (kps[11, 1] + kps[12, 1]) / 2
                ratio = (hip_y - shoulder_y) / frame.shape[0]
                posture_state = "Standing" if ratio < 0.25 else "Sitting"
            else:
                posture_state = "Unknown"
        status_text = f"{current_known_name} detected ({posture_state})"
        color = (0, 255, 0)

    # ==== DRAW ====
    cv2.putText(frame, status_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Secure Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
