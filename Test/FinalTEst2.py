import os
import time
import ctypes
import numpy as np
import cv2
import face_recognition
from ultralytics import YOLO

# ==== CONFIG ====
FACES_DIR = "../TranningData"  # folder with known face(s)
POSE_MODEL = "yolov8n-pose.pt"       # lightweight pose model
LOCK_DELAY = 5                       # seconds after known person leaves
FACE_SKIP_FRAMES = 2                 # process face every N frames

# ==== LOAD MODELS ====
print("[*] Loading models...")
pose_model = YOLO(POSE_MODEL)

# ==== LOAD KNOWN FACES ====
known_encodings = []
known_names = []

if not os.path.isdir(FACES_DIR):
    raise RuntimeError(f"[!] Folder not found: {FACES_DIR}")

for filename in os.listdir(FACES_DIR):
    path = os.path.join(FACES_DIR, filename)
    try:
        img = face_recognition.load_image_file(path)
        enc = face_recognition.face_encodings(img)
        if len(enc) == 0:
            print(f"[!] No face found in {filename}, skipping.")
            continue
        known_encodings.append(enc[0])
        known_names.append(os.path.splitext(filename)[0])
        print(f"[+] Loaded {filename}")
    except Exception as e:
        print(f"[!] Error loading {filename}: {e}")

if len(known_encodings) == 0:
    print("[!] No known faces found. Exiting...")
    exit()

print(f"[**] Total known persons: {len(known_encodings)}")

# ==== START CAMERA ====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not found.")

print("[*] Camera started. Press 'q' to quit.")

frame_count = 0
last_seen_time = time.time()
known_visible = False
posture_state = "Unknown"
user_name = known_names[0]  # main authorized user (first trained person)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[!] Failed to grab frame")
        break

    frame_count += 1
    rgb_small = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.37, fy=0.37), cv2.COLOR_BGR2RGB)

    # ==== FACE RECOGNITION ====
    if frame_count % FACE_SKIP_FRAMES == 0:
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        found_known = False
        for enc in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, enc)
            face_distances = face_recognition.face_distance(known_encodings, enc)
            if len(face_distances) > 0:
                best_idx = np.argmin(face_distances)
                if matches[best_idx]:
                    found_known = True
                    last_seen_time = time.time()
                    break

        known_visible = found_known

    # ==== POSE DETECTION ====
    results = pose_model(frame, verbose=False)
    persons = [r for r in results if len(r.keypoints) > 0]

    if known_visible:
        # Extract posture of the *first* person (assume it's the known one)
        if len(persons) > 0:
            kps = persons[0].keypoints.xy[0]
            if len(kps) >= 13:
                shoulder_y = (kps[5, 1] + kps[6, 1]) / 2
                hip_y = (kps[11, 1] + kps[12, 1]) / 2
                ratio = (hip_y - shoulder_y) / frame.shape[0]
                posture_state = "Standing" if ratio < 0.25 else "Sitting"
            else:
                posture_state = "Unknown"
        else:
            posture_state = "Unknown"

        cv2.putText(frame, f"{user_name} detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Posture: {posture_state}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    else:
        cv2.putText(frame, f"{user_name} not visible", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # ==== LOCK CONDITIONS ====
    # (1) If known person not visible for LOCK_DELAY seconds
    # (2) Or if person was standing and left the frame
    if (not known_visible and time.time() - last_seen_time > LOCK_DELAY) or \
       (known_visible and posture_state == "Standing" and time.time() - last_seen_time > LOCK_DELAY):
        print("[🔒] Locking workstation - known person left or stood up.")
        ctypes.windll.user32.LockWorkStation()


    cv2.imshow("Smart Lock System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
