# merged_face_pose_lock.py
import os
import time
import ctypes
import numpy as np
import cv2
import face_recognition
from ultralytics import YOLO

# ==== CONFIG ====
FACES_DIR = "TranningData"           # folder with known faces
POSE_MODEL = "yolov8n-pose.pt"       # pose model
CONF_THRESH = 0.5
LOCK_DELAY = 5                       # seconds after known person leaves
FACE_SKIP_FRAMES = 2                 # run face recognition every N frames

# ==== LOAD MODELS ====
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
        if len(encs) == 0:
            print(f"[!] No face found in {filename}, skipping.")
            continue
        known_faces.append(encs[0])
        known_names.append(os.path.splitext(filename)[0])
        print(f"[+] Loaded {filename}")
    except Exception as e:
        print(f"[!] Error loading {filename}: {e}")

print(f"[**] Total known faces: {len(known_faces)}")
if len(known_faces) == 0:
    print("[!] No known faces loaded. Exiting...")
    exit()

# ==== START CAMERA ====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not found.")

print("[*] Camera started. Press 'q' to quit.")

frame_count = 0
last_seen_known_time = time.time()
current_name = "Unknown"
posture_state = "Unknown"

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

        if len(face_encodings) > 0:
            match_name = "Unknown"
            for enc in face_encodings:
                matches = face_recognition.compare_faces(known_faces, enc)
                face_distances = face_recognition.face_distance(known_faces, enc)
                if len(face_distances) > 0:
                    best_idx = np.argmin(face_distances)
                    if matches[best_idx]:
                        match_name = known_names[best_idx]
                        last_seen_known_time = time.time()
            current_name = match_name
        else:
            current_name = "No face"

    # ==== POSE DETECTION ====
    results = pose_model(frame, verbose=False)
    persons = [r for r in results if len(r.keypoints) > 0]

    if len(persons) == 0:
        cv2.putText(frame, "No person detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Lock if timeout exceeded
        if time.time() - last_seen_known_time > LOCK_DELAY:
            print("[🔒] Locking workstation (no known person)...")
            ctypes.windll.user32.LockWorkStation()
            
    else:
        # Get posture
        kps = persons[0].keypoints.xy[0]
        if len(kps) >= 13:
            shoulder_y = (kps[5, 1] + kps[6, 1]) / 2
            hip_y = (kps[11, 1] + kps[12, 1]) / 2
            ratio = (hip_y - shoulder_y) / frame.shape[0]
            posture_state = "Standing" if ratio < 0.25 else "Sitting"
        else:
            posture_state = "Unknown"

    # ==== DRAW OVERLAY ====
    cv2.putText(frame, f"User: {current_name}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if current_name != "Unknown" else (0, 0, 255), 2)
    cv2.putText(frame, f"Posture: {posture_state}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow("Smart Lock Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
