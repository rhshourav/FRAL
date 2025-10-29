import cv2
import face_recognition
import os
import time
import ctypes
import numpy as np

# ===== CONFIG =====
LOCK_DELAY = 3           # seconds to wait before locking after authorized person leaves
AUTHORIZED_NAME = "Shouravv"  # set authorized person
FACES_DIR = "TranningData"
FACE_TOLERANCE = 0.42    # max distance for face match
MIN_CONFIDENCE = 55       # minimum confidence in %

# ===== LOAD FACES =====
person_faces = []
person_names = []

for filename in os.listdir(FACES_DIR):
    img_path = os.path.join(FACES_DIR, filename)
    try:
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            print(f"[!] No face found in {filename}, skipping.")
            continue
        person_faces.append(encodings[0])
        person_names.append(os.path.splitext(filename)[0])
        print(f"[+] Loaded face for {os.path.splitext(filename)[0]}")
    except Exception as e:
        print(f"[!] Error loading {filename}: {e}")

if len(person_faces) == 0:
    print("[!] No faces loaded. Exiting...")
    exit()

# ===== START CAMERA =====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not found.")
print("[*] Camera started. Press 'q' to quit.")

# ===== STATE VARIABLES =====
tracking = False
current_target_encoding = None
last_seen_time = time.time()
last_confidence = 0.0

# ===== MAIN LOOP =====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if not tracking:
        # ===== DETECTION MODE =====
        found_authorized = False
        best_confidence = 0

        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(person_faces, face_encoding)
            if len(face_distances) == 0:
                continue

            best_idx = np.argmin(face_distances)
            distance = face_distances[best_idx]
            confidence = (1 - distance) * 100
            name = person_names[best_idx]

            print(f"[DEBUG] Distance: {distance:.3f}, Confidence: {confidence:.1f}%, Name: {name}")

            if name == AUTHORIZED_NAME and distance <= FACE_TOLERANCE and confidence >= MIN_CONFIDENCE:
                # Authorized person detected, switch to tracking
                tracking = True
                current_target_encoding = face_encoding
                last_seen_time = time.time()
                last_confidence = confidence
                found_authorized = True
                print(f"[+] Authorized person detected: {name}, starting tracking")
                break

        if not found_authorized:
            cv2.putText(frame, f"{AUTHORIZED_NAME} NOT detected",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"{AUTHORIZED_NAME} detected ({last_confidence:.1f}%)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    else:
        # ===== TRACKING MODE =====
        if len(face_encodings) == 0:
            # no faces in frame
            if time.time() - last_seen_time > LOCK_DELAY:
                print("[🔒] Authorized person left. Locking workstation...")
                ctypes.windll.user32.LockWorkStation()
                tracking = False
                current_target_encoding = None
        else:
            # compare only with the current target
            matches = face_recognition.compare_faces([current_target_encoding], face_encodings[0])
            distance = face_recognition.face_distance([current_target_encoding], face_encodings[0])[0]
            confidence = (1 - distance) * 100
            last_confidence = round(confidence, 2)

            if matches[0] and confidence >= MIN_CONFIDENCE:
                last_seen_time = time.time()
                cv2.putText(frame, f"Tracking {AUTHORIZED_NAME} ({last_confidence:.1f}%)",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # not matched
                if time.time() - last_seen_time > LOCK_DELAY:
                    print("[🔒] Authorized person left. Locking workstation...")
                    ctypes.windll.user32.LockWorkStation()
                    tracking = False
                    current_target_encoding = None

    # ===== DRAW ALL FACE BOXES =====
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)

    cv2.imshow("FRAL - Detect & Track", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
