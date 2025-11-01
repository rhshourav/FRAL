import cv2
import face_recognition
import os
import time
import ctypes
import numpy as np
import threading
from pynput import mouse
from ultralytics import YOLO

# === Configuration ===
POSE_MODEL = "yolov8n-pose.pt"
LOCK_DELAY = 3
CONF_THRESH = 0.5
ADAPTIVE_TOLERANCE = 0.42
MIN_CONFIDENCE = 55
faces_DIR = "TranningData"
MOUSE_INACTIVITY_LIMIT = 10  # seconds before enabling camera

# === Global States ===
last_mouse_move = time.time()
lock_active = False
camera_enabled = False

# === Mouse inactivity tracker ===
def on_move(x, y):
    global last_mouse_move, lock_active, camera_enabled
    last_mouse_move = time.time()

    if camera_enabled:
        print("[INFO] Mouse moved — pausing camera tracking.")
        camera_enabled = False
    if lock_active:
        print("[INFO] Device unlocked and mouse moved — resetting system.")
        lock_active = False

def monitor_mouse():
    with mouse.Listener(on_move=on_move) as listener:
        listener.join()

threading.Thread(target=monitor_mouse, daemon=True).start()

# === Load known faces ===
person_Face, person_Name = [], []
for filename in os.listdir(faces_DIR):
    img_path = os.path.join(faces_DIR, filename)
    try:
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            print(f"No face found in {filename}, skipping.")
            continue
        person_Face.append(encodings[0])
        person_Name.append(os.path.splitext(filename)[0])
        print(f"Loaded face for {os.path.splitext(filename)[0]}")
    except Exception as e:
        print(f"Error loading {filename}: {e}")

if not person_Face:
    print("No faces loaded. Exiting...")
    exit()

# === Optional Pose Model ===
model = YOLO(POSE_MODEL)

cv2.setNumThreads(1)  # Prevent multithread tracker issues
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not found.")

state = "idle"  # idle → searching → tracking
tracker = None
tracking_name = None
last_seen_time = time.time()
person_present = False

# === Helpers ===
def create_tracker():
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    elif hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    elif hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    else:
        raise RuntimeError("No compatible tracker found.")

def normalize_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def face_location_to_bbox(face_loc, frame_shape):
    top, right, bottom, left = map(int, face_loc)
    h, w, _ = frame_shape
    left = max(0, min(left, w - 1))
    top = max(0, min(top, h - 1))
    right = max(left + 2, min(right, w - 1))
    bottom = max(top + 2, min(bottom, h - 1))
    width = right - left
    height = bottom - top
    if width < 10 or height < 10:
        width, height = 50, 50  # ensure valid bbox
    return (left, top, width, height)

def bbox_posture(bbox, frame_h):
    _, _, _, h = bbox
    return "Standing" if h / float(frame_h) > 0.35 else "Sitting"

print("System ready. Waiting for mouse inactivity... (Move mouse to pause)")

# === Main Loop ===
while True:
    inactive_time = time.time() - last_mouse_move

    # --- Mouse inactive: activate camera
    if inactive_time >= MOUSE_INACTIVITY_LIMIT:
        if not camera_enabled:
            print("[INFO] Mouse inactive for 10s — activating camera for face check.")
            camera_enabled = True
            state = "searching"
    else:
        if camera_enabled:
            print("[INFO] Mouse moved — deactivating camera.")
            camera_enabled = False
            state = "idle"
        paused_frame = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(paused_frame, "Mouse active - paused", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("FRAL Face+Pose Tracking", paused_frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        continue

    # --- Camera active
    ret, frame = cap.read()
    if not ret:
        break

    normalized_frame = normalize_frame(frame)
    rgb_frame = cv2.cvtColor(normalized_frame, cv2.COLOR_BGR2RGB)

    if state == "searching":
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        best_confidence, best_bbox, best_name = 0, None, None

        for (face_location, face_encoding) in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(person_Face, face_encoding)
            if not len(distances):
                continue
            idx = np.argmin(distances)
            distance = float(distances[idx])
            confidence = (1 - distance) * 100
            name = person_Name[idx]
            if distance <= ADAPTIVE_TOLERANCE and confidence >= MIN_CONFIDENCE:
                if confidence > best_confidence:
                    best_confidence, best_bbox, best_name = confidence, face_location, name

        if best_confidence > 0 and best_bbox is not None:
            x, y, w, h = face_location_to_bbox(best_bbox, frame.shape)
            print(f"[DEBUG] Initializing tracker with bbox: x={x}, y={y}, w={w}, h={h}, frame={frame.shape}")
            if w <= 0 or h <= 0:
                print("[ERROR] Invalid bbox — skipping tracker init.")
                continue

            tracker = create_tracker()
            ok = tracker.init(frame, (x, y, w, h))
            if ok:
                print(f"[+] Authorized face recognized ({best_name}, {best_confidence:.1f}%). Starting tracking.")
                tracking_name = best_name
                state = "tracking"
                last_seen_time = time.time()
                person_present = True
            else:
                print("❌ Tracker init failed.")
        else:
            cv2.putText(frame, "No authorized face found", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)

    elif state == "tracking":
        ok, bbox = tracker.update(frame)
        if ok:
            last_seen_time = time.time()
            person_present = True
            x, y, w, h = map(int, bbox)
            posture = bbox_posture((x, y, w, h), frame.shape[0])

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
            cv2.putText(frame, f"{tracking_name} - {posture}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

            if posture == "Standing" and (x < 30 or (x + w) > frame.shape[1] - 30):
                print("[🔒] Person leaving — locking workstation...")
                ctypes.windll.user32.LockWorkStation()
                lock_active = True
                camera_enabled = False
                state = "idle"
                tracker = None
                tracking_name = None
                person_present = False
                time.sleep(1)
                continue
        else:
            cv2.putText(frame, "Tracking lost", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)
            if person_present and (time.time() - last_seen_time > LOCK_DELAY):
                print("[🔒] Tracking lost — locking workstation...")
                ctypes.windll.user32.LockWorkStation()
                lock_active = True
                camera_enabled = False
                state = "idle"
                tracker = None
                tracking_name = None
                person_present = False

    cv2.imshow("FRAL Face+Pose Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
