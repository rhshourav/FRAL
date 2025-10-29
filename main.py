import cv2
import face_recognition
import os
import time
import ctypes
import numpy as np
from ultralytics import YOLO

# === Configuration ===
POSE_MODEL = "yolov8n-pose.pt"  # Optional pose model
LOCK_DELAY = 3                  # seconds before lock after loss
CONF_THRESH = 0.5
ADAPTIVE_TOLERANCE = 0.42
MIN_CONFIDENCE = 55
faces_DIR = "TranningData"

# === Load known faces ===
person_Face = []
person_Name = []

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

if len(person_Face) == 0:
    print("No faces loaded. Exiting...")
    exit()

# Optional YOLO model (used later for pose if you want)
model = YOLO(POSE_MODEL)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not found.")

state = "searching"   # 'searching' or 'tracking'
tracker = None
tracking_name = None
last_seen_time = time.time()
person_present = False

# === Tracker Creation Helper ===
def create_tracker():
    """Create a compatible tracker regardless of OpenCV version."""
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        print("[INFO] Using cv2.legacy.TrackerCSRT_create()")
        return cv2.legacy.TrackerCSRT_create()
    elif hasattr(cv2, "TrackerCSRT_create"):
        print("[INFO] Using cv2.TrackerCSRT_create()")
        return cv2.TrackerCSRT_create()
    elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        print("[INFO] Using cv2.legacy.TrackerKCF_create()")
        return cv2.legacy.TrackerKCF_create()
    elif hasattr(cv2, "TrackerKCF_create"):
        print("[INFO] Using cv2.TrackerKCF_create()")
        return cv2.TrackerKCF_create()
    else:
        raise RuntimeError("No compatible tracker found. Install opencv-contrib-python.")

# === Helper functions ===
def normalize_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    normalized = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return normalized

def face_location_to_bbox(face_loc, frame_shape):
    # face_loc = (top, right, bottom, left)
    top, right, bottom, left = face_loc
    h, w, _ = frame_shape

    # Clamp to valid region
    top = max(0, top)
    left = max(0, left)
    right = min(w - 1, right)
    bottom = min(h - 1, bottom)

    bbox_w = max(1, right - left)
    bbox_h = max(1, bottom - top)

    return (left, top, bbox_w, bbox_h)

def bbox_posture(bbox, frame_h):
    # Simple posture heuristic
    _, _, _, h = bbox
    ratio = h / float(frame_h)
    if ratio > 0.35:
        return "Standing"
    else:
        return "Sitting"

print("Camera started. Press 'q' to quit.")

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    normalized_frame = normalize_frame(frame)
    rgb_frame = cv2.cvtColor(normalized_frame, cv2.COLOR_BGR2RGB)

    if state == "searching":
        # Face recognition phase
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        best_confidence = 0
        best_bbox = None
        best_name = None

        for (face_location, face_encoding) in zip(face_locations, face_encodings):
            face_distances = face_recognition.face_distance(person_Face, face_encoding)
            if len(face_distances) == 0:
                continue
            idx = np.argmin(face_distances)
            distance = float(face_distances[idx])
            confidence = (1 - distance) * 100
            name = person_Name[idx]
            if distance <= ADAPTIVE_TOLERANCE and confidence >= MIN_CONFIDENCE:
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_bbox = face_location
                    best_name = name

        if best_confidence > 0 and best_bbox is not None:
            # Convert and clamp bbox safely
            x, y, w, h = face_location_to_bbox(best_bbox, frame.shape)
            print(f"[DEBUG] Tracker bbox: ({x},{y},{w},{h}) Frame: {frame.shape}")

            if w > 0 and h > 0:
                tracker = create_tracker()
                ok = tracker.init(frame, (x, y, w, h))
                if ok:
                    state = "tracking"
                    tracking_name = best_name
                    last_seen_time = time.time()
                    person_present = True
                    print(f"[+] Found {tracking_name} ({best_confidence:.1f}%). Starting tracker.")
                    cv2.putText(frame, f"Tracking {tracking_name} ({best_confidence:.1f}%)", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                else:
                    print("❌ Tracker.init() still failed — OpenCV may need contrib build.")
            else:
                print("❌ Invalid bbox size, skipping tracker init.")
        else:
            cv2.putText(frame, "No authorized face found", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)

    elif state == "tracking":
        ok, bbox = tracker.update(frame)
        if ok:
            last_seen_time = time.time()
            person_present = True
            x, y, w, h = map(int, bbox)
            cx = x + w // 2
            cy = y + h // 2

            # Draw tracker box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{tracking_name}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # Posture detection
            posture = bbox_posture((x, y, w, h), frame.shape[0])
            cv2.putText(frame, f"{posture}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            # If standing and near border — lock
            if posture == "Standing" and (x < 30 or (x + w) > frame.shape[1] - 30):
                print("[🔒] Person stood up and is leaving. Locking workstation...")
                ctypes.windll.user32.LockWorkStation()
                state = "searching"
                tracker = None
                tracking_name = None
                person_present = False
                time.sleep(0.5)
                continue
        else:
            cv2.putText(frame, f"{tracking_name}: Tracking lost", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)

            # Lock if missing too long
            if person_present and (time.time() - last_seen_time > LOCK_DELAY):
                print("[🔒] Lost tracking for too long. Locking workstation...")
                ctypes.windll.user32.LockWorkStation()
                state = "searching"
                tracker = None
                tracking_name = None
                person_present = False

    cv2.imshow("FRAL Face+Pose Tracking (Fixed Tracker)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
