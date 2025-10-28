from ultralytics import YOLO
import cv2, time, ctypes

# ---- CONFIG ----
LOCK_DELAY = 5          # seconds after no person seen
POSE_MODEL = "yolov8n-pose.pt"   # or yolov8s-pose.pt for better accuracy
CONF_THRESH = 0.5

# ---- LOAD MODEL ----
model = YOLO(POSE_MODEL)
print("[*] Model loaded. Starting camera...")

# ---- INIT CAMERA ----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not found.")

last_seen_time = time.time()
state = "unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    persons = [r for r in results if len(r.keypoints) > 0]

    if len(persons) == 0:
        # No person in frame
        if state != "left":
            print("[!] Person left view.")
            state = "left"
        # Check timer
        if time.time() - last_seen_time > LOCK_DELAY:
            print("[🔒] Locking workstation...")
            ctypes.windll.user32.LockWorkStation()

    else:
        last_seen_time = time.time()
        # Take first detected person
        kps = persons[0].keypoints.xy[0]
        if len(kps) >= 13:  # keypoints: shoulders 5/6, hips 11/12
            shoulder_y = (kps[5,1] + kps[6,1]) / 2
            hip_y = (kps[11,1] + kps[12,1]) / 2
            ratio = (hip_y - shoulder_y) / frame.shape[0]

            if ratio < 0.25:
                posture = "Standing"
            else:
                posture = "Sitting"
        else:
            posture = "Unknown"

        if posture != state:
            print(f"[*] Posture changed: {posture}")
            state = posture

        cv2.putText(frame, posture, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    cv2.imshow("Pose monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
