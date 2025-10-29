# merged_face_pose_lock_tracked.py
import os
import time
import ctypes
import numpy as np
import cv2
import face_recognition
from ultralytics import YOLO

# ===== CONFIG =====
FACES_DIR = "TranningData"        # folder with authorized face images (one file per person)
POSE_MODEL = "yolov8n-pose.pt"
LOCK_DELAY = 4                    # seconds after known person stood up & is no longer visible -> lock
FACE_SKIP_FRAMES = 2              # run face recognition every N frames (speed)
IMG_SCALE = 0.37                  # resizing factor used for face recognition (must match usage below)

# ===== LOAD MODELS =====
print("[*] Loading models...")
pose_model = YOLO(POSE_MODEL)

# ===== LOAD KNOWN FACE(S) =====
known_encodings = []
known_names = []

if not os.path.isdir(FACES_DIR):
    raise RuntimeError(f"Faces folder not found: {FACES_DIR}")

for fn in os.listdir(FACES_DIR):
    path = os.path.join(FACES_DIR, fn)
    try:
        img = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(img)
        if len(encs) == 0:
            print(f"[!] No face found in {fn}, skipping.")
            continue
        known_encodings.append(encs[0])
        known_names.append(os.path.splitext(fn)[0])
        print(f"[+] Loaded {fn}")
    except Exception as e:
        print(f"[!] Error loading {fn}: {e}")

if len(known_encodings) == 0:
    print("[!] No known faces loaded. Exiting.")
    exit()

# Choose the primary authorized user (first one)
authorized_name = known_names[0]
print(f"[*] Authorized (tracked) user: {authorized_name}")

# ===== START CAMERA =====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not found.")

frame_count = 0
last_seen_known_face_time = 0.0   # last time we actually saw the known face (via face_recognition)
known_face_currently_visible = False

# Tracking state about the matched person (by comparing face center <-> person keypoints center)
tracked_person_id = None         # index in persons list of the matched person in the current frame
tracked_posture = "Unknown"      # "Sitting" or "Standing" for the tracked person
last_posture_change_time = time.time()
standing_started_time = None     # when posture became standing (if it did)

print("[*] Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[!] Failed to grab frame")
        break

    frame_count += 1
    # small frame used for face_recognition (faster)
    small_frame = cv2.resize(frame, (0, 0), fx=IMG_SCALE, fy=IMG_SCALE)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # ----- FACE RECOGNITION (periodic) -----
    face_centers = []     # centers of all faces found in small_frame, scaled to original frame
    face_names = []
    if frame_count % FACE_SKIP_FRAMES == 0:
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for loc, enc in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, enc)
            face_distances = face_recognition.face_distance(known_encodings, enc)
            name = "Unknown"
            if len(face_distances) > 0:
                best_idx = np.argmin(face_distances)
                if matches[best_idx]:
                    name = known_names[best_idx]

            # compute center in original frame coordinates
            top, right, bottom, left = loc
            # scale back because loc is from small_frame
            scale = 1.0 / IMG_SCALE
            cx = ((left + right) / 2.0) * scale
            cy = ((top + bottom) / 2.0) * scale
            face_centers.append((cx, cy))
            face_names.append(name)

            if name == authorized_name:
                known_face_currently_visible = True
                last_seen_known_face_time = time.time()

        # if no face encodings found in this cycle:
        if len(face_encodings) == 0:
            known_face_currently_visible = False

    # ----- POSE DETECTION (every frame) -----
    results = pose_model(frame, verbose=False)   # returns list of results for detections
    # Build a list of persons and their keypoints centers
    persons = []
    for r in results:
        # r.keypoints.xy is an array (N_people x K x 2), or r.keypoints.xy[0] for first person
        try:
            kps = r.keypoints.xy[0]    # shape (K,2)
            if kps is None or len(kps) == 0:
                continue
            # compute center of keypoints as representative position
            center_x = float(np.mean(kps[:, 0]))
            center_y = float(np.mean(kps[:, 1]))
            persons.append({
                "kps": kps,
                "center": (center_x, center_y),
                "raw": r
            })
        except Exception:
            continue

    # If we have a known face visible, attempt to match it to one of the detected persons
    matched_index = None
    if known_face_currently_visible and len(face_centers) > 0 and len(persons) > 0:
        # find index of the authorized face center in face_centers
        auth_face_indexes = [i for i, n in enumerate(face_names) if n == authorized_name]
        if len(auth_face_indexes) > 0:
            # if multiple faces match authorized_name (unlikely), just take first
            auth_idx = auth_face_indexes[0]
            fx, fy = face_centers[auth_idx]

            # find the person whose keypoints center is closest to the face center
            min_dist = float("inf")
            min_i = None
            for i, p in enumerate(persons):
                px, py = p["center"]
                d = (px - fx) ** 2 + (py - fy) ** 2
                if d < min_dist:
                    min_dist = d
                    min_i = i
            # threshold: accept match if distance not too big (empiric)
            if min_i is not None and min_dist < (frame.shape[1] * 0.25) ** 2:
                matched_index = min_i
                tracked_person_id = matched_index
            else:
                tracked_person_id = None
        else:
            tracked_person_id = None
    else:
        tracked_person_id = None

    # Determine posture of the tracked person (if any)
    previous_posture = tracked_posture
    if tracked_person_id is not None and tracked_person_id < len(persons):
        kps = persons[tracked_person_id]["kps"]
        # Ensure we have shoulder and hip keypoints indices matched to the dataset used by YOLOv8 pose:
        # earlier code used indices 5/6 for shoulders and 11/12 for hips — keep same assumption
        if len(kps) >= 13:
            shoulder_y = (kps[5, 1] + kps[6, 1]) / 2.0
            hip_y = (kps[11, 1] + kps[12, 1]) / 2.0
            ratio = (hip_y - shoulder_y) / frame.shape[0]
            tracked_posture = "Standing" if ratio < 0.25 else "Sitting"
        else:
            tracked_posture = "Unknown"
    else:
        tracked_posture = "Unknown"

    # Detect posture change time
    if tracked_posture != previous_posture:
        last_posture_change_time = time.time()
        if tracked_posture == "Standing":
            standing_started_time = time.time()
        else:
            standing_started_time = None

    # Draw overlays
    status_text = f"Tracked: {authorized_name if tracked_person_id is not None else 'None'}"
    cv2.putText(frame, status_text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if tracked_person_id is not None else (0, 0, 255), 2)
    cv2.putText(frame, f"Posture: {tracked_posture}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # Optionally draw matched person's keypoints rectangle / center
    if tracked_person_id is not None:
        cx, cy = persons[tracked_person_id]["center"]
        cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 0), -1)

    # ----- LOCK LOGIC -----
    # We only lock when:
    # 1) The authorized user was last seen, and then the user stood up (tracked_posture == "Standing")
    #    and after that the authorized user's face is no longer visible for LOCK_DELAY seconds.
    #    (This indicates they stood up and left.)
    #
    # 2) Or if authorized face disappears and we already saw them standing before leaving,
    #    and it has been more than LOCK_DELAY seconds since last seen.
    #
    # Important: other people in frame are ignored entirely.
    now = time.time()

    # Condition helper: did we recently see the known face?
    seen_recently = (now - last_seen_known_face_time) <= LOCK_DELAY

    should_lock = False

    # if tracked_posture was standing recently (we saw standing) and now the face is gone long enough -> lock
    if (tracked_posture == "Standing" or (standing_started_time is not None and (now - standing_started_time) < (LOCK_DELAY + 1))) :
        # If known face not visible and not seen recently -> lock
        if not known_face_currently_visible and (now - last_seen_known_face_time) > LOCK_DELAY:
            should_lock = True

    # Also, if we had known_face visible and then it disappeared without tracked person present and enough time passed:
    if not known_face_currently_visible and (now - last_seen_known_face_time) > LOCK_DELAY and tracked_posture == "Unknown":
        # This covers the case: they were present but left (and we can't find pose match)
        # but only lock if last seen posture before disappearance was Standing (we track it above)
        # If we never had standing, don't lock immediately (avoid false lock if person just moved face partially)
        # For simplicity, require that we had standing_started_time set within last 10s
        if standing_started_time is not None and (now - standing_started_time) < 10:
            should_lock = True

    if should_lock:
        print("[🔒] Locking workstation — authorized person stood up and left.")
        ctypes.windll.user32.LockWorkStation()
        break

    # show frame
    cv2.imshow("FRAL - Tracked", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
