import cv2
import face_recognition
import os
import numpy as np
import time

# ====== FOLDER CONTAINING FACES ======
faces_DIR = "TranningData"

person_Face = []
person_Name = []
check = ""

# ====== LOAD AND ENCODE TRAINING IMAGES ======
for filename in os.listdir(faces_DIR):
    img_path = os.path.join(faces_DIR, filename)
    try:
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            print(f"[!] No face found in {filename}, skipping.")
            continue
        person_Face.append(encodings[0])
        person_Name.append(os.path.splitext(filename)[0])
        print(f"[+] Loaded {filename}")
    except Exception as e:
        print(f"[!] Error loading {filename}: {e}")

print(f"[**] Total faces loaded: {len(person_Face)}")
if len(person_Face) == 0:
    print("[!] No valid faces found. Exiting...")
    exit()

# ====== INITIALIZE WEBCAM ======
video = cv2.VideoCapture(0)
print("[**] Starting Video Capture. Press 'q' to exit.")

process_this_frame = True  # speed optimization
prev_frame = None          # store previous frame for motion detection
last_motion_time = time.time()
motion_timeout = 5  # seconds before we say "no motion"

while True:
    ret, frame = video.read()
    if not ret:
        print("[!] Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # ===== MOTION DETECTION =====
    motion_detected = False
    if prev_frame is None:
        prev_frame = gray
        continue

    frame_delta = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 5000:  # adjust sensitivity
            motion_detected = True
            last_motion_time = time.time()
            break

    prev_frame = gray

    # ===== FACE DETECTION =====
    small_frame = cv2.resize(frame, (0, 0), fx=0.37, fy=0.37)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = []
    face_names = []

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        if len(face_locations) == 0:
            print("No faces detected.")
        else:
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(person_Face, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(person_Face, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = person_Name[best_match_index]

                face_names.append(name)
                print(name)

    process_this_frame = not process_this_frame

    # ===== DISPLAY RESULTS =====
    if len(face_locations) == 0 and (time.time() - last_motion_time) > motion_timeout:
        # No faces + no motion for a while
        text = "No one in frame"
        print(text)
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    elif motion_detected:
        cv2.putText(frame, "Motion Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.imshow("Face + Motion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
