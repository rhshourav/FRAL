import cv2
import face_recognition
import os

# ====== FOLDER CONTAINING FACES ======
faces_DIR = "TranningData"

person_Face = []
person_Name = []
person_Chena= 0
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

process_this_frame = True  # for speed optimization

while True:
    ret, frame = video.read()
    if not ret:
        print("[!] Failed to grab frame")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.37, fy=0.37)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(person_Face, face_encoding)
            name = "Unknown"

            # Use face distance to find best match
            face_distances = face_recognition.face_distance(person_Face, face_encoding)
            if len(face_distances) > 0:
                best_match_index = face_distances.argmin()
                if matches[best_match_index]:
                    name = person_Name[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame  # skip every other frame for speed

    # Display results


    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
