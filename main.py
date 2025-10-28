import cv2
import face_recognition
import os

#Folder containing Faces

faces_DIR = "faces"

person_Face = []
person_Name = []

#load person Faces
for name in os.listdir(faces_DIR):
    img_path = os.path.join(faces_DIR, name)
    image = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(image)[0]

    person_Face.append(encoding)
    person_Name.append(os.path.splitext(name)[0])


# Initialize webcam
video = cv2.VideoCapture(0)
print("[**] Starting Video Capture. Press 'q' to exit.")
while True:
    ret, frame = video.read()
    if not ret:
        break

        #resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Fing all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encofings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            #compare with known faces
    matches =  face_recognition.compare_faces(person_Face, face_encoding)
    name = "Unknown"

    if True in matches:
        match_index = matches.index(True)
        name = person_Name[match_index]


        #Scale face locations back up
        top, right, bottom, left = [ v* 4 for v in face_locations ]

        #draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (lef + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
