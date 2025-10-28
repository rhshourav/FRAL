import cv2
import os
import numpy as np
import time

# ----- Config -----
# Path to MobileNet-SSD model files (Caffe)
PROTO_PATH = "MobileNetSSD_deploy.prototxt"
MODEL_PATH = "MobileNetSSD_deploy.caffemodel"

# Confidence threshold for detections
CONF_THRESH = 0.5

# Person class id for the common MobileNet-SSD Caffe model
PERSON_CLASS_ID = 15

# Use webcam 0
VIDEO_DEVICE = 0

# ----- Initialize detector -----
net = None
use_dnn = False

if os.path.isfile(PROTO_PATH) and os.path.isfile(MODEL_PATH):
    print("[*] Loading MobileNet-SSD model (DNN)...")
    net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
    use_dnn = True
else:
    print("[!] DNN model files not found. Falling back to HOG person detector.")
    # HOG people detector (slower / less robust than modern DNNs but built-in)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Optional: enable CUDA backend if OpenCV built with CUDA
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# ----- Video capture -----
cap = cv2.VideoCapture(VIDEO_DEVICE)
if not cap.isOpened():
    raise RuntimeError("Could not open video device")

print("[*] Starting video. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    h, w = frame.shape[:2]

    person_detected = False

    if use_dnn:
        # Prepare blob — MobileNet SSD expects 300x300 input
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Iterate detections
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            class_id = int(detections[0, 0, i, 1])

            if confidence > CONF_THRESH and class_id == PERSON_CLASS_ID:
                # scale box back to frame size
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw rectangle & label
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 165, 255), 2)
                label = f"Person {confidence:.2f}"
                cv2.putText(frame, label, (startX, startY - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

                person_detected = True

    else:
        # HOG detectMultiScale works on resized frame for speed
        small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        rects, weights = hog.detectMultiScale(small, winStride=(8,8), padding=(8,8), scale=1.05)

        for (x, y, w_box, h_box) in rects:
            # scale back up coordinates
            startX = int(x * 2)
            startY = int(y * 2)
            endX = int((x + w_box) * 2)
            endY = int((y + h_box) * 2)

            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            cv2.putText(frame, "Person", (startX, startY - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            person_detected = True

    # Overlay top-left status
    if person_detected:
        cv2.putText(frame, "Person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        print("Person detected")
    else:
        cv2.putText(frame, "No person", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        print("No person")

    cv2.imshow("Person Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
