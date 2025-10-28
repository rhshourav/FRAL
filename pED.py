from ultralytics import YOLO
import cv2

# Load YOLOv8 model (pretrained on COCO dataset, includes "person" class)
model = YOLO("yolov8n.pt")  # use yolov8s.pt for higher accuracy

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("[*] Starting detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame, verbose=False)

    person_detected = False

    # Loop over detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])  # class id
            conf = float(box.conf[0])  # confidence

            # Class 0 = "person" in COCO dataset
            if cls == 0 and conf > 0.4:
                person_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display status
    if person_detected:
        cv2.putText(frame, "Person detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        print("Person detected")
    else:
        cv2.putText(frame, "No person", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        print("No person")

    cv2.imshow("Human Detector (YOLOv8)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
