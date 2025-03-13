import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

def draw_boxes(frame, results):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            label = result.names[int(box.cls[0])]  # Class label
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display label and confidence
            text = f"{label}: {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLOv8 inference
    results = model(frame)
    
    # Draw bounding boxes on the frame
    draw_boxes(frame, results)
    
    # Display the frame
    cv2.imshow("YOLOv8 Object Detection", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
