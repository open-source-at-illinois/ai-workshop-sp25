import cv2
from ultralytics import YOLO

def main():
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)
        annotated_frame = results[0].plot()  # Draw boxes/labels
        
        # Display results
        cv2.imshow('YOLOv8 Object Detection', annotated_frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
