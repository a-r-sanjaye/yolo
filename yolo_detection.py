from ultralytics import YOLO
import cv2

def main():
    model = YOLO("yolov8n.pt")
    video_path =r"C:\Users\sanja_ib805js\OneDrive\realtime detection\yolo\song.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame, conf=0.5)
        annotated_frame = results[0].plot()

        # Display detections
        cv2.imshow("Movement Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

