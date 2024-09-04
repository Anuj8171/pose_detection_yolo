import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

def draw_landmarks(frame, keypoints):
    for person in keypoints:
        for keypoint in person:
            x, y, _ = keypoint
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Draw keypoint as a circle
    return frame

def detect_pose_in_video(video_path):
    # Open the video file or capture device
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform pose detection on the current frame
        results = model(frame)

        # Extract keypoints
        keypoints = results[0].keypoints.cpu().numpy()  # Get keypoints for each detected person

        # Draw landmarks on the frame
        frame_with_landmarks = draw_landmarks(frame, keypoints)
        print(frame_with_landmarks)
        # Show the output frame with landmarks
        cv2.imshow("Pose Detection", frame_with_landmarks)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the input video or use 0 for webcam
    input_video = "./video1.mp4"  # or 0 for webcam

    # Run pose detection and display the result in real-time
    detect_pose_in_video(input_video)
