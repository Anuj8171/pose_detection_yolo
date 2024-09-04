import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

def detect_pose(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Perform pose detection
    results = model(image)

    # Visualize the results
    annotated_image = results[0].plot()

    # Show the output image
    cv2.imshow("Pose Detection", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the input image
    input_image = "./run.jpg"

    # Run pose detection and show the result
    detect_pose(input_image)
