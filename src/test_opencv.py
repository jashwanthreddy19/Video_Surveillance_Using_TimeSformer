# src/test_opencv.py

import cv2

# Path to the sample video file (make sure to place a sample video in the data/ folder)
video_path = r"data\Normal_Videos_010_x264.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Processing video... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to 224x224
    resized_frame = cv2.resize(frame, (224, 224))
    
    # Display the frame
    cv2.imshow("Resized Frame", resized_frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Video processing complete.")
