import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Path to the main directory containing class folders
base_path = r"data\raw videos data\ucf_crime_dataset"  # Update with your path

# Dictionary to store video durations for each class
class_durations = {}

# Loop through each class folder
for class_name in os.listdir(base_path):
    class_path = os.path.join(base_path, class_name)
    
    # Ensure it's a directory
    if os.path.isdir(class_path):
        durations = []
        
        # Loop through videos in the class folder
        for video_name in os.listdir(class_path):
            video_path = os.path.join(class_path, video_name)
            
            # Read video and extract duration
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0  # Avoid division by zero
            cap.release()
            
            durations.append(duration)
        
        # Store durations for the class
        class_durations[class_name] = durations

# Plot the results
num_classes = len(class_durations)
cols = 3  # Number of columns in the subplot grid
rows = int(np.ceil(num_classes / cols))  # Calculate rows dynamically

fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))  # Adjust figure size

# Flatten the axes array if it's multi-dimensional
axes = axes.flatten() if num_classes > 1 else [axes]

# Iterate over each class and plot
for i, (class_name, durations) in enumerate(class_durations.items()):
    axes[i].bar(range(len(durations)), durations, color='blue')
    axes[i].set_title(class_name)  # Set subplot title
    axes[i].set_xlabel("Video Index")
    axes[i].set_ylabel("Duration (seconds)")
    
# Hide unused subplots if the number of classes is less than total subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()
