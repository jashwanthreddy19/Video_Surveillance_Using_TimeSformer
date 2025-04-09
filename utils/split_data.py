import os
import shutil
import random

# Set random seed for reproducibility
random.seed(42)

# Paths
data_dir = os.path.join("data\processed video clips")  # Path to the original data folder
train_dir = os.path.join("data", "train")
val_dir = os.path.join("data", "val")

# Split ratio
train_ratio = 0.8
val_ratio = 0.2

# Create train and val folders if they don't exist
for folder in [train_dir, val_dir]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Get all class folders
class_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

for class_name in class_folders:
    class_path = os.path.join(data_dir, class_name)
    
    # Get all video clips in the class folder
    videos = [f for f in os.listdir(class_path) if f.endswith(('.avi', '.mp4', '.mov'))]
    random.shuffle(videos)  # Shuffle videos to ensure randomness
    
    # Split the dataset
    # split_idx = int(len(videos) * train_ratio)
    split_idx = 200
    train_videos = videos[:split_idx]
    val_videos = videos[split_idx:250]

    # Create class folders in train and val directories
    train_class_path = os.path.join(train_dir, class_name)
    val_class_path = os.path.join(val_dir, class_name)
    os.makedirs(train_class_path, exist_ok=True)
    os.makedirs(val_class_path, exist_ok=True)

    # Move training videos
    for video in train_videos:
        shutil.copy(os.path.join(class_path, video), os.path.join(train_class_path, video))

    # Move validation videos
    for video in val_videos:
        shutil.copy(os.path.join(class_path, video), os.path.join(val_class_path, video))

    # Optional: Remove empty original class folder
    if len(os.listdir(class_path)) == 0:
        os.rmdir(class_path)

print(f"✅ Dataset successfully split into {train_dir} and {val_dir}!")
