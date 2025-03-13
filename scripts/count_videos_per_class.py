import os
import matplotlib.pyplot as plt

dataset_path = "data/raw videos data/ucf_crime_dataset"  # Path to your UCF Crime dataset 
video_extensions = ['.avi', '.mp4', '.mov', '.mpeg', '.mpg']  # Common video file extensions
videos_count_per_class = {}

if not os.path.exists(dataset_path):
    print(f"Error: Dataset path not found: {dataset_path}")
else:
    print(f"Analyzing video classes in: {dataset_path}")
    class_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

    if not class_folders:
        print("Error: No class folders found within the dataset directory.")
    else:
        print("\nVideo counts per class:")
        for class_name in class_folders:
            class_dir_path = os.path.join(dataset_path, class_name)
            video_count = 0
            for filename in os.listdir(class_dir_path):
                if any(filename.lower().endswith(ext) for ext in video_extensions):
                    video_count += 1
            videos_count_per_class[class_name] = video_count
            print(f"- Class: {class_name}: {video_count} videos")

plt.bar(videos_count_per_class.keys(), videos_count_per_class.values(), color='skyblue')
plt.xlabel("Class Name")
plt.ylabel("Number of Videos")
plt.title("Number of Videos per Class")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(r"scripts\plots\Number of Videos per Class.png", dpi=600)
plt.show()
