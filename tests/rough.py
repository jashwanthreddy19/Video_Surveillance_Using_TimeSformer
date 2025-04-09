import os
import sys
import torch
import cv2
import numpy as np
from torchvision import transforms
sys.path.append(r"C:\Users\jashw\Desktop\Video Surveillance")
from datasets.custom_dataset import get_data_loader
from models.TimeSformer.timesformer.models.vit import TimeSformer


# ----- CONFIGURATION -----
MODEL_PATH = r"C:\Users\jashw\Desktop\Video Surveillance\models\data\trained_models\checkpoints\best_model_epoch11_acc90.34.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class_list = ['Abuse', 'Explosion', 'Fighting', 'RoadAccident', 'Robbery', 'Shooting', 'Vandalism', 'Normal Video']
video_path = r"C:\Users\jashw\Desktop\Video Surveillance\data\train\RoadAccidents\RoadAccidentsRoadAccidents005_x264_clip01.avi"

# Loading the model and moving to GPU
print(f"Loading fine-tuned model from: {MODEL_PATH}")

    # Create the TimeSformer model instance without pretrained_model
model = TimeSformer(img_size=224, num_classes=8, num_frames=96,
                    attention_type='divided_space_time', pretrained_model=None) #pretrained_model=None

# Load the state_dict from your fine-tuned model
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(checkpoint, strict=True) #strict true
model.to(DEVICE)

print("Model loaded successfully!")


def read_video(video_path):
        """Reads video and returns frames as numpy array."""
        clip_len = 96
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < clip_len:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()

        # If the video has fewer frames than clip_len, pad with the last frame
        if len(frames) < clip_len:
            last_frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
            while len(frames) < clip_len:
                frames.append(last_frame)
        
        return np.array(frames)


def process_chunk(chunk_path: str, model,class_list):
    """Dummy function to process a single chunk."""
    # In a real scenario, load your model and process the video chunk
    # This dummy version uses brightness as a placeholder
    # try:
    frames = read_video(chunk_path)
    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    if transform:
            frames = [transform(frame) for frame in frames]  # Each frame becomes [C, H, W]
        
    frames = torch.stack(frames)  # Shape: (T, C, H, W)
    frames_tensor = frames.permute(1, 0, 2, 3).float()
    frames_tensor = frames_tensor.unsqueeze(0)
    frames_tensor = frames_tensor.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(frames_tensor)
        _, predicted = torch.max(outputs.data, 1)
        
    predicted = class_list[predicted]
    
    return predicted
    # except Exception as e:
    #     print(f"Error processing chunk {chunk_path}: {e}")
    #     return "error"


print(process_chunk(video_path,model,class_list))