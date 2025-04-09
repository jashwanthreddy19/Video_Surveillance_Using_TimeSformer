import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class VideoClipDataset(Dataset):
    def __init__(self, root_dir, clip_len=96, transform=None):
        """
        Args:
            root_dir (str): Path to the directory containing class subfolders.
            clip_len (int): Number of frames per video clip.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Get all video paths
        self.video_paths = []
        for cls_name in self.classes:
            cls_path = os.path.join(root_dir, cls_name)
            for video_name in os.listdir(cls_path):
                if video_name.endswith(('.avi', '.mp4')):
                    video_path = os.path.join(cls_path, video_name)
                    self.video_paths.append((video_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.video_paths)

    def read_video(self, video_path):
        """Reads video and returns frames as numpy array."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < self.clip_len:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()

        # If the video has fewer frames than clip_len, pad with the last frame
        if len(frames) < self.clip_len:
            last_frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
            while len(frames) < self.clip_len:
                frames.append(last_frame)
        
        return np.array(frames)

    def __getitem__(self, idx):
        video_path, label = self.video_paths[idx]
        
        # Read and process video
        frames = self.read_video(video_path)
        
        # Apply transformations to each frame
        if self.transform:
            frames = [self.transform(frame) for frame in frames]  # Each frame becomes [C, H, W]
        
        frames = torch.stack(frames)  # Shape: (T, C, H, W)
        frames_tensor = frames.permute(1, 0, 2, 3).float()  # (C, T, H, W)
        
        return frames_tensor, label



# Define transformations (resize and normalize)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_data_loader(data_dir, batch_size=8, clip_len=96, shuffle=True):
    """Creates and returns DataLoader for the dataset."""
    dataset = VideoClipDataset(root_dir=data_dir, clip_len=clip_len, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return data_loader
