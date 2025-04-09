import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Define the Simple3DCNN model (same as before)
class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=8, num_frames=32):
        super(Simple3DCNN, self).__init__()
        self.num_frames = num_frames

        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Adjust the fully connected layer input size based on the actual output size
        self.fc1 = nn.Linear(128 * (num_frames // 4) * 28 * 28, 512) # Corrected for temporal pooling
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.flatten(1)
        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)
        return x

# ----------------------
# Custom Dataset Class (YOU NEED TO IMPLEMENT THIS)
# ----------------------
class UCFCrimeDataset(Dataset):
    def __init__(self, root_dir, num_frames=32, transform=None):
        """
        Args:
            root_dir (string): Directory with all the video folders.
            num_frames (int): Number of frames to extract from each video.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir)) # Assuming each class has its own folder
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}
        self.video_paths = self._load_video_paths()

    def _load_video_paths(self):
        video_paths = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for video_file in os.listdir(class_dir):
                    if video_file.endswith(('.avi', '.mp4', '.mpeg', '.mov')): # Add your video extensions
                        video_path = os.path.join(class_dir, video_file)
                        video_paths.append((video_path, self.class_to_idx[class_name]))
        return video_paths

    def _extract_frames(self, video_path):
        # **IMPLEMENT YOUR FRAME EXTRACTION LOGIC HERE**
        # This function should take a video path, extract self.num_frames,
        # resize them to 224x224, and return them as a torch tensor of shape
        # (num_frames, 3, 224, 224) or (3, num_frames, 224, 224) depending on your needs.
        # You might need libraries like OpenCV (cv2) for video processing.
        # Example (very basic and might need adjustments):
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)
            frames = []
            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                if i in indices:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    if self.transform:
                        frame = self.transform(frame)
                    frames.append(frame)
            cap.release()
            if len(frames) < self.num_frames:
                # Handle cases where the video is shorter than num_frames
                padding = [frames[-1]] * (self.num_frames - len(frames))
                frames.extend(padding)
            frames_tensor = torch.stack(frames) # Shape: (num_frames, 3, 224, 224)
            return frames_tensor
        except Exception as e:
            print(f"Error processing video: {video_path} - {e}")
            return None

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path, label = self.video_paths[idx]
        frames = self._extract_frames(video_path)
        if frames is not None:
            # You might need to permute the dimensions to (channels, frames, height, width)
            frames = frames.permute(1, 0, 2, 3)
            return frames, label
        else:
            # Handle the case where frame extraction failed
            return torch.zeros((3, self.num_frames, 224, 224)), label # Return a zero tensor


# ----------------------
# Training Loop
# ----------------------
if __name__ == '__main__':
    # Hyperparameters
    num_classes = 8
    num_frames = 32 # You can change this to 96
    batch_size = 1
    learning_rate = 0.001
    num_epochs = 50
    root_dir = r"C:\Users\jashw\Desktop\Video Surveillance\data\processed video clips" # **CHANGE THIS TO YOUR DATASET PATH**

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the model
    model = Simple3DCNN(num_classes=num_classes, num_frames=num_frames).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create the dataset and data loader
    train_dataset = UCFCrimeDataset(root_dir=root_dir, num_frames=num_frames, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) # Adjust num_workers

    # TensorBoard writer
    writer = SummaryWriter(f'runs/ucf_crime_training_frames_{num_frames}')

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 9:  # Log every 10 batches
                avg_loss = running_loss / 10
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {avg_loss:.4f}')
                writer.add_scalar('training_loss', avg_loss, epoch * len(train_loader) + i + 1)
                running_loss = 0.0

        print(f'Epoch [{epoch+1}/{num_epochs}] finished.')

    print('Finished Training')
    writer.close()

    # You can add code here to save the trained model
    torch.save(model.state_dict(), f'ucf_crime_model_frames_{num_frames}.pth')