import os
import sys
import torch
import numpy as np
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler  # Mixed Precision Training
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
sys.path.append(r"C:\Users\jashw\Desktop\Video Surveillance")
from datasets.custom_dataset import get_data_loader  # Your custom dataset loader
from models.TimeSformer.timesformer.models.vit import TimeSformer # Import TimeSformer class

# -------------------
# Hyperparameters
# -------------------
train_dir = "data/train/"
val_dir = "data/val/"
batch_size = 1  # Reduce batch size to lower VRAM usage
num_epochs = 5
learning_rate = 1e-4
num_classes = 8
accumulation_steps = 4  # Gradient accumulation

# -------------------
# Load Pre-trained TimeSformer Checkpoint
# -------------------
def load_model(pretrained_path):
    print(f"Loading pre-trained model from: {pretrained_path}")
    
    model = TimeSformer(img_size=224, num_classes=num_classes, num_frames=96,  # Reduced frames for lower memory
                        attention_type='divided_space_time', pretrained_model=pretrained_path)

    checkpoint = torch.load(pretrained_path, map_location="cpu")
    model.load_state_dict(checkpoint.get('model', checkpoint), strict=False)

    print("Model loaded successfully!")
    return model


# -------------------
# Training Function
# -------------------
def train_model(model, train_loader, val_loader, device, class_weights,class_counts):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler(device='cuda')  # Mixed Precision Training

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        i = 1

        for step, (videos, labels) in enumerate(train_loader):
            videos, labels = videos.to(device), labels.to(device)
            print(f"Processing Video {i} Remaining {(sum(class_counts) - i)}")
            with torch.amp.autocast(device_type='cuda'):  # Mixed Precision
                videos = videos.permute(0, 3, 2, 1, 4)  # [B, C, T, H, W]
                outputs = model(videos)
                loss = criterion(outputs, labels) / accumulation_steps  # Scale loss

            scaler.scale(loss).backward()  # Scaled backpropagation

            if (step + 1) % accumulation_steps == 0:  # Update weights after accumulating gradients
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()  # Free unused GPU memory

            running_loss += loss.item() * accumulation_steps
            i += 1

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        validate_model(model, val_loader, device)


# -------------------
# Validation Function
# -------------------
def validate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            videos = videos.permute(0, 3, 2, 1, 4)
            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        print("No validation samples found. Skipping validation.")
    else:
        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")



# -------------------
# Main Function
# -------------------
def main():
    device = torch.device("cuda")
    print(f"Using device: {device}")

    pretrained_path = "data/trained_models/Kinetics-400 models/TimeSformer_divST_96x4_224_K400.pyth"
    model = load_model(pretrained_path)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"📦 Total parameters       : {total_params}")
    print(f"🔧 Trainable parameters   : {trainable_params}")
    print(f"🚫 Non-trainable parameters: {non_trainable_params}")

    print("\n🧩 Trainable layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()} params")


    train_loader = get_data_loader(train_dir, batch_size=batch_size, clip_len=32, shuffle=True)  # Reduce clip_len
    val_loader = get_data_loader(val_dir, batch_size=batch_size, clip_len=32, shuffle=False)

    # Compute class weights
    data_path = r"C:\Users\jashw\Desktop\Video Surveillance\data\train"
    class_counts = [len(os.listdir(os.path.join(data_path, class_name))) for class_name in os.listdir(data_path)]
    class_labels = np.arange(len(class_counts))
    class_weights = compute_class_weight('balanced', classes=class_labels, y=np.concatenate([[i] * count for i, count in enumerate(class_counts)]))
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Train model
    train_model(model, train_loader, val_loader, device, class_weights,class_counts)

    # Save fine-tuned model
    model_path = "data/trained_models/timesformer_finetuned.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Fine-tuned model saved to {model_path}")


if __name__ == "__main__":
    main()
