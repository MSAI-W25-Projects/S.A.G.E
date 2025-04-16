import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

# -------------------------------
# Custom UTKFace Gender Dataset
# -------------------------------
class UTKGenderDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        gender = int(img_name.split("_")[1])  # Extract gender label
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, gender

# -------------------------------
# Main Training Logic
# -------------------------------
if __name__ == "__main__":
    # ✅ Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Set your dataset path
    data_dir = "UTKFace"  # TODO: Change this to your actual UTKFace path

    # ✅ Define transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ✅ Prepare dataset
    print("🔍 Loading dataset...")
    full_dataset = UTKGenderDataset(data_dir, transform=train_transforms)
    print(f"✅ Total dataset size: {len(full_dataset)} samples")

    if len(full_dataset) == 0:
        raise ValueError("🚫 Dataset folder is empty or path is wrong!")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"📦 Train set: {len(train_dataset)}, Validation set: {len(val_dataset)}")

    # ✅ Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)
    print("✅ DataLoaders initialized.")

    # ✅ Load ResNet-18 with modern pretrained weights
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 gender classes
    model = model.to(device)
    print("✅ Model loaded and moved to device.")

    # ✅ Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ✅ Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"\n🚀 Starting Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Progress"):
            print(f"🌀 Processing batch with {images.size(0)} samples")  # DEBUG LINE

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"📊 Epoch {epoch+1} Results — Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # ✅ Save model
        model_filename = f"resnet18_utk_gender_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), model_filename)
        print(f"💾 Model saved to {model_filename}")

    print("\n✅ Training complete. All models saved.")