import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Constants
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DATA_DIR = "/Users/vandanderia/Downloads/Assignment"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

# Dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.label_map = {}
        self._prepare_dataset()

    def _prepare_dataset(self):
        folders = [f for f in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, f))]
        for idx, folder in enumerate(folders):
            self.label_map[folder] = idx
            folder_path = os.path.join(self.root_dir, folder)
            for file in os.listdir(folder_path):
                if file.endswith(".png"):
                    self.data.append(os.path.join(folder_path, file))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms with additional data augmentation
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
dataset = CustomDataset(DATA_DIR, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the CNN model with Dropout
class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = CustomResNet(num_classes=len(dataset.label_map)).to(DEVICE)

# Loss, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training function
def train_model(model, loader):
    model.train()
    running_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).float()
    return running_loss / len(loader.dataset), correct / len(loader.dataset)

# Evaluation function
def evaluate_model(model, loader):
    model.eval()
    running_loss, correct = 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average="weighted")
    return running_loss / len(loader.dataset), correct / len(loader.dataset), precision

# Visualization of predictions
def visualize_predictions(model, loader, label_map, num_images=5):
    model.eval()
    images_shown = 0
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break

                img = images[i].cpu().numpy().transpose((1, 2, 0))
                img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)

                true_label = list(label_map.keys())[list(label_map.values()).index(labels[i].item())]
                predicted_label = list(label_map.keys())[list(label_map.values()).index(preds[i].item())]

                axes[images_shown].imshow(img)
                axes[images_shown].axis('off')
                axes[images_shown].set_title(f"True: {true_label}\nPred: {predicted_label}")
                images_shown += 1

            if images_shown >= num_images:
                break
    plt.show()

# Training loop
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

for epoch in range(EPOCHS):
    train_loss, train_acc = train_model(model, train_loader)
    test_loss, test_acc, test_precision = evaluate_model(model, test_loader)
    scheduler.step()

    train_losses.append(train_loss)
    train_accuracies.append(train_acc.item())
    test_losses.append(test_loss)
    test_accuracies.append(test_acc.item())

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test Precision: {test_precision:.4f}")

# Plot loss and accuracy
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
plt.plot(range(1, EPOCHS + 1), test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Over Epochs")

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), train_accuracies, label="Train Accuracy")
plt.plot(range(1, EPOCHS + 1), test_accuracies, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Over Epochs")

plt.show()

# Visualize predictions
visualize_predictions(model, test_loader, dataset.label_map)
