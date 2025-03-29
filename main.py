import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_size_from_filename(filename):
    return float(filename.split(".")[0].split("_")[1])

class LeafAgeDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.files = os.listdir(image_folder)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        img = Image.open(os.path.join(self.image_folder, file))
        img_tensor = self.transform(img)

        age = get_size_from_filename(file)
        return img_tensor, torch.tensor([age], dtype=torch.float32)


dataset = LeafAgeDataset("destination_images")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


class AgePredictionCNN(nn.Module):
    def __init__(self):
        super(AgePredictionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

model = AgePredictionCNN().to(device)
print(model)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
print(torch.cuda.is_available())
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

print("Training abgeschlossen! 🚀")


def predict_image(image_name):
    image_path = os.path.join("test_images", image_name)

    image = Image.open(image_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        predicted_age = model(image_tensor).item()

    print(f"Predicted transpiration area for the image {image_name}: {predicted_age:.2f} real transpiration area: {get_size_from_filename(image_name)}")

predict_image("_5_15.png")
predict_image("_18_5.png")
predict_image("_31_2.png")
predict_image("_240.png")
predict_image("_140_2.png")