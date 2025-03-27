import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Prüfe, ob eine GPU verfügbar ist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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

        age = float(file.split(".")[0].split("_")[1])
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
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
print(torch.cuda.is_available())  # Sollte "True" ausgeben, wenn CUDA erkannt wird
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)  # Auf GPU verschieben
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

print("Training abgeschlossen! 🚀")


def predict_image(image_name):
    # Das Bild, das du inferieren möchtest
    image_path = os.path.join("test_images", image_name)

    # Lade das Bild
    image = Image.open(image_path)

    # Wende die Transformationen an (Umwandlung in Tensor)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)  # Hinzufügen einer Dimension für das Batch und auf das Gerät verschieben

    # Setze das Modell in den Evaluationsmodus
    model.eval()

    # Mache eine Vorhersage
    with torch.no_grad():
        predicted_age = model(image_tensor).item()

    print(f"Predicted transpiration area for the image {image_name}: {predicted_age:.2f}")

predict_image("_5_15.png")
predict_image("_18_5.png")
predict_image("_31_2.png")
predict_image("_92.png")
predict_image("_140_2.png")