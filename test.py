import os

import torch
from PIL import Image
from torch import device, nn
from torchvision import transforms

class AgePredictionCNN(nn.Module):
    def __init__(self):
        super(AgePredictionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)
def get_size_from_filename(filename):
    return float(filename.split(".")[0].split("_")[1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = torch.load("model.pth", map_location=device)
model.eval()

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
    return predicted_age, get_size_from_filename(image_name)

out = []

for image in os.listdir("test_images"):
    out.append(predict_image(image))
print(out)