import torch
import torch.nn as nn
import torch.optim as optim


class AgePredictionCNN(nn.Module):
    def __init__(self):
        super(AgePredictionCNN, self).__init__()
        #Features extrahieren
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        #erzeuge einer Feature Map -> übersetzt Feature in Zahl
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Alter in Tagen
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)


model = AgePredictionCNN()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)
print(f"Geschätztes Alter: {output.item():.2f} Tage")