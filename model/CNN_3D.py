import torch
import torch.nn as nn
import torch.nn.functional as F

class HSI_3DCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # Spectral-first convolutions
        self.conv1 = nn.Conv3d(1, 8, kernel_size=(7,3,3), padding=(3,1,1))
        self.bn1 = nn.BatchNorm3d(8)

        self.conv2 = nn.Conv3d(8, 16, kernel_size=(5,3,3), padding=(2,1,1))
        self.bn2 = nn.BatchNorm3d(16)

        self.conv3 = nn.Conv3d(16, 32, kernel_size=(3,3,3), padding=1)
        self.bn3 = nn.BatchNorm3d(32)

        self.pool = nn.AdaptiveAvgPool3d((1,1,1))

        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x