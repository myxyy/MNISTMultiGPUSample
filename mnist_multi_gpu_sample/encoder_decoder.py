import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, width, height, hidden_dim):
        super().__init__()
        self.width = width
        self.height = height
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 64, 3)
        self.fc = nn.Linear((width-4)*(height-4)*64, hidden_dim)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = x.view(-1, (self.width-4)*(self.height-4)*64)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, width, height, hidden_dim):
        super().__init__()
        self.width = width
        self.height = height
        self.act = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, (width-4)*(height-4)*64)
        self.conv2 = nn.ConvTranspose2d(64, 8, 3)
        self.conv1 = nn.ConvTranspose2d(8, 1, 3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 64, self.width-4, self.height-4)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x
