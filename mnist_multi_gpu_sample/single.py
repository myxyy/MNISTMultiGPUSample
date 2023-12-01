import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
import torchvision.transforms as transform
from tqdm import tqdm

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform.ToTensor(), download = True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform.ToTensor(), download = True)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x

model = Net(28*28, 1024, 10).to('cuda')
model.train()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters())

batch_size = 32

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

num_epochs = 10

for _ in range(num_epochs):
    pbar = tqdm(train_loader)
    for batch in pbar:
        inputs, labels = batch
        inputs = inputs.view(-1, 28*28).to('cuda')
        labels = labels.to('cuda')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())