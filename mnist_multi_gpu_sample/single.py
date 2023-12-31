import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
import torchvision.transforms as transform
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os
from encoder_decoder import Encoder, Decoder

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform.ToTensor(), download = True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform.ToTensor(), download = True)

class AutoEncoder(nn.Module):
    def __init__(self, width, height, hidden_dim):
        super().__init__()
        self.encoder = Encoder(width, height, hidden_dim)
        self.decoder = Decoder(width, height, hidden_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


width = 28
height = 28
hidden_dim = 32

model = AutoEncoder(width, height, hidden_dim).cuda()
model.train()

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters())

batch_size = 32

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

num_epochs = 10

train_start = time.time()
for epoch in range(num_epochs):
    pbar = tqdm(train_loader, desc=f'epoch {epoch}')
    for batch in pbar:
        inputs, _ = batch
        x = inputs.view(-1, 1, width, height).cuda()
        optimizer.zero_grad()
        x_hat, z = model(x)
        loss = criterion(x_hat, x)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())
train_end = time.time()
train_time = train_end - train_start
print(f'train time: {train_time}')

model.eval()

num_figs = 5
plt.figure(figsize=(num_figs, 3))
for i in range(num_figs):
    input, label = test_dataset[i]
    x = input.view(1, 1, width, height).cuda()
    x_hat, z = model(x)

    ax = plt.subplot(3, num_figs, i+1)
    plt.imshow(x.cpu().detach().numpy().reshape(width, height), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, num_figs, i+1+num_figs)
    plt.imshow(z.cpu().detach().numpy().reshape(1, hidden_dim))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, num_figs, i+1+num_figs*2)
    plt.imshow(x_hat.cpu().detach().numpy().reshape(28, 28), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig('outputs/compare.png')

