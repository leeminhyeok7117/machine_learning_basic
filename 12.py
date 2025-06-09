import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

batch_size = 128
latent_dim = 32
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='.', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),     
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),  
            nn.Unflatten(1, (1, 28, 28))  
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
    
model = AutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for xb, _ in train_loader:
        xb = xb.to(device)
        output = model(xb)
        loss = criterion(output, xb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}], Loss: {running_loss/len(train_loader):.4f}')
    
    
images, _ = next(iter(test_loader))
x = images[0].unsqueeze(0).to(device)

with torch.no_grad():
    z = model.encoder(x)
    
dim_to_modify = 31
mod_range = torch.linspace(-10, 10, steps=20).to(device)

modified_latents = z.repeat(len(mod_range), 1)
modified_latents[:, dim_to_modify] = mod_range

with torch.no_grad():
    decoded_images = model.decoder(modified_latents).cpu()
    
plt.figure(figsize=(18, 2))
for i in range(len(mod_range)):
    plt.subplot(1, len(mod_range), i + 1)
    plt.imshow(decoded_images[i].squeeze(), cmap='gray')
    plt.title(f"{mod_range[i].item():.1f}")
    plt.axis('off')
plt.suptitle(f'Modified latent dimension {dim_to_modify}')
plt.show()

