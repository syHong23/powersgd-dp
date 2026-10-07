"""
Single-worker plain SGD control (no DP, no compression), seed 42.

An earlier single-purpose runner, kept for provenance: its result is the
seed-42 SGD cell of the single-worker cross-check grid (KNOWN_SEED42 in
fill_grid.py). For new experiments, use fill_grid.py or run_2gpu.py.

Settings match the paper: 30 epochs, batch 64, SGD (lr=0.01, momentum=0.9).

Usage
-----
    python3 scripts/run_sgd_control.py
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SEED = 42
NUM_EPOCHS = 30
BATCH_SIZE = 64
LR = 0.01
MOMENTUM = 0.9


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return torch.log_softmax(self.fc2(x), dim=1)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            _, pred = model(data).max(1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()
    return 100.0 * correct / total


def main():
    torch.manual_seed(SEED)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("=" * 62)
    print(f"CONTROL: plain SGD (no DP, no compression) — seed {SEED}")
    print(f"{NUM_EPOCHS} epochs, batch {BATCH_SIZE}, "
          f"lr={LR}, momentum={MOMENTUM}")
    print(f"device: {device}")
    print("=" * 62)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST('./data', train=True, download=True,
                               transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params:,}\n")

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    criterion = nn.NLLLoss()

    history = []
    start = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()

        acc = evaluate(model, test_loader, device)
        history.append(acc)
        print(f"Epoch {epoch + 1:2d}/{NUM_EPOCHS} | Test Acc: {acc:.2f}%")

    elapsed = time.time() - start

    print("\n" + "=" * 62)
    print(f"Final: {history[-1]:.2f}%  |  Time: {elapsed:.1f}s")
    print("=" * 62)



if __name__ == "__main__":
    main()
