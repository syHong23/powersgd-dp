"""
Single-worker PowerSGD-only control (no DP), batch 64, seed 42.

An earlier single-purpose runner, kept for provenance: its result is the
seed-42 PowerSGD cell of the single-worker cross-check grid (KNOWN_SEED42
in fill_grid.py). For new experiments, use fill_grid.py or run_2gpu.py.

The gradient path is the same as in PowerSGD-DP: the gradient is copied,
compressed at rank 1 and written back, so the only difference from the
private configuration is the absence of DP.

Usage
-----
    python3 scripts/run_powersgd_matched.py
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.insert(0, '.')
from powersgd_dp import BasicPowerSGD, BasicConfig, DPConfig

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

    print("=" * 64)
    print(f"MATCHED CONTROL: PowerSGD only, single GPU — seed {SEED}")
    print(f"{NUM_EPOCHS} epochs, batch {BATCH_SIZE}, lr={LR}, "
          f"momentum={MOMENTUM}, rank=1")
    print(f"device: {device}")
    print("=" * 64)

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
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    criterion = nn.NLLLoss()

    params = [p for p in model.parameters()]
    powersgd = BasicPowerSGD(params,
                             config=BasicConfig(rank=1, num_iters_per_step=2))
    powersgd.dp_config = DPConfig(enable_dp=False)

    history = []
    start = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            criterion(model(data), target).backward()

            # Same path as PowerSGD-DP: copy the gradient, compress, write back
            plist = [p for p in model.parameters() if p.grad is not None]
            grads = [p.grad.data.clone() for p in plist]
            for p, g in zip(plist, powersgd.aggregate(grads)):
                p.grad.data.copy_(g)

            optimizer.step()

        acc = evaluate(model, test_loader, device)
        history.append(acc)
        print(f"Epoch {epoch + 1:2d}/{NUM_EPOCHS} | Test Acc: {acc:.2f}%")

    elapsed = time.time() - start
    ps = history[-1]

    print("\n" + "=" * 64)
    print(f"Final: {ps:.2f}%  |  Time: {elapsed:.1f}s")
    # compression_rate is the library's analytic estimate; the measured
    # per-step ratio (118.6x) is given by count_comm.py.
    print(f"Compression (library estimate): {powersgd.compression_rate:.1f}x")
    print("=" * 64)


if __name__ == "__main__":
    main()
