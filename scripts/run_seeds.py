"""
Single-worker seed runs for DP-SGD only and PowerSGD-DP.

An earlier single-purpose runner, kept for provenance: its seed-0 and seed-7
results are imported into the single-worker cross-check grid by
fill_grid.py. For new experiments, use fill_grid.py (single worker) or
run_2gpu.py (two workers), which cover all four configurations.

Settings match the paper: 30 epochs, batch 64, SGD (lr=0.01, momentum=0.9),
sigma=1.0, C=1.0, delta=1e-5, RDP accounting.

Usage
-----
    python3 scripts/run_seeds.py                # seeds 0 and 7
    python3 scripts/run_seeds.py --seeds 0      # one seed
    python3 scripts/run_seeds.py --epochs 5     # quick functional check

Results accumulate in seed_results.json; an interrupted run resumes.
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from opacus import PrivacyEngine

import sys
sys.path.insert(0, '.')
from powersgd_dp import BasicPowerSGD, BasicConfig, DPConfig

SIGMA = 1.0
MAX_GRAD_NORM = 1.0
DELTA = 1e-5
BATCH_SIZE = 64
DATASET_SIZE = 60000
LR = 0.01
MOMENTUM = 0.9
RESULTS_FILE = 'seed_results.json'


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


def get_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST('./data', train=True, download=True,
                               transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    return (DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False))


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


def run_dpsgd(seed, epochs, device):
    """DP-SGD only: the standard Opacus optimizer step."""
    torch.manual_seed(seed)
    train_loader, test_loader = get_loaders()

    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    criterion = nn.NLLLoss()

    engine = PrivacyEngine(accountant='rdp')
    model, optimizer, train_loader = engine.make_private(
        module=model, optimizer=optimizer, data_loader=train_loader,
        noise_multiplier=SIGMA, max_grad_norm=MAX_GRAD_NORM,
    )

    start = time.time()
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            criterion(model(data), target).backward()
            optimizer.step()
        acc = evaluate(model, test_loader, device)
        eps = engine.get_epsilon(delta=DELTA)
        print(f"  [DP-SGD s{seed}] {epoch + 1:2d}/{epochs} "
              f"acc={acc:.2f}% eps={eps:.2f}")

    return {'acc': acc, 'eps': eps, 'time': time.time() - start}


def run_powersgd_dp(seed, epochs, device):
    """PowerSGD-DP on one worker.

    Opacus clips and adds noise first; PowerSGD then compresses the noised
    gradient, so compression is post-processing.
    """
    torch.manual_seed(seed)
    train_loader, test_loader = get_loaders()
    sample_rate = BATCH_SIZE / DATASET_SIZE

    model = SimpleCNN().to(device)
    criterion = nn.NLLLoss()

    engine = PrivacyEngine(accountant='rdp')
    base_opt = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    model, dp_opt, train_loader = engine.make_private(
        module=model, optimizer=base_opt, data_loader=train_loader,
        noise_multiplier=SIGMA, max_grad_norm=MAX_GRAD_NORM,
    )

    params = [p for p in model.parameters()]
    powersgd = BasicPowerSGD(params,
                             config=BasicConfig(rank=1, num_iters_per_step=2))
    powersgd.dp_config = DPConfig(enable_dp=False)  # DP is handled by Opacus

    start = time.time()
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            dp_opt.zero_grad()
            criterion(model(data), target).backward()

            dp_opt.clip_and_accumulate()
            dp_opt.add_noise()
            dp_opt.scale_grad()
            engine.accountant.step(noise_multiplier=SIGMA,
                                   sample_rate=sample_rate)

            plist = [p for p in model.parameters() if p.grad is not None]
            grads = [p.grad.data.clone() for p in plist]
            for p, g in zip(plist, powersgd.aggregate(grads)):
                p.grad.data.copy_(g)

            dp_opt.original_optimizer.step()

        acc = evaluate(model, test_loader, device)
        eps = engine.get_epsilon(delta=DELTA)
        print(f"  [PowerSGD-DP s{seed}] {epoch + 1:2d}/{epochs} "
              f"acc={acc:.2f}% eps={eps:.2f}")

    return {'acc': acc, 'eps': eps, 'time': time.time() - start}


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    # Placeholder seed-42 entries from an earlier round on a different
    # machine. They are not rerun here, and fill_grid.py excludes them.
    return {
        'dpsgd': {'42': {'acc': 88.18, 'eps': 1.07}},
        'powersgd_dp': {'42': {'acc': 87.58, 'eps': 1.07}},
    }


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def summarize(results):
    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)

    stats = {}
    for key, label in [('dpsgd', 'DP-SGD only'),
                       ('powersgd_dp', 'PowerSGD-DP')]:
        accs = [v['acc'] for v in results[key].values()]
        n = len(accs)
        mean = sum(accs) / n
        if n > 1:
            var = sum((a - mean) ** 2 for a in accs) / (n - 1)
            sd = var ** 0.5
        else:
            sd = 0.0
        stats[key] = (mean, sd, n)
        seeds = ', '.join(sorted(results[key]))
        print(f"{label:<14} {mean:.2f} ± {sd:.2f}%  "
              f"(n={n}, seeds: {seeds})")

    gap = stats['dpsgd'][0] - stats['powersgd_dp'][0]
    print(f"\ncompression cost: {gap:.2f} p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 7])
    ap.add_argument('--epochs', type=int, default=30)
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    results = load_results()

    print("=" * 62)
    print(f"SEED RUNS — seeds {args.seeds}, {args.epochs} epochs")
    print(f"device: {device}")
    print("=" * 62)

    for seed in args.seeds:
        s = str(seed)

        if s not in results['dpsgd']:
            print(f"\n--- DP-SGD only, seed {seed} ---")
            results['dpsgd'][s] = run_dpsgd(seed, args.epochs, device)
            save_results(results)
        else:
            print(f"\n--- DP-SGD seed {seed}: already done, skipping ---")

        if s not in results['powersgd_dp']:
            print(f"\n--- PowerSGD-DP, seed {seed} ---")
            results['powersgd_dp'][s] = run_powersgd_dp(
                seed, args.epochs, device)
            save_results(results)
        else:
            print(f"\n--- PowerSGD-DP seed {seed}: already done, skipping ---")

    summarize(results)
    print(f"\nSaved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
