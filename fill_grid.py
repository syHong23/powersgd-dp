"""
Single-worker cross-check: the 2x2 grid (four configurations x five seeds)
on one GPU at batch 64.

These runs are not part of Table I. They are the independent replication of
the dispersion result referred to in Section V of the paper (compression
cost 0.29 +/- 0.13 points without DP against 0.93 +/- 1.09 with DP).

Provenance of the published grid file
-------------------------------------
The grid was filled incrementally in a single environment (RTX 4500 Ada,
torch 2.11.0, opacus 1.6.0). When no result file exists yet, load_grid()
seeds it with cells that had already been measured in that environment:

  * seed 42 of the two non-private configurations (KNOWN_SEED42), and
  * seeds 0 and 7 of the two private configurations, read from
    seed_results.json written by scripts/run_seeds.py, if present.

All remaining cells are run by this script. Results of an earlier round on a
different machine with unrecorded library versions are stored under
`_legacy_rtx3090` for the record and are excluded from every statistic.

To run the whole grid from scratch, delete KNOWN_SEED42 entries and make
sure no seed_results.json is present.

Usage
-----
    python3 fill_grid.py --dry-run          # list the cells to run
    python3 fill_grid.py --epochs 1         # quick functional check
    python3 fill_grid.py                    # full run
    python3 fill_grid.py --seeds 42 0 7     # a subset of seeds

Finished cells are skipped, so an interrupted run resumes when restarted.
The result file name encodes the GPU and torch version, and the file
records the environment and per-epoch curves.
"""

import argparse
import json
import math
import os
import platform
import time
from datetime import date

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.insert(0, '.')

SIGMA = 1.0
MAX_GRAD_NORM = 1.0
DELTA = 1e-5
BATCH_SIZE = 64
DATASET_SIZE = 60000
LR = 0.01
MOMENTUM = 0.9
DEFAULT_SEEDS = [42, 0, 7, 1, 2]
CONFIGS = ['sgd', 'powersgd', 'dpsgd', 'powersgd_dp']
LABEL = {'sgd': 'SGD', 'powersgd': 'PowerSGD only',
         'dpsgd': 'DP-SGD only', 'powersgd_dp': 'PowerSGD-DP'}

OLD = 'seed_results.json'          # written by scripts/run_seeds.py

# Cells already measured in this environment before this script was run
KNOWN_SEED42 = {
    'sgd':      {'acc': 99.09, 'eps': None, 'time': 672.7},
    'powersgd': {'acc': 98.88, 'eps': None, 'time': 823.9},
}

# Earlier results on a different machine: kept for the record only
LEGACY = {
    '_note': '2026-07, RTX 3090, torch/opacus versions not recorded. '
             'Not the environment described in the paper; excluded '
             'from all statistics.',
    'dpsgd': {'42': {'acc': 88.18, 'eps': 1.07, 'time': 1430}},
    'powersgd_dp': {'42': {'acc': 87.58, 'eps': 1.07, 'time': 1592}},
}


# --------------------------------------------------------- environment
def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'


def slug(s):
    return (s.replace('NVIDIA', '').replace('GeForce', '')
             .replace(' ', '').replace('-', '').lower())


def env_info():
    def ver(mod):
        try:
            m = __import__(mod)
            return getattr(m, '__version__', 'unknown')
        except Exception:
            return 'not installed'
    return {
        'gpu': gpu_name(),
        'torch': torch.__version__,
        'opacus': ver('opacus'),
        'powersgd': ver('powersgd'),
        'cuda': torch.version.cuda,
        'python': platform.python_version(),
        'date': str(date.today()),
        'batch': BATCH_SIZE, 'sigma': SIGMA, 'C': MAX_GRAD_NORM,
        'delta': DELTA, 'lr': LR, 'momentum': MOMENTUM,
        'powersgd_rank': 1, 'power_iters': 2, 'error_feedback': False,
    }


def grid_path():
    t = torch.__version__.split('+')[0].rsplit('.', 1)[0]
    return f"grid_{slug(gpu_name())}_torch{t}.json"


# --------------------------------------------------------------- model
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


def loaders():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    tr = datasets.MNIST('./data', train=True, download=True, transform=tf)
    te = datasets.MNIST('./data', train=False, transform=tf)
    return (DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(te, batch_size=256, shuffle=False))


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, p = model(x).max(1)
            total += y.size(0)
            correct += p.eq(y).sum().item()
    model.train()
    return 100.0 * correct / total


def make_powersgd(model):
    from powersgd_dp import BasicPowerSGD, BasicConfig, DPConfig
    ps = BasicPowerSGD([p for p in model.parameters()],
                       config=BasicConfig(rank=1, num_iters_per_step=2))
    ps.dp_config = DPConfig(enable_dp=False)   # DP is handled by Opacus
    return ps


def compress(model, powersgd):
    plist = [p for p in model.parameters() if p.grad is not None]
    grads = [p.grad.data.clone() for p in plist]
    for p, g in zip(plist, powersgd.aggregate(grads)):
        p.grad.data.copy_(g)


# ------------------------------------------------------------- runners
def run_plain(seed, epochs, device, use_compression):
    torch.manual_seed(seed)
    tr, te = loaders()
    model = SimpleCNN().to(device)
    opt = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    crit = nn.NLLLoss()
    ps = make_powersgd(model) if use_compression else None

    hist, start = [], time.time()
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            if use_compression:
                compress(model, ps)
            opt.step()
        a = evaluate(model, te, device)
        hist.append(round(a, 2))
        print(f"    ep {ep+1:2d}/{epochs} acc={a:.2f}%", flush=True)

    return {'acc': hist[-1], 'eps': None, 'time': time.time() - start,
            'history': hist}


def run_private(seed, epochs, device, use_compression):
    from opacus import PrivacyEngine

    torch.manual_seed(seed)
    tr, te = loaders()
    model = SimpleCNN().to(device)
    base = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    crit = nn.NLLLoss()

    engine = PrivacyEngine(accountant='rdp')
    model, dp_opt, tr = engine.make_private(
        module=model, optimizer=base, data_loader=tr,
        noise_multiplier=SIGMA, max_grad_norm=MAX_GRAD_NORM,
    )
    ps = make_powersgd(model) if use_compression else None
    sample_rate = BATCH_SIZE / DATASET_SIZE

    hist, start = [], time.time()
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            dp_opt.zero_grad()
            crit(model(x), y).backward()

            if use_compression:
                # Same ordering as the paper: Opacus clips and adds noise
                # first; compression is applied to the noised gradient.
                dp_opt.clip_and_accumulate()
                dp_opt.add_noise()
                dp_opt.scale_grad()
                engine.accountant.step(noise_multiplier=SIGMA,
                                       sample_rate=sample_rate)
                compress(model, ps)
                dp_opt.original_optimizer.step()
            else:
                dp_opt.step()

        a = evaluate(model, te, device)
        e = engine.get_epsilon(DELTA)
        hist.append(round(a, 2))
        print(f"    ep {ep+1:2d}/{epochs} acc={a:.2f}% eps={e:.2f}",
              flush=True)

    return {'acc': hist[-1], 'eps': e, 'time': time.time() - start,
            'history': hist}


RUNNERS = {
    'sgd':         lambda s, e, d: run_plain(s, e, d, False),
    'powersgd':    lambda s, e, d: run_plain(s, e, d, True),
    'dpsgd':       lambda s, e, d: run_private(s, e, d, False),
    'powersgd_dp': lambda s, e, d: run_private(s, e, d, True),
}


# --------------------------------------------------------------- state
def load_grid(path):
    if os.path.exists(path):
        return json.load(open(path))

    g = {'_env': env_info(), '_legacy_rtx3090': LEGACY}
    for c in CONFIGS:
        g[c] = {}
    for c, v in KNOWN_SEED42.items():
        g[c]['42'] = dict(v)

    if os.path.exists(OLD):
        old, n = json.load(open(OLD)), 0
        for c in ['dpsgd', 'powersgd_dp']:
            for s, v in old.get(c, {}).items():
                if s != '42':      # seed 42 there is the legacy result
                    g[c][s] = dict(v)
                    n += 1
        print(f"Imported {n} cells from {OLD} (seeds 0/7).")
        print("The legacy seed-42 values are kept under _legacy_rtx3090 "
              "and excluded from the statistics.\n")
    return g


def stats(vals):
    n = len(vals)
    m = sum(vals) / n
    sd = (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, sd, n


def summarize(g, seeds):
    e = g.get('_env', {})
    print("\n" + "=" * 78)
    print(f"GRID — {e.get('gpu','?')} | torch {e.get('torch','?')} | "
          f"opacus {e.get('opacus','?')}")
    print("=" * 78)

    w = 9
    hdr = f"{'config':<14}" + "".join(f"{'s'+str(s):>{w}}" for s in seeds) \
          + f"{'mean ± sd':>16}"
    print(hdr)
    print("-" * len(hdr))

    st, accs_by_seed = {}, {}
    for c in CONFIGS:
        cells = g.get(c, {})
        row, vals = "", []
        for s in seeds:
            if str(s) in cells:
                a = cells[str(s)]['acc']
                vals.append(a)
                accs_by_seed.setdefault(c, {})[s] = a
                row += f"{a:>{w-1}.2f} "
            else:
                row += f"{'--':>{w}}"
        if len(vals) >= 2:
            m, sd, n = stats(vals)
            st[c] = (m, sd, n)
            tail = f"{m:>10.2f} ± {sd:.2f}"
        else:
            tail = f"{'(n<2)':>16}"
        print(f"{LABEL[c]:<14}{row}{tail}")
    print("-" * len(hdr))

    if len(st) < 4:
        print("\nSome cells are empty; skipping analysis.")
        return

    # ---- paired differences (same seed), then summary ----
    def paired(c_hi, c_lo):
        common = [s for s in seeds
                  if s in accs_by_seed.get(c_hi, {})
                  and s in accs_by_seed.get(c_lo, {})]
        d = [accs_by_seed[c_hi][s] - accs_by_seed[c_lo][s] for s in common]
        return common, d

    print("\nCompression cost (paired by seed)")
    print("-" * 78)
    for lab, hi, lo in [('no DP', 'sgd', 'powersgd'),
                        ('with DP', 'dpsgd', 'powersgd_dp')]:
        common, d = paired(hi, lo)
        m, sd, n = stats(d)
        per = "  ".join(f"s{s}:{v:+.2f}" for s, v in zip(common, d))
        print(f"  {lab}: {m:+.2f} ± {sd:.2f} p  (n={n})")
        print(f"          {per}")

    _, dn = paired('sgd', 'powersgd')
    _, dd = paired('dpsgd', 'powersgd_dp')
    mn, sn, nn_ = stats(dn)
    md, sdd, nd = stats(dd)
    sep = math.sqrt(sn ** 2 / nn_ + sdd ** 2 / nd)
    print(f"\n  difference of the two costs: {md - mn:+.2f} p "
          f"(standard error {sep:.2f})")
    if sep > 0 and abs(md - mn) > 2 * sep:
        print("  -> larger than twice the standard error: the means differ")
        print("     at this n (treat as an observation, n is small).")
    else:
        print("  -> within twice the standard error: these runs do not")
        print("     separate the means.")

    print(f"\nPrivacy cost: "
          f"{st['sgd'][0]-st['dpsgd'][0]:+.2f} p (uncompressed), "
          f"{st['powersgd'][0]-st['powersgd_dp'][0]:+.2f} p (compressed)")

    # ---- LaTeX ----
    print("\n" + "=" * 78)
    print("[Table rows, LaTeX]")
    print("=" * 78)
    tex = {'sgd': 'SGD', 'powersgd': 'PowerSGD only',
           'dpsgd': 'DP-SGD only',
           'powersgd_dp': r'\textbf{PowerSGD-DP (ours)}'}
    comp = {'sgd': r'$1\times$', 'powersgd': r'$118.6\times$',
            'dpsgd': r'$1\times$', 'powersgd_dp': r'$118.6\times$'}
    for c in CONFIGS:
        m, sd, n = st[c]
        eps = r'$\infty$' if c in ('sgd', 'powersgd') else '1.07'
        print(f"{tex[c]} & {m:.2f} $\\pm$ {sd:.2f}\\% & {eps} & "
              f"{comp[c]} \\\\")

    n_seeds = st['sgd'][2]
    print("\n[Setup sentence]")
    print(f"Each configuration is run with {n_seeds} seeds "
          f"({', '.join(str(s) for s in seeds)}); we report the mean and")
    print("standard deviation of final-epoch test accuracy, and list the")
    print("individual runs in the text where the dispersion matters.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    path = grid_path()
    g = load_grid(path)

    todo = [(c, s) for c in CONFIGS for s in args.seeds
            if str(s) not in g.get(c, {})]

    print("=" * 78)
    for k, v in g['_env'].items():
        print(f"  {k:<14}: {v}")
    print(f"  {'result file':<14}: {path}")
    print(f"\nseeds {args.seeds} | epochs {args.epochs} | cells to run: {len(todo)}")
    for c, s in todo:
        print(f"    {LABEL[c]:<14} seed {s}")
    print("=" * 78, flush=True)

    if args.dry_run:
        summarize(g, args.seeds)
        return

    t0 = time.time()
    for i, (c, s) in enumerate(todo, 1):
        print(f"\n--- [{i}/{len(todo)}] {LABEL[c]}, seed {s} ---", flush=True)
        r = RUNNERS[c](s, args.epochs, device)
        g.setdefault(c, {})[str(s)] = r
        json.dump(g, open(path, 'w'), indent=2)
        done = time.time() - t0
        eta = done / i * (len(todo) - i)
        print(f"  acc={r['acc']:.2f}%  "
              f"eps={'-' if r['eps'] is None else round(r['eps'], 2)}  "
              f"time={r['time']:.0f}s  |  ETA {eta/60:.0f} min",
              flush=True)

    summarize(g, args.seeds)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
