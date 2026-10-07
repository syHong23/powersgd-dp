"""
Two-worker diagnostics and ablations for PowerSGD-DP.

This script was used during development to locate a scaling error in the
two-worker PowerSGD-DP path, and is kept so that the diagnosis can be
reproduced. The runs reported in the paper (run_2gpu.py) use the corrected
path, which corresponds to the `noscale` variant below.

Symptom
-------
With an earlier version of run_2gpu.py, DP-SGD reached the same accuracy on
one and two workers, but PowerSGD-DP lost about four points on two workers.
Since only the compressed path degraded, the cause had to lie in the
PowerSGD aggregation.

Candidate causes
----------------
    single worker (fill_grid.py)
        clip_and_accumulate -> add_noise -> scale_grad -> compress
    two workers (earlier run_2gpu.py)
        clip_and_accumulate -> add_noise -> compress (x world) -> scale_grad

    (a) the order of scale_grad and compress. Harmless: the low-rank
        approximation commutes with multiplication by a positive scalar.
    (b) the extra multiplication by world. If Opacus' scale_grad already
        accounts for the world size, the gradient is scaled twice.

Modes
-----
--diag
    On the same batch, compare the exact sum (dp_opt.reduce_gradients())
    with the low-rank approximation (PowerSGD, multiplied by world as in the
    earlier implementation). A low-rank approximation should have cosine
    below 1 and a norm ratio below 1; a norm ratio near world_size points
    to a scaling error. Run at world size 1 as well for a baseline.

--ablate
    Train four variants for 30 epochs each:
        base        earlier implementation (compress x world, then scale)
        noscale     multiplication by world removed   <- used in the paper
        scaleorder  single-worker order (scale, then compress), no x world
        sym         noise added on every rank with sigma / sqrt(W) instead
                    of once on rank 0 (same total variance)

--sgd-comm
    Measure the bytes of uncompressed SGD with a manual per-parameter
    all_reduce (DDP's C++ reducer is not visible to a Python hook). Bytes
    equal those of DDP; the collective count is an upper bound, since DDP
    merges tensors into buckets.

Usage
-----
    torchrun --nproc_per_node=2 ablate_2gpu.py --diag
    torchrun --nproc_per_node=1 ablate_2gpu.py --diag      # baseline
    torchrun --nproc_per_node=2 ablate_2gpu.py --ablate > ablate.log 2>&1
    torchrun --nproc_per_node=2 ablate_2gpu.py --sgd-comm
"""

import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

import sys
sys.path.insert(0, '.')

SIGMA = 1.0
MAX_GRAD_NORM = 1.0
DELTA = 1e-5
GLOBAL_BATCH = 64
DATASET_SIZE = 60000
LR = 0.01
MOMENTUM = 0.9
VARIANTS = ['base', 'noscale', 'scaleorder', 'sym']


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


def loaders(rank, world, seed):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    if rank == 0:
        datasets.MNIST('./data', train=True, download=True, transform=tf)
    dist.barrier()
    tr = datasets.MNIST('./data', train=True, download=False, transform=tf)
    te = datasets.MNIST('./data', train=False, download=(rank == 0),
                        transform=tf)
    dist.barrier()
    if world > 1:
        sampler = DistributedSampler(tr, num_replicas=world, rank=rank,
                                     shuffle=True, seed=seed)
        loader = DataLoader(tr, batch_size=GLOBAL_BATCH // world,
                            sampler=sampler)
    else:
        sampler = None
        loader = DataLoader(tr, batch_size=GLOBAL_BATCH, shuffle=True)
    return loader, DataLoader(te, batch_size=256, shuffle=False), sampler


def evaluate(model, loader, device):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, p = model(x).max(1)
            t += y.size(0)
            c += p.eq(y).sum().item()
    model.train()
    return 100.0 * c / t


def make_powersgd(model):
    from powersgd_dp import BasicPowerSGD, BasicConfig, DPConfig
    ps = BasicPowerSGD([p for p in model.parameters()],
                       config=BasicConfig(rank=1, num_iters_per_step=2))
    ps.dp_config = DPConfig(enable_dp=False)
    return ps


def powersgd_apply(model, ps, factor):
    """Write the PowerSGD approximation back to p.grad, multiplied by factor."""
    pl = [p for p in model.parameters() if p.grad is not None]
    g = [p.grad.data.clone() for p in pl]
    for p, o in zip(pl, ps.aggregate(g)):
        p.grad.data.copy_(o * factor if factor != 1.0 else o)


def build_private(seed, world, rank, device):
    from opacus import PrivacyEngine
    torch.manual_seed(seed)
    tr, te, sampler = loaders(rank, world, seed)

    if world > 1:
        from opacus.distributed import (
            DifferentiallyPrivateDistributedDataParallel as DPDDP)
        model = DPDDP(SimpleCNN().to(device))
    else:
        model = SimpleCNN().to(device)

    base = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    engine = PrivacyEngine(accountant='rdp')
    model, dp_opt, tr = engine.make_private(
        module=model, optimizer=base, data_loader=tr,
        noise_multiplier=SIGMA, max_grad_norm=MAX_GRAD_NORM)
    return model, dp_opt, engine, tr, te


# ====================================================================
# Diagnosis: compare the two aggregation paths directly
# ====================================================================
def diagnose(rank, world, device, steps=20):
    model, dp_opt, engine, tr, te = build_private(42, world, rank, device)
    ps = make_powersgd(model)
    crit = nn.NLLLoss()
    params = [p for p in model.parameters() if p.requires_grad]

    if rank == 0:
        print("=" * 78)
        print(f"Diagnosis: world_size={world}, "
              f"per-GPU batch={GLOBAL_BATCH // world}")
        print("=" * 78)
        print("Norm ratio and cosine similarity of the low-rank approximation")
        print("(PowerSGD) against the exact sum (reduce_gradients). A norm")
        print("ratio near world_size indicates a scaling error.")
        print()
        print(f"{'step':>5}{'||approx||/||exact||':>24}{'cosine':>10}"
              f"{'||exact||':>12}{'||approx||':>12}")
        print("-" * 78)

    ratios, coss = [], []
    it = iter(tr)
    for s in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        dp_opt.zero_grad()
        crit(model(x), y).backward()

        dp_opt.clip_and_accumulate()
        dp_opt.add_noise()

        snap = [p.grad.detach().clone() for p in params]

        # (1) exact sum
        if world > 1:
            dp_opt.reduce_gradients()
        exact = [p.grad.detach().clone() for p in params]

        # (2) low-rank approximation, multiplied by world as in the earlier implementation
        for p, sn in zip(params, snap):
            p.grad.data.copy_(sn)
        powersgd_apply(model, ps, float(world))
        approx = [p.grad.detach().clone() for p in params]

        ne = math.sqrt(sum(float(t.pow(2).sum()) for t in exact))
        na = math.sqrt(sum(float(t.pow(2).sum()) for t in approx))
        dot = sum(float((a * e).sum()) for a, e in zip(approx, exact))
        cos = dot / (ne * na) if ne * na > 0 else 0.0
        ratios.append(na / ne if ne > 0 else 0.0)
        coss.append(cos)

        if rank == 0:
            print(f"{s+1:>5}{na/ne:>24.4f}{cos:>10.4f}{ne:>12.3f}{na:>12.3f}")

        # No optimizer step: only the PowerSGD state advances
        dp_opt.zero_grad()

    if rank == 0:
        m = sum(ratios) / len(ratios)
        c = sum(coss) / len(coss)
        print("-" * 78)
        print(f"mean norm ratio {m:.4f},  mean cosine {c:.4f}")
        print()
        if m > 1.5:
            print("  >> Norm ratio well above 1: the multiplication by world")
            print("     is redundant. Compare with the noscale variant.")
        elif m < 0.6:
            print("  >> Norm ratio small: the approximation loses much of the")
            print("     energy, or the scale is too small. Compare with w=1.")
        else:
            print("  >> Scale is in the expected range.")
        print("\n  Compare with the same diagnosis at world size 1:")
        print("    torchrun --nproc_per_node=1 ablate_2gpu.py --diag")


# ====================================================================
# Ablations
# ====================================================================
def run_variant(variant, seed, epochs, rank, world, device, log):
    model, dp_opt, engine, tr, te = build_private(seed, world, rank, device)
    ps = make_powersgd(model)
    crit = nn.NLLLoss()
    sample_rate = GLOBAL_BATCH / DATASET_SIZE

    from opacus.optimizers.optimizer import DPOptimizer

    hist = []
    start = time.time()
    for ep in range(epochs):
        model.train()
        if hasattr(tr, 'sampler') and hasattr(tr.sampler, 'set_epoch'):
            tr.sampler.set_epoch(ep)
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            dp_opt.zero_grad()
            crit(model(x), y).backward()

            dp_opt.clip_and_accumulate()

            if variant == 'sym':
                # Every rank adds sigma/sqrt(W); the sum keeps variance sigma^2
                nm = dp_opt.noise_multiplier
                dp_opt.noise_multiplier = SIGMA / math.sqrt(world)
                DPOptimizer.add_noise(dp_opt)   # runs on every rank
                dp_opt.noise_multiplier = nm
            else:
                dp_opt.add_noise()              # DPDDP: rank 0 only

            if variant == 'scaleorder':
                # Single-worker order: scale, then compress, no x world
                dp_opt.scale_grad()
                powersgd_apply(model, ps, 1.0)
            elif variant == 'noscale':
                powersgd_apply(model, ps, 1.0)
                dp_opt.scale_grad()
            else:                                # base, sym
                powersgd_apply(model, ps, float(world))
                dp_opt.scale_grad()

            engine.accountant.step(noise_multiplier=SIGMA,
                                   sample_rate=sample_rate)
            dp_opt.original_optimizer.step()

        a = evaluate(model, te, device)
        e = engine.get_epsilon(DELTA)
        hist.append(round(a, 2))
        log(f"    ep {ep+1:2d}/{epochs} acc={a:.2f}% eps={e:.2f}")

    return {'acc': hist[-1], 'eps': round(e, 4), 'history': hist,
            'time': round(time.time() - start, 1)}


# ====================================================================
# SGD communication (manual all_reduce, no DDP)
# ====================================================================
def sgd_comm(rank, world, device, epochs=1, log=print):
    torch.manual_seed(42)
    tr, te, sampler = loaders(rank, world, 42)
    model = SimpleCNN().to(device)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    opt = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    crit = nn.NLLLoss()

    nbytes = calls = steps = 0
    tcoll = 0.0
    start = time.time()
    for ep in range(epochs):
        if sampler:
            sampler.set_epoch(ep)
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for p in model.parameters():
                dist.all_reduce(p.grad)
                p.grad /= world
                nbytes += p.grad.numel() * p.grad.element_size()
                calls += 1
            torch.cuda.synchronize()
            tcoll += time.perf_counter() - t0
            opt.step()
            steps += 1
        if rank == 0:
            log(f"    ep {ep+1}/{epochs} acc={evaluate(model, te, device):.2f}%")

    if rank == 0:
        print("\n" + "=" * 78)
        print("SGD communication (manual per-parameter all_reduce, no DDP)")
        print("=" * 78)
        print(f"  steps           : {steps}")
        print(f"  bytes sent       : {nbytes/1048576:.1f} MiB")
        print(f"  per step         : {nbytes/steps/1048576:.3f} MiB")
        print(f"  collectives/step : {calls/steps:.1f}")
        print(f"  collective time  : {tcoll:.1f}s / total {time.time()-start:.0f}s")
        print("\n  DDP merges tensors into buckets and issues fewer")
        print("  collectives, but transmits the same bytes. The collective")
        print("  count above is therefore an upper bound.")


# ====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag', action='store_true')
    ap.add_argument('--ablate', action='store_true')
    ap.add_argument('--sgd-comm', action='store_true')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 0])
    ap.add_argument('--variants', nargs='+', default=VARIANTS,
                    choices=VARIANTS)
    args = ap.parse_args()

    dist.init_process_group(backend='nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    main_proc = (rank == 0)

    def log(m):
        if main_proc:
            print(m, flush=True)

    assert GLOBAL_BATCH % world == 0

    if args.diag:
        diagnose(rank, world, device)

    elif args.sgd_comm:
        sgd_comm(rank, world, device, epochs=1, log=log)

    elif args.ablate:
        path = f'ablate_w{world}.json'
        res = json.load(open(path)) if os.path.exists(path) else {}
        todo = [(v, s) for v in args.variants for s in args.seeds
                if str(s) not in res.get(v, {})]
        log(f"ablation: {len(todo)} cells  variants={args.variants}  "
            f"seeds={args.seeds}  world={world}\n")

        for i, (v, s) in enumerate(todo, 1):
            log(f"--- [{i}/{len(todo)}] variant={v}, seed={s} ---")
            r = run_variant(v, s, args.epochs, rank, world, device, log)
            dist.barrier()
            if main_proc:
                res.setdefault(v, {})[str(s)] = r
                json.dump(res, open(path, 'w'), indent=2)
                log(f"  acc={r['acc']:.2f}%  eps={r['eps']:.2f}  "
                    f"time={r['time']:.0f}s\n")

        if main_proc:
            print("\n" + "=" * 78)
            print(f"Ablation results (world_size={world}, PowerSGD-DP)")
            print("=" * 78)
            print(f"{'variant':<14}" +
                  "".join(f"{'s'+str(s):>9}" for s in args.seeds) +
                  f"{'mean':>9}   description")
            print("-" * 78)
            desc = {
                'base': 'earlier implementation (compress x world -> scale)',
                'noscale': 'x world removed (used in the paper)',
                'scaleorder': 'single-worker order (scale -> compress)',
                'sym': 'noise on every rank, sigma/sqrt(W)',
            }
            for v in args.variants:
                cells = res.get(v, {})
                vals = [cells[str(s)]['acc'] for s in args.seeds
                        if str(s) in cells]
                row = "".join(f"{cells[str(s)]['acc']:>9.2f}"
                              if str(s) in cells else f"{'--':>9}"
                              for s in args.seeds)
                m = sum(vals)/len(vals) if vals else 0
                print(f"{v:<14}{row}{m:>9.2f}   {desc[v]}")
            print("-" * 78)
            print("\nReference values recorded during development:")
            print("  single-worker PowerSGD-DP      = 86.41 ± 0.45")
            print("  two-worker base (earlier run)  = 82.05 ± 1.03")
            print("\nA variant that recovers the single-worker level")
            print("identifies the cause.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
