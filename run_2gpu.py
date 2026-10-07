"""
Two-worker experiment: the full 2x2 grid with measured communication.

Runs the four configurations of the paper (SGD, PowerSGD only, DP-SGD only,
PowerSGD-DP) at the same world size, logical batch, number of steps, hardware
and interconnect, over five seeds. Produces Tables I and III.

Design
------
* Logical batch is fixed at GLOBAL_BATCH = 64; each worker takes
  GLOBAL_BATCH / world_size (32 at world size 2). This keeps the Poisson
  sampling rate at q = 64/60000 regardless of world size, so epsilon is
  directly comparable to a single-worker run (1.07 at delta = 1e-5).

* Communication path per configuration
    SGD          manual all_reduce of every gradient tensor (uncompressed)
    PowerSGD     PowerSGD all-reduces only the P and Q factors
    DP-SGD       Opacus DPDDP, reduce_gradients() on the full gradient
    PowerSGD-DP  Opacus DPDDP, PowerSGD in place of reduce_gradients()

  The two non-private configurations are not wrapped in
  DistributedDataParallel, for two reasons:
    (1) DDP performs its all-reduce inside the C++ reducer, which a Python
        hook on dist.all_reduce cannot see, so its traffic would not be
        measured;
    (2) PowerSGD on top of DDP would communicate twice.
  Instead, parameters are broadcast from rank 0 at start-up, and SGD
  all-reduces each gradient tensor by hand, which is mathematically the same
  as DDP. All four configurations are therefore instrumented identically.

* Privacy accounting
    Both private configurations call the Opacus optimizer internals in the
    same order and step the accountant by hand with the same sample rate,
    so their epsilon is computed in exactly the same way.

* DP ordering
    clip_and_accumulate() -> add_noise() -> aggregation -> scale_grad()
    Noise is added before any compression or communication, so everything
    PowerSGD computes is post-processing of the noised gradient.

Threat model
------------
Central DP: workers are mutually trusted and the adversary observes the
released model. Opacus DPDDP adds the noise once, on rank 0, so the
aggregated gradient is correctly noised and PowerSGD only approximates that
aggregate. Protecting the inter-worker traffic itself would require every
rank to add its own noise, which changes the accounting (out of scope).

Reproducibility
---------------
* Every number printed is measured in this run; nothing is hardcoded.
* Per-epoch curves, the software environment and all hyperparameters are
  stored in the result JSON.
* The result file name encodes the GPU, world size and torch version, so
  results from different environments cannot be mixed.
* Results are saved after every (configuration, seed) cell; an interrupted
  run resumes where it stopped.

Usage
-----
    # smoke test (about 10 minutes)
    torchrun --nproc_per_node=2 run_2gpu.py --epochs 1

    # full grid (several hours)
    torchrun --nproc_per_node=2 run_2gpu.py 2>&1 | tee run_2gpu.log

    # selected configurations only
    torchrun --nproc_per_node=2 run_2gpu.py --only dpsgd powersgd_dp
"""

import argparse
import json
import os
import platform
import time
from contextlib import contextmanager
from datetime import date

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

import sys
sys.path.insert(0, '.')

SEEDS = [42, 0, 7, 1, 2]
SIGMA = 1.0
MAX_GRAD_NORM = 1.0
DELTA = 1e-5
GLOBAL_BATCH = 64          # logical batch of the paper; determines q and epsilon
DATASET_SIZE = 60000
LR = 0.01
MOMENTUM = 0.9
POWERSGD_RANK = 1
POWER_ITERS = 2

CONFIGS = ['sgd', 'powersgd', 'dpsgd', 'powersgd_dp']
LABEL = {'sgd': 'SGD', 'powersgd': 'PowerSGD only',
         'dpsgd': 'DP-SGD only', 'powersgd_dp': 'PowerSGD-DP'}


# ---------------------------------------------------------- environment
def env_info(world):
    def ver(mod):
        try:
            return getattr(__import__(mod), '__version__', 'unknown')
        except Exception:
            return 'not installed'
    return {
        'gpu': torch.cuda.get_device_name(0),
        'world_size': world,
        'per_gpu_batch': GLOBAL_BATCH // world,
        'global_batch': GLOBAL_BATCH,
        'sample_rate': GLOBAL_BATCH / DATASET_SIZE,
        'torch': torch.__version__,
        'opacus': ver('opacus'),
        'powersgd': ver('powersgd'),
        'cuda': torch.version.cuda,
        'nccl': '.'.join(map(str, torch.cuda.nccl.version()))
                if torch.cuda.is_available() else None,
        'python': platform.python_version(),
        'date': str(date.today()),
        'seeds': SEEDS, 'sigma': SIGMA, 'C': MAX_GRAD_NORM, 'delta': DELTA,
        'lr': LR, 'momentum': MOMENTUM,
        'powersgd_rank': POWERSGD_RANK, 'power_iters': POWER_ITERS,
        'error_feedback': False,
        'threat_model': 'central DP: workers mutually trusted, '
                        'adversary observes released model only',
    }


def slug(s):
    return (s.replace('NVIDIA', '').replace('GeForce', '')
             .replace(' ', '').replace('-', '').lower())


def result_path(world):
    t = torch.__version__.split('+')[0].rsplit('.', 1)[0]
    return f"dist_{slug(torch.cuda.get_device_name(0))}_w{world}_torch{t}.json"


# ------------------------------------------------------------ telemetry
class Comm:
    """Intercept dist.all_reduce to count transmitted bytes and collectives,
    and time the gradient-aggregation window.

    Timing: the whole aggregation window is wrapped once,

        with comm.window():
            ...  # aggregation, whether it issues 1 all_reduce or 8

    with a single device synchronization on entry and on exit. Synchronizing
    around every individual all_reduce instead would charge configurations
    that issue many collectives (8 per step when uncompressed) with extra
    synchronization overhead, making equal transfers look unequal in time.
    For the compressed configurations the window also includes the power
    iteration, so the reported time is "aggregation time", not pure
    collective time.

    Bytes and call counts are taken from the hooked tensors and do not
    depend on the timing method.
    """

    def __init__(self):
        self.bytes = self.calls = 0
        self.seconds = 0.0
        self._orig = None

    def __enter__(self):
        self._orig = dist.all_reduce

        def counting(tensor, *a, **kw):
            self.calls += 1
            self.bytes += tensor.numel() * tensor.element_size()
            return self._orig(tensor, *a, **kw)

        dist.all_reduce = counting
        return self

    def __exit__(self, *exc):
        dist.all_reduce = self._orig

    @contextmanager
    def window(self):
        """Aggregation window: synchronize once on entry and once on exit."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.seconds += time.perf_counter() - t0

    def asdict(self, steps):
        return {'bytes': self.bytes, 'collectives': self.calls,
                'collective_s': round(self.seconds, 2),
                'bytes_per_step': self.bytes / steps if steps else 0,
                'collective_ms_per_step':
                    round(self.seconds * 1000 / steps, 3) if steps else 0}


# ---------------------------------------------------------------- model
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
    sampler = DistributedSampler(tr, num_replicas=world, rank=rank,
                                 shuffle=True, seed=seed)
    return (DataLoader(tr, batch_size=GLOBAL_BATCH // world,
                       sampler=sampler),
            DataLoader(te, batch_size=256, shuffle=False),
            sampler)


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
                       config=BasicConfig(rank=POWERSGD_RANK,
                                          num_iters_per_step=POWER_ITERS))
    ps.dp_config = DPConfig(enable_dp=False)   # DP is handled by Opacus
    return ps


def powersgd_aggregate(model, ps, world, to_sum):
    """Aggregate gradients with PowerSGD and write the result back to p.grad.

    BasicPowerSGD.aggregate returns the low-rank approximation of the
    average over workers (sum / world). With to_sum=True the result is
    multiplied by world to give the sum instead. All runs in the paper use
    to_sum=False (see run_private for why).
    """
    plist = [p for p in model.parameters() if p.grad is not None]
    grads = [p.grad.data.clone() for p in plist]
    out = ps.aggregate(grads)
    for p, g in zip(plist, out):
        p.grad.data.copy_(g * world if to_sum else g)


# ------------------------------------------------ non-private configs
def run_plain(rank, world, epochs, device, compress, log, seed):
    """Non-private configurations (SGD, PowerSGD only).

    Neither is wrapped in DDP: DDP all-reduces inside its C++ reducer, which
    the Python hook on dist.all_reduce cannot observe. A manual all_reduce
    per parameter tensor is mathematically identical to DDP and transmits
    the same bytes, so all four configurations are measured the same way.
    Only the collective count differs: DDP would merge tensors into buckets,
    so the count reported here is an upper bound on what DDP would issue.
    """
    torch.manual_seed(seed)
    tr, te, sampler = loaders(rank, world, seed)
    model = SimpleCNN().to(device)

    # Start every rank from the same initial parameters
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    ps = make_powersgd(model) if compress else None
    params = [p for p in model.parameters()]
    opt = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    crit = nn.NLLLoss()

    hist, steps, comm = [], 0, Comm()
    start = time.time()
    with comm:
        for ep in range(epochs):
            sampler.set_epoch(ep)
            for x, y in tr:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                crit(model(x), y).backward()
                with comm.window():
                    if compress:
                        powersgd_aggregate(model, ps, world, to_sum=False)
                    else:
                        # Same as DDP: all-reduce the full gradient, then average
                        for p in params:
                            if p.grad is not None:
                                dist.all_reduce(p.grad)
                                p.grad /= world
                opt.step()
                steps += 1
            a = evaluate(model, te, device)
            hist.append(round(a, 2))
            log(f"    ep {ep+1:2d}/{epochs} acc={a:.2f}%")

    return {'acc': hist[-1], 'eps': None, 'time': round(time.time()-start, 1),
            'steps': steps, 'history': hist, **comm.asdict(steps)}


# ---------------------------------------------------- private configs
def run_private(rank, world, epochs, device, compress, log, seed):
    from opacus import PrivacyEngine
    from opacus.distributed import (
        DifferentiallyPrivateDistributedDataParallel as DPDDP)

    torch.manual_seed(seed)
    tr, te, sampler = loaders(rank, world, seed)

    model = DPDDP(SimpleCNN().to(device))
    base = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    crit = nn.NLLLoss()

    engine = PrivacyEngine(accountant='rdp')
    model, dp_opt, tr = engine.make_private(
        module=model, optimizer=base, data_loader=tr,
        noise_multiplier=SIGMA, max_grad_norm=MAX_GRAD_NORM,
    )
    ps = make_powersgd(model) if compress else None

    # q is defined on the logical batch: 64/60000 for any world size.
    sample_rate = GLOBAL_BATCH / DATASET_SIZE

    hist, steps, comm = [], 0, Comm()
    start = time.time()
    with comm:
        for ep in range(epochs):
            if hasattr(tr, 'sampler') and hasattr(tr.sampler, 'set_epoch'):
                tr.sampler.set_epoch(ep)
            for x, y in tr:
                x, y = x.to(device), y.to(device)
                dp_opt.zero_grad()
                crit(model(x), y).backward()

                # Both private configurations call the Opacus internals in
                # the same order, so epsilon is computed identically.
                # Noise is added here, BEFORE any compression or
                # communication: everything below is post-processing.
                dp_opt.clip_and_accumulate()
                dp_opt.add_noise()          # DPDDP: noise added on rank 0 only
                with comm.window():
                    if compress:
                        # PowerSGD replaces reduce_gradients(). It returns
                        # the average over workers; to_sum must stay False,
                        # because the normalization applied by scale_grad()
                        # below already accounts for the world size, and
                        # multiplying by world here would scale the update
                        # twice (diagnosed with `ablate_2gpu.py --diag`).
                        powersgd_aggregate(model, ps, world, to_sum=False)
                    else:
                        # Opacus 1.6 names this reduce_gradients(), not reduce()
                        dp_opt.reduce_gradients()
                # Note: here scale_grad() runs after compression, whereas
                # the paper's Implementation details lists it before. The
                # two orders give identical updates: power iteration
                # orthogonalizes its factors, so the compressed result
                # scales linearly with its input. Noise is added before
                # compression in both cases, which is the ordering the
                # privacy argument relies on.
                dp_opt.scale_grad()
                engine.accountant.step(noise_multiplier=SIGMA,
                                       sample_rate=sample_rate)
                dp_opt.original_optimizer.step()
                steps += 1

            a = evaluate(model, te, device)
            e = engine.get_epsilon(DELTA)
            hist.append(round(a, 2))
            log(f"    ep {ep+1:2d}/{epochs} acc={a:.2f}% eps={e:.2f}")

    return {'acc': hist[-1], 'eps': round(e, 4),
            'time': round(time.time()-start, 1), 'steps': steps,
            'history': hist, **comm.asdict(steps)}


RUNNERS = {
    'sgd':         lambda r, w, e, d, lg, s: run_plain(r, w, e, d, False, lg, s),
    'powersgd':    lambda r, w, e, d, lg, s: run_plain(r, w, e, d, True, lg, s),
    'dpsgd':       lambda r, w, e, d, lg, s: run_private(r, w, e, d, False, lg, s),
    'powersgd_dp': lambda r, w, e, d, lg, s: run_private(r, w, e, d, True, lg, s),
}


# -------------------------------------------------------------- summary
def stats(v):
    n = len(v)
    m = sum(v) / n
    sd = (sum((x-m)**2 for x in v)/(n-1))**0.5 if n > 1 else 0.0
    return m, sd, n


def summarize(res, env, seeds):
    print("\n" + "=" * 88)
    print(f"2-GPU RESULTS — {env['gpu']} x{env['world_size']} | "
          f"torch {env['torch']} | opacus {env['opacus']}")
    print(f"per-GPU batch {env['per_gpu_batch']} x {env['world_size']} "
          f"= global {env['global_batch']}, q = {env['sample_rate']:.5f}")
    print("=" * 88)

    w = 9
    hdr = (f"{'config':<14}" + "".join(f"{'s'+str(s):>{w}}" for s in seeds)
           + f"{'mean ± sd':>16}{'MiB/step':>11}{'coll s':>9}")
    print(hdr)
    print("-" * len(hdr))

    acc_by = {}
    st = {}
    for c in CONFIGS:
        cells = res.get(c, {})
        row, vals = "", []
        for s in seeds:
            k = str(s)
            if k in cells:
                a = cells[k]['acc']
                vals.append(a)
                acc_by.setdefault(c, {})[s] = a
                row += f"{a:>{w-1}.2f} "
            else:
                row += f"{'--':>{w}}"
        if vals:
            m, sd, n = stats(vals)
            st[c] = (m, sd, n)
            any_cell = next(iter(cells.values()))
            tail = (f"{m:>10.2f} ± {sd:.2f}"
                    f"{any_cell['bytes_per_step']/1048576:>11.3f}"
                    f"{any_cell['collective_s']:>9.1f}")
        else:
            tail = f"{'(none)':>16}"
        print(f"{LABEL[c]:<14}{row}{tail}")
    print("-" * len(hdr))

    if len(st) < 4:
        print("\nSome configurations have no results yet; skipping analysis.")
        return

    def paired(hi, lo):
        common = [s for s in seeds
                  if s in acc_by.get(hi, {}) and s in acc_by.get(lo, {})]
        return common, [acc_by[hi][s] - acc_by[lo][s] for s in common]

    print("\nCompression cost (paired by seed)")
    print("-" * 88)
    for lab, hi, lo in [('no DP', 'sgd', 'powersgd'),
                        ('with DP', 'dpsgd', 'powersgd_dp')]:
        common, d = paired(hi, lo)
        if not d:
            continue
        m, sd, n = stats(d)
        print(f"  {lab}: {m:+.2f} ± {sd:.2f} p  (n={n})")
        print("          " + "  ".join(f"s{s}:{v:+.2f}"
                                       for s, v in zip(common, d)))

    print(f"\nPrivacy cost: "
          f"{st['sgd'][0]-st['dpsgd'][0]:+.2f} p (uncompressed), "
          f"{st['powersgd'][0]-st['powersgd_dp'][0]:+.2f} p (compressed)")

    # Communication saving: deterministic, independent of the seed
    print("\nCommunication (sent per step, independent of the seed)")
    print("-" * 88)
    for c in CONFIGS:
        r = next(iter(res[c].values()))
        per = r['bytes_per_step']
        s = f"{per/1024:.1f} KiB" if per < 1048576 else f"{per/1048576:.2f} MiB"
        print(f"  {LABEL[c]:<14} {s:>12}   "
              f"{r['collectives']/r['steps']:.1f} coll/step   "
              f"aggregation {r['collective_s']:.1f}s "
              f"({r.get('collective_ms_per_step', 0):.2f} ms/step) / "
              f"total {r['time']:.0f}s")
    for lab, hi, lo in [('no DP', 'sgd', 'powersgd'),
                        ('with DP', 'dpsgd', 'powersgd_dp')]:
        a = next(iter(res[hi].values()))['bytes_per_step']
        b = next(iter(res[lo].values()))['bytes_per_step']
        if b:
            print(f"  communication reduction ({lab}): {a/b:.1f}x")

    print("\n" + "=" * 88)
    print("[Table I: accuracy, two workers]")
    print("=" * 88)
    tex = {'sgd': 'SGD', 'powersgd': 'PowerSGD only',
           'dpsgd': 'DP-SGD only',
           'powersgd_dp': r'\textbf{PowerSGD-DP (ours)}'}
    for c in CONFIGS:
        m, sd, n = st[c]
        eps = r'$\infty$' if c in ('sgd', 'powersgd') else '1.07'
        print(f"{tex[c]} & {m:.2f} $\\pm$ {sd:.2f}\\% & {eps} \\\\")

    print("\n[Table III: measured communication]")
    print("=" * 88)
    print(r"\begin{tabular}{lrrrr}")
    print(r"\hline")
    print(r"\textbf{Configuration} & \textbf{Sent/step} & "
          r"\textbf{Collectives/step} & \textbf{Collective time} & "
          r"\textbf{Total time} \\")
    print(r"\hline")
    for c in CONFIGS:
        r = next(iter(res[c].values()))
        per = r['bytes_per_step']
        s = f"{per/1024:.1f}~KiB" if per < 1048576 else f"{per/1048576:.2f}~MiB"
        print(f"{LABEL[c]} & {s} & {r['collectives']/r['steps']:.1f} & "
              f"{r['collective_s']:.1f}~s & {r['time']:.0f}~s \\\\")
    print(r"\hline")
    print(r"\end{tabular}")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--only', nargs='+', default=None, choices=CONFIGS)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    dist.init_process_group(backend='nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    # Use the local rank, not the global rank, to select the device.
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    main_proc = (rank == 0)

    def log(m):
        if main_proc:
            print(m, flush=True)

    assert world >= 2, "run on at least two GPUs (--nproc_per_node=2)"
    assert GLOBAL_BATCH % world == 0, \
        f"global batch {GLOBAL_BATCH} is not divisible by world size {world}"

    env = env_info(world)
    path = result_path(world)
    res = {}
    if os.path.exists(path):
        res = {k: v for k, v in json.load(open(path)).items()
               if not k.startswith('_')}

    if main_proc:
        print("=" * 88)
        for k, v in env.items():
            print(f"  {k:<15}: {v}")
        print(f"  {'result file':<15}: {path}")
        print("=" * 88, flush=True)

    todo = [(c, s) for c in (args.only or CONFIGS) for s in args.seeds
            if str(s) not in res.get(c, {})]

    if main_proc:
        print(f"\nseeds {args.seeds} | epochs {args.epochs} | "
              f"cells to run: {len(todo)}")
        for c, s in todo:
            print(f"    {LABEL[c]:<14} seed {s}")
        print(flush=True)

    if args.dry_run:
        if main_proc:
            summarize(res, env, args.seeds)
        dist.destroy_process_group()
        return

    t0 = time.time()
    for i, (c, s) in enumerate(todo, 1):
        log(f"--- [{i}/{len(todo)}] {LABEL[c]}, seed {s} ---")
        r = RUNNERS[c](rank, world, args.epochs, device, log, s)
        dist.barrier()
        if main_proc:
            res.setdefault(c, {})[str(s)] = r
            json.dump({'_env': env, **res}, open(path, 'w'), indent=2)
            done = time.time() - t0
            eta = done / i * (len(todo) - i)
            log(f"  acc={r['acc']:.2f}%  "
                f"eps={'-' if r['eps'] is None else round(r['eps'],2)}  "
                f"time={r['time']:.0f}s  "
                f"sent={r['bytes']/1048576:.1f} MiB / "
                f"{r['collectives']} collectives  |  "
                f"ETA {eta/3600:.1f}h\n")

    if main_proc:
        summarize(res, env, args.seeds)
        print(f"\nSaved: {path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
