"""
Per-tensor accounting of the scalars PowerSGD transmits (Table II).

The transmitted volume is counted in two independent ways and the results
are cross-checked:

  [A] static count  - from the tensor shapes: each tensor is viewed as an
                      (m, n) matrix and sends r(m + n) scalars at rank r
  [B] dynamic count - dist.all_reduce is intercepted during one PowerSGD
                      step and the elements actually passed to it are summed

The two agree exactly for the model of the paper: 10,113 scalars per step
in place of 1,199,882, a ratio of 118.6x, in two collectives.

Usage
-----
    python3 count_comm.py

  - no GPU and no training needed; runs in seconds on CPU
  - if powersgd_dp.py or the powersgd package cannot be imported,
    only [A] is run
"""

import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, '.')

RANK = 1
NUM_ITERS_PER_STEP = 2
BYTES_PER_ELEM = 4  # fp32


# ----------------------------------------------------------------------
# Model of the paper (identical to the training scripts)
# ----------------------------------------------------------------------
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


def rule(n=78):
    print("-" * n)


# ----------------------------------------------------------------------
# [A] static count
# ----------------------------------------------------------------------
def static_count(model):
    print("\n[A] static count: P/Q size per tensor")
    rule()
    hdr = (f"{'Tensor':<15}{'Shape':>16}{'(m, n)':>15}"
           f"{'Params':>11}{'Sent':>9}{'Ratio':>9}")
    print(hdr)
    rule()

    rows, total_params, total_sent = [], 0, 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        shape = tuple(p.shape)
        numel = p.numel()
        total_params += numel

        # as in powersgd_dp.view_as_matrix: tensor.view(shape[0], -1)
        m = shape[0]
        n = numel // m
        sent = RANK * (m + n)

        total_sent += sent
        rows.append((name, shape, m, n, numel, sent))
        ratio = numel / sent
        print(f"{name:<15}{str(shape):>16}{f'({m}, {n})':>15}"
              f"{numel:>11,}{sent:>9,}{ratio:>8.1f}x")

    rule()
    print(f"{'TOTAL':<15}{'':>16}{'':>15}"
          f"{total_params:>11,}{total_sent:>9,}"
          f"{total_params / total_sent:>8.1f}x")
    return rows, total_params, total_sent


# ----------------------------------------------------------------------
# [B] dynamic count: intercept all_reduce and sum what is transmitted
# ----------------------------------------------------------------------
def dynamic_count(model, total_params):
    try:
        from powersgd_dp import BasicPowerSGD, BasicConfig, DPConfig
    except ImportError as e:
        print(f"\n[B] skipped: cannot import powersgd_dp.py: {e}")
        print("    Run this script from the repository root.")
        return None

    import torch.distributed as dist

    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29555')
    try:
        dist.init_process_group(backend='gloo', rank=0, world_size=1)
    except Exception as e:
        print(f"\n[B] skipped: process group initialization failed: {e}")
        return None

    stats = {'calls': 0, 'elements': 0, 'per_call': []}
    original_all_reduce = dist.all_reduce

    def counting_all_reduce(tensor, *args, **kwargs):
        stats['calls'] += 1
        stats['elements'] += tensor.numel()
        stats['per_call'].append(tensor.numel())
        return original_all_reduce(tensor, *args, **kwargs)

    dist.all_reduce = counting_all_reduce

    try:
        params = [p for p in model.parameters()]
        agg = BasicPowerSGD(
            params,
            config=BasicConfig(rank=RANK,
                               num_iters_per_step=NUM_ITERS_PER_STEP),
        )
        agg.dp_config = DPConfig(enable_dp=False)

        grads = [torch.randn_like(p) for p in params]
        agg.aggregate(grads)   # one step

        print("\n[B] dynamic count: elements passed to all_reduce (1 step)")
        rule()
        print(f"collectives          : {stats['calls']}")
        print(f"elements per call    : {stats['per_call']}")
        print(f"elements sent        : {stats['elements']:,}")
        print(f"bytes sent           : "
              f"{stats['elements'] * BYTES_PER_ELEM:,} B "
              f"({stats['elements'] * BYTES_PER_ELEM / 1024:.1f} KiB)")
        print(f"bytes uncompressed   : "
              f"{total_params * BYTES_PER_ELEM:,} B "
              f"({total_params * BYTES_PER_ELEM / 1024:.1f} KiB)")
        print(f"measured ratio       : "
              f"{total_params / stats['elements']:.1f}x")
        # The library's compression_rate property is an analytic estimate
        # (it sums the raw tensor dimensions and treats 1-D tensors as size
        # n), so it differs from the measured ratio above, which is the
        # figure reported in the paper.
        print(f"library compression_rate property (estimate): "
              f"{agg.compression_rate:.1f}x")
        return stats['elements']
    finally:
        dist.all_reduce = original_all_reduce
        if dist.is_initialized():
            dist.destroy_process_group()


# ----------------------------------------------------------------------
def main():
    model = SimpleCNN()

    print("=" * 78)
    print("PowerSGD transmitted volume")
    print(f"rank r={RANK}, num_iters_per_step={NUM_ITERS_PER_STEP}, fp32")
    print("=" * 78)

    rows, total_params, static_sent = static_count(model)
    measured = dynamic_count(model, total_params)

    print("\n[Summary]")
    rule()
    print(f"trainable parameters  : {total_params:,}")
    print(f"static count [A]      : {static_sent:,} elements "
          f"-> {total_params / static_sent:.1f}x")
    if measured is not None:
        print(f"dynamic count [B]     : {measured:,} elements "
              f"-> {total_params / measured:.1f}x")
        if measured == static_sent:
            print("The two counts agree.")
        else:
            print(f"The two counts differ by {abs(measured - static_sent):,}; "
                  f"[B] is what is actually transmitted.")

    final = measured if measured is not None else static_sent
    print(f"\nsent / uncompressed   : {100 * final / total_params:.2f}%")

    ub = math.sqrt(total_params) / (2 * RANK)
    print(f"\n[sanity check] even treating the whole model as one optimally "
          f"square matrix, the ratio is at most sqrt({total_params:,})/2 = {ub:.1f}x")
    print("       A reported ratio above this bound cannot be correct.")

    print("\n\n[LaTeX table]")
    rule()
    print(r"\begin{tabular}{lccrr}")
    print(r"\hline")
    print(r"\textbf{Tensor} & \textbf{Shape} & \textbf{$(m,n)$} & "
          r"\textbf{Params} & \textbf{Sent} \\")
    print(r"\hline")
    for name, shape, m, n, numel, sent in rows:
        tex_name = name.replace('_', r'\_')
        shape_str = str(shape).replace(' ', '')
        print(f"\\texttt{{{tex_name}}} & {shape_str} & ({m}, {n}) & "
              f"{numel:,} & {sent:,} \\\\")
    print(r"\hline")
    print(f"\\textbf{{Total}} & & & \\textbf{{{total_params:,}}} & "
          f"\\textbf{{{final:,}}} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
