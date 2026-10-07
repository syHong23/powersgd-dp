"""
PowerSGD aggregator used by all experiments of PowerSGD-DP.

This module is the BasicPowerSGD / PowerSGD implementation of the reference
library by Vogels et al. (https://github.com/epfml/powersgd, version 0.0.2,
commit f07be92), reproduced with one addition: a `dp_config` attribute that
the experiment scripts set to DPConfig(enable_dp=False).

Differential privacy is NOT implemented here. In every experiment, DP is
applied by Opacus before this aggregator is called:

    per-sample clipping -> Gaussian noise (Opacus) -> PowerSGD (this module)

so everything computed here, including what is all-reduced between workers,
is post-processing of the already-noised gradient.

With dp_config.enable_dp=False the computation is identical to the reference
library: on the same inputs, outputs and residuals match it exactly.

Notes on the configuration used in the paper
--------------------------------------------
* BasicPowerSGD is used directly (rank 1, two power iterations per step):
  every tensor is compressed from the first step, with no minimum
  compression rate and no uncompressed warm-up phase.
* aggregate() writes the compression residual back into its input
  tensors (for error feedback). The experiment scripts pass copies of the
  gradients and discard them, so error feedback is not applied.
* The compression_rate property is the library's analytic estimate. The
  ratio reported in the paper (118.6x) is the measured one; see
  count_comm.py.

Portions Copyright (c) 2019 EPFL Machine Learning and Optimization Laboratory.
Licensed under the MIT License (see LICENSE).
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, NamedTuple, Union

import torch

from powersgd.orthogonalization import orthogonalize
from powersgd.utils import allreduce_average, pack, unpack, is_distributed


class DPConfig(NamedTuple):
    """Kept for the experiment scripts, which set enable_dp=False.

    DP is applied by Opacus before aggregation; this module does not clip
    or add noise, and enable_dp=True is rejected.
    """
    enable_dp: bool = False


class Aggregator(ABC):
    @abstractmethod
    def aggregate(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Aggregates gradients across workers into an (approximate) average gradient.
        This method also changes its input gradients. It either sets them to zero if there is no compression,
        or to the compression errors, for error feedback.
        """
        pass


class AllReduce(Aggregator):
    def aggregate(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(gradients) == 0:
            return []
        buffer, shapes = pack(gradients)
        allreduce_average(buffer)
        out = unpack(buffer, shapes)
        for g in gradients:
            g.zero_()
        return out


class Config(NamedTuple):
    rank: int  # lower rank => more aggressive compression
    min_compression_rate: float = 2  # skip compression on some gradients
    num_iters_per_step: int = 1  # lower number => more aggressive compression
    start_compressing_after_num_steps: int = 100


class PowerSGD(Aggregator):
    """
    Applies PowerSGD only after a configurable number of steps,
    and only on parameters with strong compression.
    """

    def __init__(self, params: List[torch.Tensor], config: Config):
        self.config = config
        self.device = list(params)[0].device
        self.is_compressed_mask = [self._should_compress(p.shape) for p in params]

        self.step_counter = 0

        compressed_params, _ = self._split(params)
        self._powersgd = BasicPowerSGD(
            compressed_params,
            config=BasicConfig(
                rank=config.rank,
                num_iters_per_step=config.num_iters_per_step,
            ),
        )
        self._allreduce = AllReduce()

    def aggregate(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        self.step_counter += 1

        if self.step_counter <= self.config.start_compressing_after_num_steps:
            return self._allreduce.aggregate(gradients)

        compressed_grads, uncompressed_grads = self._split(gradients)
        return self._merge(
            self._powersgd.aggregate(compressed_grads),
            self._allreduce.aggregate(uncompressed_grads),
        )

    def _split(self, params: List[torch.Tensor]):
        compressed_params = []
        uncompressed_params = []
        for param, is_compressed in zip(params, self.is_compressed_mask):
            if is_compressed:
                compressed_params.append(param)
            else:
                uncompressed_params.append(param)
        return compressed_params, uncompressed_params

    def _merge(
        self, compressed: List[torch.Tensor], uncompressed: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        assert len(compressed) + len(uncompressed) == len(self.is_compressed_mask)
        compressed_iter = iter(compressed)
        uncompressed_iter = iter(uncompressed)
        merged_list = []
        for is_compressed in self.is_compressed_mask:
            if is_compressed:
                merged_list.append(next(compressed_iter))
            else:
                merged_list.append(next(uncompressed_iter))

        return merged_list

    def _should_compress(self, shape: torch.Size) -> bool:
        return (
            shape.numel() / avg_compressed_size(shape, self.config)
            > self.config.min_compression_rate
        )


class BasicConfig(NamedTuple):
    rank: int  # lower rank => more aggressive compression
    num_iters_per_step: int = 1  # lower number => more aggressive compression


class BasicPowerSGD(Aggregator):
    def __init__(self, params: List[torch.Tensor], config: BasicConfig):
        # Configuration
        self.config = config
        self.params = list(params)
        self.device = self.params[0].device
        self.dtype = self.params[0].dtype
        self.params_per_shape = self._matrices_per_shape(self.params)

        # State
        self.generator = torch.Generator(device=self.device).manual_seed(0)
        self.step_counter = 0

        # Initilize and allocate the low rank approximation matrices p and q.
        # _ps_buffer and _qs_buffer are contiguous memory that can be easily all-reduced, and
        # _ps and _qs are pointers into this memory.
        # _ps and _qs represent batches p/q for all tensors of the same shape.
        self._ps_buffer, ps_shapes = pack(
            [
                self._init_p_batch(shape, params)
                for shape, params in self.params_per_shape.items()
            ]
        )
        self._ps = unpack(self._ps_buffer, ps_shapes)

        self._qs_buffer, qs_shapes = pack(
            [
                self._init_q_batch(shape, params)
                for shape, params in self.params_per_shape.items()
            ]
        )
        self._qs = unpack(self._qs_buffer, qs_shapes)

        # DP is handled by Opacus before aggregation (see module docstring)
        self.dp_config = DPConfig(enable_dp=False)

    def aggregate(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Create a low-rank approximation of the average gradients by communicating with other workers.
        Modifies its inputs so that they contain the 'approximation error', used for the error feedback
        mechanism.
        """
        if self.dp_config.enable_dp:
            raise ValueError(
                "powersgd_dp does not implement DP. Apply clipping and noise "
                "with Opacus before calling aggregate(), and keep "
                "dp_config.enable_dp=False."
            )

        # Allocate memory for the return value of this function
        output_tensors = [torch.empty_like(g) for g in gradients]

        # Group the gradients per shape, and view them as matrices (2D tensors)
        gradients_per_shape = self._matrices_per_shape(gradients)
        outputs_per_shape = self._matrices_per_shape(output_tensors)
        shape_groups = [
            dict(
                shape=shape,
                grads=matrices,
                outputs=outputs_per_shape[shape],
                grad_batch=torch.stack(matrices),
                approximation=torch.zeros(
                    size=(len(matrices), *shape), device=self.device, dtype=self.dtype
                ),
            )
            for shape, matrices in list(gradients_per_shape.items())
        ]

        num_iters_per_step = self.config.num_iters_per_step
        for it in range(num_iters_per_step):
            # Alternate between left and right matrix multiplications
            iter_is_even = (self.step_counter * num_iters_per_step + it) % 2 == 0
            if iter_is_even:
                maybe_transpose = lambda g: g
                out_batches, in_batches = self._qs, self._ps
                out_buffer = self._qs_buffer
            else:
                maybe_transpose = batch_transpose
                out_batches, in_batches = self._ps, self._qs
                out_buffer = self._ps_buffer

            # Matrix multiplication
            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                orthogonalize(in_batch)
                torch.bmm(
                    batch_transpose(maybe_transpose(group["grad_batch"])), 
                    in_batch, 
                    out=out_batch
                )

            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                maybe_transpose(group["grad_batch"]).baddbmm_(
                    in_batch, 
                    batch_transpose(out_batch), 
                    alpha=-1
                )

            # Average across workers
            if is_distributed():
                num_workers = torch.distributed.get_world_size()
                torch.distributed.all_reduce(out_buffer)
            else:
                num_workers = 1

            # Construct low-rank reconstruction and update the approximation and error buffer
            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                maybe_transpose(group["approximation"]).baddbmm_(
                    in_batch, 
                    batch_transpose(out_batch),
                    alpha=1/num_workers
                )

        # Un-batch the approximation and error feedback, write to the output
        for group in shape_groups:
            for o, m, approx, mb in zip(
                group["outputs"],
                group["grads"],
                group["approximation"],
                group["grad_batch"],
            ):
                o.copy_(approx)
                m.copy_(mb)

        # Increment the step counter
        self.step_counter += 1

        return output_tensors

    def _init_p_batch(
        self, shape: torch.Size, params: List[torch.Tensor]
    ) -> torch.Tensor:
        rank = min(self.config.rank, min(shape))
        return torch.randn(
            [len(params), shape[0], rank], generator=self.generator, device=self.device
        )

    def _init_q_batch(
        self, shape: torch.Size, params: List[torch.Tensor]
    ) -> torch.Tensor:
        rank = min(self.config.rank, min(shape))
        return torch.randn(
            [len(params), shape[1], rank], generator=self.generator, device=self.device
        )

    @classmethod
    def _matrices_per_shape(
        cls,
        tensors: List[torch.Tensor],
    ) -> Dict[torch.Size, List[torch.Tensor]]:
        shape2tensors = defaultdict(list)
        for tensor in tensors:
            matrix = view_as_matrix(tensor)
            shape = matrix.shape
            shape2tensors[shape].append(matrix)
        return shape2tensors

    @property
    def uncompressed_num_floats(self) -> int:
        return sum(param.shape.numel() for param in self.params)

    @property
    def compressed_num_floats(self) -> float:
        return sum(avg_compressed_size(p.shape, self.config) for p in self.params)

    @property
    def compression_rate(self) -> float:
        return self.uncompressed_num_floats / self.compressed_num_floats



def batch_transpose(batch_of_matrices):
    return batch_of_matrices.permute([0, 2, 1])


def view_as_matrix(tensor: torch.Tensor):
    """
    Reshape a gradient tensor into a matrix shape, where the matrix has structure
    [output features, input features].
    For a convolutional layer, this groups all "kernel" dimensions with "input features".
    """
    return tensor.view(tensor.shape[0], -1)


def avg_compressed_size(shape: torch.Size, config: Union[Config, BasicConfig]) -> float:
    rank = min(config.rank, min(shape))
    return 0.5 * config.num_iters_per_step * rank * sum(shape)
