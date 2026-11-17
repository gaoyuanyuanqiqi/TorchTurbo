"""Bridge between TorchTurbo's FSDP-owned experts and HyperMulticore on Ascend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .experts import GroupedMLP
from .moe_layer import MoELayer

if TYPE_CHECKING:
    from hyper_multicore import MegaMoeExperts


@dataclass
class MegaMoeBackend:
    """Serial layers with matching geometry and numerics share one HMC executor."""

    executor: MegaMoeExperts
    local_token_capacity: int | None

    def execute(
        self, experts: GroupedMLP, inputs: torch.Tensor, ids: torch.Tensor, probs: torch.Tensor,
    ) -> torch.Tensor:
        """Consume live unsharded weights inside the original expert module forward."""
        if inputs.dtype != torch.bfloat16:
            raise ValueError("MegaMoE requires BF16 expert activations.")
        if self.local_token_capacity is not None and ids.shape[0] > self.local_token_capacity:
            raise ValueError(f"MegaMoE input exceeds moe.local_token_capacity={self.local_token_capacity}.")
        hidden = experts.config.hidden_size
        intermediate = experts.config.moe_ffn_hidden_size
        local = experts.num_local_experts
        w1, w2 = experts.weight1, experts.weight2
        if any(w.dtype not in (torch.bfloat16, torch.float32) for w in (w1, w2)):
            raise ValueError("MegaMoE requires BF16 expert weights or FP32 master parameters.")
        weights = (
            w1.to(inputs.dtype).view(local, hidden, 2 * intermediate),
            w2.to(inputs.dtype).view(local, intermediate, hidden),
        )
        return self.executor(inputs, ids.int(), probs, expert_weights=weights)


def bind_mega_moe(
    model: torch.nn.Module, ep_group: dist.ProcessGroup | None, local_token_capacity: int | None,
) -> None:
    """Bind after model replacement and before activation checkpointing and FSDP."""
    if ep_group is None or dist.get_world_size(ep_group) <= 1:
        raise ValueError("MegaMoE requires expert parallelism with ep_size > 1.")
    layers = [module for module in model.modules() if isinstance(module, MoELayer)]
    if not layers:
        raise ValueError("MegaMoE requires a model using TorchTurbo MoELayer.")
    ep_size = dist.get_world_size(ep_group)
    for layer in layers:
        cfg = layer.config
        if not isinstance(layer.experts, GroupedMLP) or layer.experts.use_te:
            raise ValueError("MegaMoE requires TorchTurbo GroupedMLP without Transformer Engine.")
        if not cfg.gated_linear_unit or cfg.activation_func is not F.silu or cfg.moe_apply_probs_on_input:
            raise ValueError("MegaMoE requires SwiGLU experts with routing probabilities applied before FC2.")
        if layer.experts.num_local_experts * ep_size != cfg.num_moe_experts:
            raise ValueError("MegaMoE expert partition does not match the EP process group.")

    try:
        from hyper_multicore import MegaMoeExperts, MegaMoeNumericsOptions
    except ModuleNotFoundError as exc:
        if exc.name != "hyper_multicore":
            raise
        raise ImportError("Install an Ascend A3 hyper-multicore wheel matching Torch and torch_npu.") from exc

    backends = {}
    for layer in layers:
        cfg = layer.config
        limit = layer.experts.swiglu_limit or None
        clamp_after = bool(limit and cfg.moe_swiglu_clamp_gate_after_activation)
        key = (
            cfg.hidden_size, cfg.moe_ffn_hidden_size, cfg.num_moe_experts,
            cfg.moe_router_topk, limit, clamp_after,
        )
        if key not in backends:
            executor = MegaMoeExperts(
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.moe_ffn_hidden_size,
                num_experts=cfg.num_moe_experts,
                top_k=cfg.moe_router_topk,
                ep_group=ep_group,
                numerics_options=MegaMoeNumericsOptions(
                    swiglu_limit=limit, clamp_after_silu=clamp_after, apply_probs_before_fc2=True,
                ),
            )
            backends[key] = MegaMoeBackend(executor, local_token_capacity)
        layer.mega_moe_backend = backends[key]
        layer.router.return_selected_probs = True
