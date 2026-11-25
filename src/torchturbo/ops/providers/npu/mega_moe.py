"""HyperMulticore expert-parallel SwiGLU on Ascend A3."""

from __future__ import annotations

from collections.abc import Callable
from importlib.util import find_spec
from typing import TYPE_CHECKING
from weakref import proxy

import torch
import torch.nn.functional as F

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...moe import get_expert_indices
from ...registry import register_op

if TYPE_CHECKING:
    from hyper_multicore import MegaMoeExperts
    from torchturbo.models.modules.moe.experts import GroupedMLP
    from torchturbo.models.modules.moe.moe_layer import CheckpointContext, MoELayer


@register_op(
    op_name="mega_moe",
    impl_name="hmc_mega_moe",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class HmcMegaMoe(OpImpl):
    def available(self) -> bool:
        return find_spec("hyper_multicore") is not None

    def __call__(self, layer: MoELayer) -> HmcRoutedExperts:
        variants = layer.submodules.megakernel_cache.setdefault(type(self), {})
        return HmcRoutedExperts(layer, variants=variants)


class HmcRoutedExperts:
    """Per-layer expert modules using a shared HMC executor for compatible shapes."""

    def __init__(self, layer: MoELayer, *, variants: dict[tuple, _HmcExpertCall]) -> None:
        from torchturbo.models.modules.moe.experts import GroupedMLP

        if not isinstance(layer.experts, GroupedMLP):
            raise ValueError("HMC MegaMoE requires TorchTurbo GroupedMLP experts.")
        experts = layer.experts
        cfg = layer.config
        if experts.use_te:
            raise ValueError("HMC MegaMoE requires GroupedMLP without Transformer Engine.")
        if layer.ep_group is None or layer.expert_parallel_size <= 1:
            raise ValueError("HMC MegaMoE requires expert parallelism with ep_size > 1.")
        if experts.num_local_experts * layer.expert_parallel_size != cfg.num_moe_experts:
            raise ValueError("MegaMoE expert partition does not match the EP process group.")
        if not cfg.gated_linear_unit or cfg.activation_func is not F.silu or cfg.moe_apply_probs_on_input:
            raise ValueError("MegaMoE requires SwiGLU with routing probabilities applied before FC2.")
        expert_balance = cfg.mega_moe.expert_balance
        if expert_balance not in (None, "hot_replica"):
            raise ValueError("HMC MegaMoE expert_balance must be null or 'hot_replica'.")
        limit = experts.swiglu_limit or None
        key = (layer.ep_group, cfg.hidden_size, cfg.moe_ffn_hidden_size, cfg.num_moe_experts,
               cfg.moe_router_topk, limit, bool(limit and cfg.moe_swiglu_clamp_gate_after_activation), expert_balance,
               cfg.mega_moe._source_token_bound)
        run = variants.get(key)
        if run is None:
            run = variants[key] = _HmcExpertCall(key)
        self._run = run
        self._experts = experts
        # Model adapters may install gated shared experts after layer construction.
        self._shared_expert_owner = proxy(layer)

    def prepare_expert_balance(self) -> Callable[[Callable[[], object]], CheckpointContext]:
        """Create bounded workspaces before routing and enable checkpoint replay."""
        if self._experts.config.mega_moe._source_token_bound is not None:
            self._run.executor()
        return self.checkpoint_context

    @staticmethod
    def checkpoint_context(parameter_version: Callable[[], object]) -> CheckpointContext:
        """Create replay state for one checkpoint invocation."""
        from hyper_multicore.modules.hot_replica.checkpoint import ReplayCheckpoint

        return ReplayCheckpoint(parameter_version, prepare_replay=True).scope

    def forward_experts(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor,
        *, expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self._experts.config
        rows = hidden_states.numel() // cfg.hidden_size
        if probs.shape != (rows, cfg.num_moe_experts):
            raise ValueError("HMC MegaMoE requires dense router weights shaped [tokens, experts].")
        ids = expert_indices
        if ids is None:
            ids = get_expert_indices(routing_map).reshape(rows, cfg.moe_router_topk)
        if ids.dtype not in (torch.int32, torch.int64) or ids.shape != (rows, cfg.moe_router_topk):
            raise ValueError("HMC MegaMoE requires integer expert IDs shaped [tokens, topk].")
        probs = probs.gather(1, ids.long())
        output, _ = self._experts(hidden_states, None, probs, routed_forward=self._run, expert_indices=ids)
        return self._shared_expert_owner.add_shared_experts(hidden_states, output)


class _HmcExpertCall:
    """Keep one executor per compatible expert configuration."""

    def __init__(self, key: tuple) -> None:
        self._key = key
        self._executor: MegaMoeExperts | None = None

    def __call__(self, experts: GroupedMLP, inputs: torch.Tensor,
                 ids: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        weight1, weight2 = experts.weight1, experts.weight2
        if inputs.dtype != torch.bfloat16:
            raise ValueError("MegaMoE requires BF16 expert activations.")
        if any(w.dtype not in (torch.bfloat16, torch.float32) for w in (weight1, weight2)):
            raise ValueError("MegaMoE requires BF16 expert weights or FP32 master parameters.")
        executor = self.executor()
        cfg = experts.config
        local = experts.num_local_experts
        weights = (
            weight1.to(inputs.dtype).view(local, cfg.hidden_size, 2 * cfg.moe_ffn_hidden_size),
            weight2.to(inputs.dtype).view(local, cfg.moe_ffn_hidden_size, cfg.hidden_size),
        )
        if executor.training != experts.training:
            executor.train(experts.training)
        return executor(inputs, ids.int(), probs, expert_weights=weights)

    def executor(self) -> MegaMoeExperts:
        if self._executor is None:
            self._executor = self._create_executor(*self._key)
        return self._executor

    @staticmethod
    def _create_executor(
        ep_group, hidden_size, intermediate_size, num_experts, top_k, limit, clamp_after_silu, expert_balance, source_token_bound,
    ) -> MegaMoeExperts:
        from hyper_multicore import MegaMoeExperts, MegaMoeNumericsOptions, SwiGLUParameters

        options = {}
        if expert_balance == "hot_replica":
            from hyper_multicore import MegaMoePlacementOptions
            from hyper_multicore.modules.hot_replica.planner import HotReplicaPlanner

            options["expert_placement"] = MegaMoePlacementOptions(
                strategy="adaptive", planner=HotReplicaPlanner(source_token_bound=source_token_bound),
            )

        return MegaMoeExperts(
            hidden_size=hidden_size, intermediate_size=intermediate_size,
            num_experts=num_experts, top_k=top_k, ep_group=ep_group,
            activation=SwiGLUParameters(clamp_limit=limit, clamp_after_silu=clamp_after_silu),
            numerics_options=MegaMoeNumericsOptions(apply_probs_before_fc2=True),
            **options,
        )
