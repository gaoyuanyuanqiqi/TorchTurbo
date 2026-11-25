  # Adapted from NVIDIA Megatron-LM:
  # https://github.com/NVIDIA/Megatron-LM
  #
  # Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
  # Licensed under the BSD 3-Clause License.
  #
  # Modified by the TorchTurbo contributors.

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from functools import partial
from dataclasses import dataclass, field, replace
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torchturbo.logging import get_logger
from torchturbo.ops import ops

from .moe_config import MoEConfig
from .router import TopKRouter
from .token_dispatcher import MoETokenDispatcher, MoEAlltoAllTokenDispatcher
from .module import ModuleSpec, build_module


logger = get_logger(__name__)

CheckpointContext = Callable[[bool], AbstractContextManager[None]]


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None

    # Each model preparation gets its own cache. Providers choose compatibility keys.
    megakernel_cache: dict = field(default_factory=dict, repr=False, compare=False, kw_only=True)


@dataclass
class MoERuntimeInputs:
    padding_mask: torch.Tensor | None = None
    # False while recomputation replays a forward that already counted its tokens.
    accumulate_tokens_per_expert: bool = True


class BaseMoELayer(torch.nn.Module, ABC):
    """Base class for a mixture of experts layer."""

    expert_checkpoint_context: Callable[[Callable[[], object]], CheckpointContext] | None = None

    def __init__(
        self,
        config: MoEConfig,
        ep_group: dist.ProcessGroup,
        layer_number: Optional[int] = None,
    ) -> None:
        super().__init__()

        ep_size = dist.get_world_size(ep_group)
        ep_rank = dist.get_rank(ep_group)

        self.config = config
        self.ep_group = ep_group
        self.expert_parallel_size = ep_size
        assert (
            self.expert_parallel_size > 0
        ), "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = (
            self.config.num_moe_experts // self.expert_parallel_size
        )
        local_expert_indices_offset = ep_rank * self.num_local_experts

        self.use_shared_expert = (
            self.config.moe_shared_expert_intermediate_size is not None
        )

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(
            map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices)
        )
        self.router: TopKRouter = None
        self.experts = None
        self.shared_experts = None
        self.shared_expert = None
        self.shared_expert_gate = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number
        self._runtime_inputs = MoERuntimeInputs()

    def set_runtime_inputs(self, inputs: MoERuntimeInputs) -> None:
        self._runtime_inputs = inputs

    def prepare_checkpoint_context(
        self, block: torch.nn.Module, use_reentrant: bool,
        parameter_version: Callable[[torch.nn.Module], object],
    ) -> Callable[[], CheckpointContext]:
        """Capture this layer's runtime inputs and optional expert replay state."""
        single_layer = sum(isinstance(child, BaseMoELayer) for child in block.modules()) == 1
        provider_context = self.expert_checkpoint_context if use_reentrant and single_layer else None

        @contextmanager
        def context(inputs: MoERuntimeInputs, provider: CheckpointContext | None, replay: bool) -> Iterator[None]:
            with ExitStack() as stack:
                if replay:
                    stack.enter_context(restore_moe_runtime_inputs(((self, inputs),)))
                if provider is not None:
                    stack.enter_context(provider(replay))
                yield

        def capture() -> CheckpointContext:
            inputs = self._runtime_inputs if use_reentrant else replace(
                self._runtime_inputs, accumulate_tokens_per_expert=False,
            )
            provider = None
            if provider_context is not None:
                provider = provider_context(lambda: parameter_version(self.experts))
            return partial(context, inputs, provider)

        return capture

    @abstractmethod
    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward method for the MoE layer."""
        pass


class MoELayer(BaseMoELayer):
    """
    Mixture of experts Layer **currently only supports no token dropping**.
    """

    def __init__(
        self,
        config: MoEConfig,
        ep_group: dist.ProcessGroup,
        submodules: Optional[MoESubmodules] = None,
        use_te: bool = False,
        layer_number: Optional[int] = None,
        sp_group: dist.ProcessGroup | None = None,
    ) -> None:
        self.use_te = use_te
        self.submodules = submodules
        super(MoELayer, self).__init__(
            config=config, ep_group=ep_group, layer_number=layer_number
        )

        # Initialize router
        self.router = TopKRouter(
            config=self.config,
            layer_number=self.layer_number,
            sp_group=sp_group,
        )

        if not config.mega_moe.enable:
            if config.moe_token_dispatcher_type != "alltoall":
                raise ValueError(f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}")
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=config, ep_group=ep_group,
            )

        # Initialize experts
        self.experts = build_module(
            self.submodules.experts,
            self.num_local_experts,
            self.use_te,
            self.config,
            layer_number=self.layer_number,
        )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(
                self.submodules.shared_experts, config=self.config
            )

        self.megakernel = ops.mega_moe(self) if config.mega_moe.enable else None
        if self.megakernel is not None and config.mega_moe.expert_balance is not None:
            prepare_balance = getattr(self.megakernel, "prepare_expert_balance", None)
            if prepare_balance is None:
                logger.warning_rank0_once(
                    "%s ignores expert_balance=%s and will use ordinary MegaMoE.",
                    type(self.megakernel).__name__, config.mega_moe.expert_balance,
                )
            else:
                self.expert_checkpoint_context = prepare_balance()

    def _forward_native_routed(
        self, hidden_states: torch.Tensor, probs: torch.Tensor,
        routing_map: torch.Tensor, expert_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        dispatcher = self.token_dispatcher
        # Reuse stage variables so completed layouts are not kept alive here.
        expert_tokens, expert_probs = dispatcher.dispatch_preprocess(
            hidden_states, routing_map, probs, expert_indices=expert_indices,
        )
        expert_tokens, expert_probs = dispatcher.token_dispatch(expert_tokens, expert_probs)
        expert_tokens, tokens_per_expert, expert_probs = dispatcher.dispatch_postprocess(
            expert_tokens, expert_probs,
        )
        expert_tokens, bias = self.experts(expert_tokens, tokens_per_expert, expert_probs)
        assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"
        expert_tokens = dispatcher.combine_preprocess(expert_tokens)
        expert_tokens = dispatcher.token_combine(expert_tokens)
        expert_tokens = dispatcher.combine_postprocess(expert_tokens)
        return expert_tokens.to(hidden_states.dtype)

    def add_shared_experts(self, hidden_states: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Add this layer's shared expert contributions to the routed output."""
        if self.shared_expert is not None:
            shared_expert_output = self.shared_expert(hidden_states)
            shared_expert_output = (
                F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
            )
            output = output + shared_expert_output
        if self.shared_experts is not None:
            output = output + self.shared_experts(hidden_states)
        return output

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        runtime = self._runtime_inputs
        probs, routing_map, expert_indices, logits = self.router(
            hidden_states, input_ids, runtime.padding_mask, runtime.accumulate_tokens_per_expert,
        )
        if self.megakernel is None:
            output = self._forward_native_routed(hidden_states, probs, routing_map, expert_indices)
            output = self.add_shared_experts(hidden_states, output)
        else:
            output = self.megakernel.forward_experts(
                hidden_states, probs, routing_map, expert_indices=expert_indices,
            )
        return output, logits.view(-1, logits.shape[-1])



@contextmanager
def restore_moe_runtime_inputs(
    snapshot: tuple[tuple[BaseMoELayer, MoERuntimeInputs], ...],
) -> Iterator[None]:
    """Use captured routing inputs and restore the caller's inputs on exit."""
    with ExitStack() as stack:
        for moe, inputs in snapshot:
            stack.callback(moe.set_runtime_inputs, moe._runtime_inputs)
            moe.set_runtime_inputs(inputs)
        yield
