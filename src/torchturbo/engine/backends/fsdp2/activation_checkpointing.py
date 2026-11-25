"""FSDP shard versions and lifecycle hooks for activation recomputation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

from torch import nn

from torchturbo.utils.torch_utils import torch_version

if torch_version() >= (2, 6, 0):
    from torch.distributed.fsdp._fully_shard import FSDPModule
    from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
else:  # pragma: no cover - torch<2.6 compatibility
    from torch.distributed._composable.fsdp._fsdp_common import TrainingState
    from torch.distributed._composable.fsdp.fully_shard import FSDPModule


def _unshard_before_forward(submodule: FSDPModule, _args: tuple[Any, ...]) -> None:
    # FSDP before 2.11 skips unsharding units still marked PRE_BACKWARD,
    # even if an earlier graph already resharded their parameters.
    submodule.unshard()


def _restart_reentrant_forward(submodule: FSDPModule, _args: tuple[Any, ...]) -> None:
    # Reentrant recomputation builds a new autograd graph. FSDP must run
    # its forward hooks again to attach gradient reduction hooks, even
    # if another graph already used this unit in the same backward.
    # Reset only the module hook state; FSDP owns the parameter state.
    submodule._get_fsdp_state()._training_state = TrainingState.IDLE


@contextmanager
def fsdp_recompute_context(module: nn.Module, use_reentrant: bool) -> Iterator[None]:
    """Prepare descendant FSDP units only while recomputation is active.

    Non-reentrant early-stop can skip FSDP's post-forward cleanup, so it must
    unshard without restarting the forward lifecycle. Reentrant recomputation
    needs that lifecycle to attach gradient reduction hooks to its new graph.
    """
    if not use_reentrant and torch_version() >= (2, 11, 0):
        yield
        return
    hook = _restart_reentrant_forward if use_reentrant else _unshard_before_forward
    with ExitStack() as stack:
        # Prepare each nested unit when it runs, keeping normal backward
        # resharding and limiting live full parameters.
        for child in module.modules():
            if child is not module and isinstance(child, FSDPModule):
                handle = child.register_forward_pre_hook(hook, prepend=use_reentrant)
                stack.callback(handle.remove)
        yield


def parameter_version(module: nn.Module) -> tuple[tuple[int, int], ...]:
    """Track the persistent FSDP shards across temporary full-weight restoration."""
    if isinstance(module, FSDPModule):
        group = module._get_fsdp_state()._fsdp_param_group
        parameters = tuple(parameter.sharded_param for parameter in group.fsdp_params)
    else:
        parameters = tuple(module.parameters(recurse=False))
    return tuple((id(parameter), parameter._version) for parameter in parameters)
