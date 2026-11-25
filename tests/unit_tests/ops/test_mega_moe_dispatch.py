"""CPU checks for HMC provider selection and live expert parameter ownership."""

import sys
import weakref
from types import SimpleNamespace

import pytest
import torch

from torchturbo.ops import OpDispatcher, OpPolicy, OpRegistry
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoERuntimeInputs, MoESubmodules
from torchturbo.ops.providers.npu import mega_moe as hmc_provider
from torchturbo.runtime import TurboDevice


@pytest.fixture
def hmc(monkeypatch):
    constructed = []

    class Executor(torch.nn.Module):
        def __init__(self, **configuration):
            super().__init__()
            self.configuration = configuration
            self.weights = []
            constructed.append(self)

        def forward(self, inputs, ids, probs, *, expert_weights):
            assert ids.dtype == torch.int32
            self.weights.append(expert_weights)
            scale = sum(weight.float().sum() for weight in expert_weights)
            return inputs * (scale * probs.sum(dim=-1, keepdim=True)).to(inputs.dtype)

    monkeypatch.setitem(
        sys.modules, "hyper_multicore",
        SimpleNamespace(MegaMoeExperts=Executor, MegaMoeNumericsOptions=SimpleNamespace, SwiGLUParameters=SimpleNamespace),
    )
    monkeypatch.setattr(hmc_provider, "find_spec", lambda name: object())
    monkeypatch.setattr(
        OpDispatcher, "_load_provider_impl_classes",
        lambda self, device: {"hmc_mega_moe": hmc_provider.HmcMegaMoe},
    )
    return constructed


def dispatcher(policy=OpPolicy.AUTO, *, overrides=None, device=TurboDevice.NPU):
    registry = OpRegistry()
    registry.register(hmc_provider.HmcMegaMoe.op_spec())
    return OpDispatcher(
        device_runtime=SimpleNamespace(turbo_device=device),
        registry=registry, default_policy=policy, overrides=overrides,
    )


def inputs(group=None, submodules=None):
    cfg = MoEConfig(hidden_size=2, moe_ffn_hidden_size=1, num_moe_experts=4,
                    moe_router_topk=2, gated_linear_unit=True, activation_func=torch.nn.functional.silu,
                    moe_swiglu_limit=[0.5], moe_swiglu_clamp_gate_after_activation=True,
                    params_dtype=torch.float32)
    x = torch.ones(2, 2, dtype=torch.bfloat16, requires_grad=True)
    ids = torch.tensor([[0, 1], [2, 3]])
    probs = torch.full((2, 2), 0.5, requires_grad=True)
    layer = MoELayer.__new__(MoELayer)
    torch.nn.Module.__init__(layer)
    layer.shared_expert = layer.shared_experts = None
    layer.config = cfg
    layer.submodules = submodules or MoESubmodules()
    layer.ep_group = object() if group is None else group
    layer.expert_parallel_size = 2
    layer.experts = GroupedMLP(2, False, cfg, layer_number=0)
    with torch.no_grad():
        layer.experts.weight1.fill_(1)
        layer.experts.weight2.fill_(1)

    class DenseRouter(torch.nn.Module):
        def forward(self, hidden, input_ids, padding_mask, accumulate):
            assert hidden is x
            dense = probs.new_zeros(2, 4).scatter(1, ids, probs)
            return dense, dense.bool(), ids, probs

    layer.router = DenseRouter()
    layer._runtime_inputs = MoERuntimeInputs()
    layer.megakernel = None
    return layer, x, probs


@pytest.mark.parametrize("policy", [OpPolicy.AUTO, OpPolicy.CUSTOM])
def test_dispatch_keeps_live_weights_and_reuses_executor(hmc, policy):
    op = dispatcher(policy)
    layer, x, probs = inputs()
    assert not hmc
    layer.megakernel = op.mega_moe(layer)
    result, logits = layer(x)
    torch.testing.assert_close(logits, probs)
    result.float().sum().backward()
    assert all(v.grad is not None for v in (x, probs, layer.experts.weight1, layer.experts.weight2))
    assert len(hmc) == 1
    assert hmc[0].configuration["ep_group"] is layer.ep_group
    assert vars(hmc[0].configuration["numerics_options"]) == {
        "apply_probs_before_fc2": True,
    }
    assert vars(hmc[0].configuration["activation"]) == {
        "clamp_limit": 0.5, "clamp_after_silu": True,
    }
    with torch.no_grad():
        layer.experts.weight1.fill_(2)
    layer.experts.eval()
    second, _ = layer(x)
    assert len(hmc) == 1 and not hmc[0].training
    torch.testing.assert_close(second, torch.full_like(second, 20))
    assert hmc[0].weights[-1][0].shape == (2, 2, 2)
    sibling, y, _ = inputs(layer.ep_group, layer.submodules)
    sibling.megakernel = op.mega_moe(sibling)
    sibling_output, _ = sibling(y)
    torch.testing.assert_close(sibling_output, result.detach())
    assert len(hmc) == 1 and hmc[0].training
    other, z, _ = inputs(layer.ep_group)
    other.megakernel = op.mega_moe(other)
    other(z)
    assert len(hmc) == 2


def test_hmc_adds_each_shared_contribution_once(hmc):
    layer, x, probs = inputs()
    layer.megakernel = dispatcher().mega_moe(layer)
    routed, _ = layer(x)
    layer.shared_expert = torch.nn.Identity()
    layer.shared_expert_gate = torch.nn.Linear(2, 1, bias=False, dtype=x.dtype)
    layer.shared_experts = torch.nn.Linear(2, 2, bias=False, dtype=x.dtype)
    with torch.no_grad():
        layer.shared_expert_gate.weight.zero_()
        layer.shared_experts.weight.fill_(1)
    output, logits = layer(x)
    torch.testing.assert_close(output, routed + 0.5 * x + 2 * x)
    torch.testing.assert_close(logits, probs)
    output.sum().backward()
    assert layer.shared_expert_gate.weight.grad is not None
    assert layer.shared_experts.weight.grad is not None


def test_hmc_backend_does_not_retain_its_layer(hmc):
    layer, _, _ = inputs()
    layer.megakernel = dispatcher().mega_moe(layer)
    owner = weakref.ref(layer)
    del layer
    assert owner() is None


def test_fused_backend_returns_complete_output_without_layer_shared_call():
    layer, x, probs = inputs()

    class FusedBackend:
        def forward_experts(self, hidden_states, weights, mask, *, expert_indices=None):
            return hidden_states + 7

    layer.megakernel = FusedBackend()
    layer.shared_experts = torch.nn.Identity()
    output, logits = layer(x)
    torch.testing.assert_close(output, x + 7)
    torch.testing.assert_close(logits, probs)


def test_native_policy_rejects_hmc(hmc):
    with pytest.raises(RuntimeError, match="policy=native"):
        dispatcher(OpPolicy.NATIVE).mega_moe(inputs()[0])
    assert not hmc


@pytest.mark.parametrize("device", [TurboDevice.CUDA, TurboDevice.CPU])
@pytest.mark.parametrize("policy", [OpPolicy.AUTO, OpPolicy.CUSTOM, OpPolicy.NATIVE])
@pytest.mark.parametrize("selection", ["default", "explicit", "override"])
def test_unsupported_device_rejects_hmc_provider(hmc, device, policy, selection):
    overrides = {"mega_moe": "hmc_mega_moe"} if selection == "override" else None
    impl = "hmc_mega_moe" if selection == "explicit" else None
    with pytest.raises(RuntimeError, match=f"device={device.value}"):
        dispatcher(policy, overrides=overrides, device=device).mega_moe(
            inputs()[0], impl=impl,
        )
    assert not hmc


@pytest.mark.parametrize("policy", [OpPolicy.AUTO, OpPolicy.CUSTOM, OpPolicy.NATIVE])
def test_cuda_has_no_mega_moe_implementation(policy):
    from torchturbo.ops.registry import registry

    op = OpDispatcher(device_runtime=SimpleNamespace(turbo_device=TurboDevice.CUDA),
                      registry=registry, default_policy=policy)
    with pytest.raises(RuntimeError, match="No implementation for op='mega_moe', device=cuda"):
        op.mega_moe(inputs()[0])


def test_override_and_explicit_selection(hmc):
    layer, x, _ = inputs()
    op = dispatcher(OpPolicy.NATIVE, overrides={"mega_moe": "hmc_mega_moe"})
    layer.megakernel = op.mega_moe(layer)
    layer(x)
    layer.megakernel = dispatcher(OpPolicy.NATIVE).mega_moe(layer, impl="hmc_mega_moe")
    layer(x)
    assert len(hmc) == 1
    with pytest.raises(RuntimeError, match="Invalid override"):
        dispatcher(overrides={"mega_moe": "missing"})


def test_unavailable_hmc_fails_without_initializing(hmc, monkeypatch):
    monkeypatch.setattr(hmc_provider, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="No implementation"):
        dispatcher().mega_moe(inputs()[0])
    with pytest.raises(RuntimeError, match="Invalid override"):
        dispatcher(overrides={"mega_moe": "hmc_mega_moe"})
    assert not hmc


@pytest.mark.parametrize("invalid", ["activation", "weights"])
def test_invalid_inputs_fail_before_workspace_creation(hmc, invalid):
    layer, x, _ = inputs()
    if invalid == "activation":
        x.data = x.data.float()
    else:
        layer.experts.double()
    with pytest.raises(ValueError):
        layer.megakernel = dispatcher().mega_moe(layer)
        layer(x)
    assert not hmc


def test_hmc_rejects_te_before_accessing_grouped_weights(hmc):
    layer, x, _ = inputs()
    layer.experts.use_te = True
    with pytest.raises(ValueError, match="Transformer Engine"):
        layer.megakernel = dispatcher().mega_moe(layer)
        layer(x)
    assert not hmc


@pytest.mark.parametrize("enabled", [False, True])
def test_hot_replica_only_configures_hmc_executor(hmc, monkeypatch, enabled):
    from unittest.mock import Mock

    planner = Mock()
    monkeypatch.setattr(sys.modules["hyper_multicore"], "MegaMoePlacementOptions", SimpleNamespace, raising=False)
    monkeypatch.setitem(sys.modules, "hyper_multicore.modules.hot_replica.planner",
                        SimpleNamespace(HotReplicaPlanner=planner))
    layer, x, _ = inputs()
    layer.config.mega_moe.expert_balance = "hot_replica" if enabled else None
    op = dispatcher()
    layer.megakernel = op.mega_moe(layer)
    layer(x)
    if enabled:
        planner.assert_called_once_with(source_token_bound=None)
        options = hmc[0].configuration["expert_placement"]
        assert options.strategy == "adaptive" and options.planner is planner.return_value
        plain, y, _ = inputs(layer.ep_group, layer.submodules)
        plain.megakernel = op.mega_moe(plain)
        plain(y)
        assert len(hmc) == 2 and "expert_placement" not in hmc[1].configuration
    else:
        planner.assert_not_called()
        assert "expert_placement" not in hmc[0].configuration


def test_hot_replica_prepares_distinct_executors_before_routing(hmc, monkeypatch):
    from unittest.mock import Mock

    planner = Mock()
    monkeypatch.setattr(sys.modules["hyper_multicore"], "MegaMoePlacementOptions", SimpleNamespace, raising=False)
    monkeypatch.setitem(sys.modules, "hyper_multicore.modules.hot_replica.planner",
                        SimpleNamespace(HotReplicaPlanner=planner))
    first, x, _ = inputs()
    second, _, _ = inputs(first.ep_group, first.submodules)
    third, _, _ = inputs(first.ep_group, first.submodules)
    replay_checkpoint = Mock()
    monkeypatch.setitem(sys.modules, "hyper_multicore.modules.hot_replica.checkpoint",
                        SimpleNamespace(ReplayCheckpoint=replay_checkpoint))
    second.experts.swiglu_limit = 7.0
    op = dispatcher()
    for layer in (first, second, third):
        layer.config.mega_moe.expert_balance = "hot_replica"
        layer.config.mega_moe._source_token_bound = 32
        layer.megakernel = op.mega_moe(layer)
        replay = layer.megakernel.prepare_expert_balance()

    replay(first.parameters)
    replay_checkpoint.assert_called_once_with(first.parameters, prepare_replay=True)

    assert len(hmc) == 2
    assert planner.call_count == 2
    planner.assert_called_with(source_token_bound=32)
    first(x)
    assert len(hmc) == 2


def test_hmc_uses_mask_only_when_router_omits_ids(hmc, monkeypatch):
    from unittest.mock import Mock

    extract_ids = Mock(wraps=hmc_provider.get_expert_indices)
    monkeypatch.setattr(hmc_provider, "get_expert_indices", extract_ids)
    layer, x, _ = inputs()
    weights, mask, ids, _ = layer.router(x, None, None, False)
    backend = dispatcher().mega_moe(layer)
    expected = backend.forward_experts(x, weights, mask, expert_indices=ids)
    extract_ids.assert_not_called()
    actual = backend.forward_experts(x, weights, mask)
    extract_ids.assert_called_once_with(mask)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("impl", [None, "other_mega_moe"])
def test_mega_moe_dispatches_layer_to_selected_backend(monkeypatch, impl):
    from unittest.mock import Mock

    op = dispatcher(device=TurboDevice.CUDA)
    backend = object()

    layer, _, _ = inputs()
    build = Mock(return_value=backend)
    select = Mock(return_value=build)
    monkeypatch.setattr(op, "_candidate_impl", select)
    assert op.mega_moe(layer, impl=impl) is backend
    build.assert_called_once_with(layer)
    select.assert_called_once_with("mega_moe", TurboDevice.CUDA, impl=impl)


def test_hmc_rejects_invalid_balance_before_creating_executor(hmc):
    layer, x, _ = inputs()
    layer.config.mega_moe.expert_balance = "invalid"
    with pytest.raises(ValueError, match="HMC MegaMoE expert_balance"):
        dispatcher().mega_moe(layer)
    assert not hmc
