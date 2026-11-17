"""Launch with torchrun -m pytest on two A3 ranks (EP) or four (EP plus FSDP)."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.mega_moe import bind_mega_moe
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
from torchturbo.ops import get_or_create_ops, OpPolicy
from torchturbo.runtime import get_device_runtime

pytestmark = pytest.mark.multi_npu


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", 1)) not in (2, 4), reason="requires two or four torchrun ranks")
def test_mega_moe_matches_native_expert_parallel():
    pytest.importorskip("torch_npu")
    from torch.utils.checkpoint import checkpoint

    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    try:
        get_or_create_ops(device_runtime=get_device_runtime(), default_policy=OpPolicy.AUTO)
        mesh = init_device_mesh("npu", (dist.get_world_size() // 2, 2), mesh_dim_names=("edp", "ep"))
        ep_group = mesh["ep"].get_group()
        cfg = MoEConfig(
            hidden_size=512, moe_ffn_hidden_size=128, num_moe_experts=8,
            expert_model_parallel_size=dist.get_world_size(ep_group),
            moe_router_topk=2, moe_router_score_function="sigmoid", moe_router_dtype="fp32",
            activation_func=F.silu, gated_linear_unit=True, params_dtype=torch.bfloat16,
            moe_swiglu_limit=[0.5], moe_swiglu_clamp_gate_after_activation=True,
        )
        submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))
        native = MoELayer(cfg, ep_group, submodules, layer_number=0).npu()
        candidate = MoELayer(cfg, ep_group, submodules, layer_number=0).npu()
        torch.manual_seed(124 + dist.get_rank(ep_group))
        with torch.no_grad():
            for parameter in native.parameters():
                parameter.normal_(std=0.035)
        candidate.load_state_dict(native.state_dict())
        bind_mega_moe(candidate, ep_group, 256)
        if dist.get_world_size() == 4:
            fully_shard(candidate.experts, mesh=mesh["edp"], mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False,
            ))
        assert list(native.state_dict()) == list(candidate.state_dict())

        for rows, recompute in ((128, False), (256, True)):
            x = torch.randn(rows, 1, 512, device="npu", dtype=torch.bfloat16, requires_grad=True)
            other = x.detach().clone().requires_grad_()
            native.zero_grad(set_to_none=True)
            candidate.zero_grad(set_to_none=True)
            expected, _ = native(x)
            actual, _ = checkpoint(candidate, other, use_reentrant=False) if recompute else candidate(other)
            upstream = torch.randn_like(expected)
            expected.backward(upstream)
            actual.backward(upstream)
            pairs = [(actual, expected), (other.grad, x.grad)]
            for name, param in candidate.named_parameters():
                gradient = param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
                pairs.append((gradient, native.get_parameter(name).grad))
            for result, reference in pairs:
                assert result is not None and reference is not None
                torch.testing.assert_close(result, reference, atol=0.012, rtol=0.04)
    finally:
        dist.destroy_process_group()
