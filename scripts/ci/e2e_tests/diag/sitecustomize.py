"""Temporary NPU E2E probe: report where a non-finite backbone value starts.

Loaded only while ``TT_LOSS_DIAG=1``.  The probe keeps device-side scalars
during forward and synchronizes once, when the loss check fails.
"""

from __future__ import annotations

import functools
import importlib.abc
import os
import sys


_ENABLED = os.environ.get("TT_LOSS_DIAG") == "1"


if _ENABLED:
    import torch

    _state = {
        "step": 0,
        "last": {},
        "meta": {},
        "installed": set(),
        "active_layer": None,
        "conv_calls": 0,
        "gather_seq_calls": 0,
    }

    def _rank():
        return os.environ.get("RANK", "?")

    def _world_size():
        return os.environ.get("WORLD_SIZE", "?")

    def _local(tensor):
        try:
            from torch.distributed.tensor import DTensor

            if isinstance(tensor, DTensor):
                return tensor.to_local()
        except Exception:
            pass
        return tensor

    def _health(tensor):
        tensor = _local(tensor).detach()
        finite = torch.isfinite(tensor)
        bad = ~finite
        if tensor.ndim >= 4:
            width = max(int(tensor.shape[-2]) * int(tensor.shape[-1]), 1)
        else:
            width = max(int(tensor.shape[-1]), 1) if tensor.ndim else 1
        bad_rows = bad.reshape(-1, width).any(-1)
        rows = torch.arange(bad_rows.numel(), device=bad_rows.device)
        first = torch.where(bad_rows, rows, rows.new_full((), bad_rows.numel())).min()
        last = torch.where(bad_rows, rows, rows.new_full((), -1)).max()
        absmax = tensor.float().nan_to_num(nan=0.0).abs().max()
        return {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "bad": bad.sum(),
            "nan": torch.isnan(tensor).sum(),
            "posinf": torch.isposinf(tensor).sum(),
            "neginf": torch.isneginf(tensor).sum(),
            "bad_rows": bad_rows.sum(),
            "first": first,
            "last": last,
            "absmax": absmax,
        }

    def _record(key, value):
        if not torch.is_tensor(value):
            return
        try:
            _state["last"][key] = _health(value)
        except Exception as exc:
            _state["last"][key] = {"error": repr(exc)}

    def _record_meta(key, value):
        try:
            _state["meta"][key] = value
        except Exception as exc:
            _state["meta"][key] = {"error": repr(exc)}

    def _first_tensor(args, kwargs):
        if args and torch.is_tensor(args[0]):
            return args[0]
        return kwargs.get("hidden_states")

    def _wrap_class_forward(cls, label):
        original = getattr(cls, "forward", None)
        if original is None or getattr(original, "_tt_diag", False):
            return

        @functools.wraps(original)
        def forward(self, *args, **kwargs):
            layer = getattr(self, "layer_idx", "?")
            _record(f"L{layer}.{label}.in", _first_tensor(args, kwargs))
            previous_layer = _state["active_layer"]
            _state["active_layer"] = layer
            if label == "attn":
                _state["conv_calls"] = 0
                _state["gather_seq_calls"] = 0
            try:
                out = original(self, *args, **kwargs)
            finally:
                _state["active_layer"] = previous_layer
            value = out[0] if isinstance(out, (tuple, list)) else out
            _record(f"L{layer}.{label}.out", value)
            return out

        forward._tt_diag = True
        cls.forward = forward

    def _wrap_attention(cls, *_args, **_kwargs):
        _wrap_class_forward(cls, "attn")

    def _install_model(module):
        _wrap_class_forward(getattr(module, "BailingMoeV3DecoderLayer", None), "layer")
        _wrap_class_forward(
            getattr(module, "BailingMoeV3MultiLatentAttention", None), "attn"
        )
        _wrap_class_forward(
            getattr(module, "BailingMoeV3KimiDeltaAttention", None), "attn"
        )

    def _install_moe(module):
        _wrap_class_forward(getattr(module, "MoELayer", None), "mlp")

    def _install_kda_patch(module):
        original = getattr(module, "patch_kda_forward", None)
        if original is None or getattr(original, "_tt_diag", False):
            return

        @functools.wraps(original)
        def patched(cls, *args, **kwargs):
            result = original(cls, *args, **kwargs)
            _wrap_attention(cls)
            return result

        patched._tt_diag = True
        module.patch_kda_forward = patched

    def _install_ops_dispatcher(module):
        cls = getattr(module, "OpDispatcher", None)
        if cls is None:
            return

        original_conv = getattr(cls, "causal_conv1d", None)
        if original_conv is not None and not getattr(original_conv, "_tt_diag", False):

            @functools.wraps(original_conv)
            def causal_conv1d(self, x, weight, bias=None, **kwargs):
                layer = _state["active_layer"]
                index = _state["conv_calls"]
                _state["conv_calls"] += 1
                name = ("q", "k", "v")[index % 3]
                if layer == 0:
                    _record(f"L0.{name}_conv.in", x)
                out = original_conv(self, x, weight, bias, **kwargs)
                if layer == 0:
                    value = out[0] if isinstance(out, (tuple, list)) else out
                    _record(f"L0.{name}_conv.out", value)
                return out

            causal_conv1d._tt_diag = True
            cls.causal_conv1d = causal_conv1d

        original_kda = getattr(cls, "kda", None)
        if original_kda is not None and not getattr(original_kda, "_tt_diag", False):

            @functools.wraps(original_kda)
            def kda(self, q, k, v, g, beta, **kwargs):
                layer = _state["active_layer"]
                if layer == 0:
                    for name, value in (
                        ("q", q),
                        ("k", k),
                        ("v", v),
                        ("g", g),
                        ("beta", beta),
                    ):
                        _record(f"L0.kda.{name}.in", value)
                    packed = kwargs.get("packed_params")
                    if packed is not None:
                        _record_meta("L0.kda.cu_seqlens", list(packed.cu_seqlens_list))
                        _record_meta(
                            "L0.kda.chunk_indices_64",
                            list(packed.chunk_indices_list["64"]),
                        )
                out = original_kda(self, q, k, v, g, beta, **kwargs)
                if layer == 0:
                    value = out[0] if isinstance(out, (tuple, list)) else out
                    _record("L0.kda.out", value)
                return out

            kda._tt_diag = True
            cls.kda = kda

    def _install_packed_params(module):
        original = getattr(module, "get_packed_params", None)
        if original is None or getattr(original, "_tt_diag", False):
            return

        @functools.wraps(original)
        def wrapper(cu_seqlens, total_tokens, *args, **kwargs):
            packed = original(cu_seqlens, total_tokens, *args, **kwargs)
            if _state["active_layer"] == 0:
                _record_meta("L0.total_tokens", total_tokens)
                if packed is not None:
                    _record_meta("L0.cu_seqlens", list(packed.cu_seqlens_list))
            return packed

        wrapper._tt_diag = True
        module.get_packed_params = wrapper

    def _install_ulysses(module):
        original_seq = getattr(module, "gather_seq_scatter_heads", None)
        if original_seq is not None and not getattr(original_seq, "_tt_diag", False):

            @functools.wraps(original_seq)
            def gather_seq_scatter_heads(x, *args, **kwargs):
                layer = _state["active_layer"]
                index = _state["gather_seq_calls"]
                _state["gather_seq_calls"] += 1
                name = ("q", "k", "v", "g", "beta")[index % 5]
                if layer == 0:
                    _record(f"L0.{name}_gather.in", x)
                out = original_seq(x, *args, **kwargs)
                if layer == 0:
                    _record(f"L0.{name}_gather.out", out)
                return out

            gather_seq_scatter_heads._tt_diag = True
            module.gather_seq_scatter_heads = gather_seq_scatter_heads

        original_heads = getattr(module, "gather_heads_scatter_seq", None)
        if original_heads is not None and not getattr(
            original_heads, "_tt_diag", False
        ):

            @functools.wraps(original_heads)
            def gather_heads_scatter_seq(x, *args, **kwargs):
                layer = _state["active_layer"]
                if layer == 0:
                    _record("L0.out_gather.in", x)
                out = original_heads(x, *args, **kwargs)
                if layer == 0:
                    _record("L0.out_gather.out", out)
                return out

            gather_heads_scatter_seq._tt_diag = True
            module.gather_heads_scatter_seq = gather_heads_scatter_seq

    def _install_fused_norm(module):
        original = getattr(module, "rms_norm_gated", None)
        if original is None or getattr(original, "_tt_diag", False):
            return

        @functools.wraps(original)
        def wrapper(x, g, *args, **kwargs):
            layer = _state["active_layer"]
            if layer == 0:
                _record("L0.o_norm.x.in", x)
                _record("L0.o_norm.g.in", g)
            out = original(x, g, *args, **kwargs)
            if layer == 0:
                value = out[0] if isinstance(out, (tuple, list)) else out
                _record("L0.o_norm.out", value)
            return out

        wrapper._tt_diag = True
        module.rms_norm_gated = wrapper

    def _install_chunk_kda(module):
        original = getattr(module, "chunk_kda_fwd", None)
        if original is None or getattr(original, "_tt_diag", False):
            return

        names = (
            "attn_out",
            "final_state",
            "gk",
            "Aqk",
            "Akk",
            "w",
            "u",
            "qg",
            "kg",
            "v_new",
            "h",
            "initial_state",
        )

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            layer = _state["active_layer"]
            out = original(*args, **kwargs)
            if layer == 0:
                for name, value in zip(names, out):
                    _record(f"L0.raw_kda.{name}", value)
            return out

        wrapper._tt_diag = True
        module.chunk_kda_fwd = wrapper

    def _dump(name, value):
        try:
            local = _local(value).detach().reshape(-1)
            local_value = local[0].item() if local.numel() else "empty"
        except Exception as exc:
            local_value = f"?({exc!r})"
        print(
            f"[ttprobe] FAIL rank={_rank()}/{_world_size()} step={_state['step']} "
            f"name={name} local={local_value}",
            flush=True,
        )
        for key, value in _state["meta"].items():
            print(f"[ttprobe]   {key} {value}", flush=True)
        for key, item in _state["last"].items():
            if "error" in item:
                print(f"[ttprobe]   {key} error={item['error']}", flush=True)
                continue
            values = {
                key: (round(val.item(), 6) if isinstance(val, torch.Tensor) else val)
                for key, val in item.items()
            }
            print(f"[ttprobe]   {key} {values}", flush=True)

    def _install_assert(module):
        original = getattr(module, "assert_finite", None)
        if original is None or getattr(original, "_tt_diag", False):
            return

        @functools.wraps(original)
        def wrapper(value, name):
            if name == "loss":
                _state["step"] += 1
            try:
                return original(value, name)
            except RuntimeError:
                if name == "loss":
                    _dump(name, value)
                raise

        wrapper._tt_diag = True
        module.assert_finite = wrapper

    def _install_trainer(module):
        try:
            from torchturbo.engine import distributed

            module.assert_finite = distributed.assert_finite
        except Exception:
            pass

    _INSTALLERS = {
        "torchturbo.models.bailing_hybrid.modeling_bailing_moe_v3": _install_model,
        "torchturbo.models.bailing_hybrid.attention": _install_kda_patch,
        "torchturbo.models.modules.moe.moe_layer": _install_moe,
        "torchturbo.models.modules.attention.ulysses.ulysses": _install_ulysses,
        "torchturbo.ops.dispatcher": _install_ops_dispatcher,
        "torchturbo.ops.packed_params": _install_packed_params,
        "fla_npu.ops.ascendc": _install_chunk_kda,
        "fla.modules.fused_norm_gate": _install_fused_norm,
        "torchturbo.engine.distributed": _install_assert,
        "torchturbo.adapters.llamafactory_v0.trainer": _install_trainer,
    }

    class _After:
        def __init__(self, loader, installer):
            self.loader = loader
            self.installer = installer

        def create_module(self, spec):
            return self.loader.create_module(spec)

        def exec_module(self, module):
            self.loader.exec_module(module)
            try:
                self.installer(module)
            except Exception as exc:
                print(f"[ttprobe] install failed: {exc!r}", flush=True)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            installer = _INSTALLERS.get(fullname)
            if installer is None or fullname in _state["installed"]:
                return None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                try:
                    spec = finder.find_spec(fullname, path, target)
                except Exception:
                    continue
                if spec is not None and spec.loader is not None:
                    _state["installed"].add(fullname)
                    spec.loader = _After(spec.loader, installer)
                    return spec
            return None

    sys.meta_path.insert(0, _Finder())
    print("[ttprobe] enabled", flush=True)
