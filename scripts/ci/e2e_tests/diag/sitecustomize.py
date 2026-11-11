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

    _state = {"step": 0, "last": {}, "installed": set()}

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
            out = original(self, *args, **kwargs)
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

    def _dump(name, value):
        try:
            local = _local(value).detach().reshape(-1)
            local_value = local[0].item() if local.numel() else "empty"
        except Exception as exc:
            local_value = f"?({exc!r})"
        print(
            f"[ttprobe] FAIL step={_state['step']} name={name} local={local_value}",
            flush=True,
        )
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
