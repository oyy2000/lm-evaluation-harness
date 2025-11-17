# lm_eval/models/steer_hf.py
from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Optional, List
from dataclasses import is_dataclass, replace

import numpy as np
import torch

# 关键：这版 harness 用 register_model 装饰器注册模型
from lm_eval.api.registry import register_model
# 关键：HuggingFace 后端类在 huggingface.py 里，类名是 HFLM
from lm_eval.models.huggingface import HFLM

print("[steer_hf] imported and registering... ………………")

def _find_block_and_mlp(model, layer_id: int):
    """
    兼容常见家族（LLaMA/Mistral/NeoX/GPT2）以返回 (block, mlp)
    如需支持别的架构（Qwen/Phi 等），可在此补路径。
    """
    p = getattr(model, "model", None)
    if p is not None and hasattr(p, "layers"):
        block = p.layers[layer_id]
        mlp = getattr(block, "mlp", None)
        return block, mlp

    p = getattr(model, "gpt_neox", None)
    if p is not None and hasattr(p, "layers"):
        block = p.layers[layer_id]
        mlp = getattr(block, "mlp", None)
        return block, mlp

    p = getattr(model, "transformer", None)
    if p is not None and hasattr(p, "h"):  # GPT-2 style
        block = p.h[layer_id]
        mlp = getattr(block, "mlp", None)
        return block, mlp

    raise AttributeError("Unsupported model family: cannot locate block/mlp")


def _dict_to_argstr(d):
    """
    把 {'use_flash_attention_2': True, 'torch_dtype': 'bfloat16'} 变成
    "use_flash_attention_2=True,torch_dtype=bfloat16"
    字符串值里如有逗号或空白，可加引号；一般无需。
    """
    parts = []
    for k, v in d.items():
        if isinstance(v, bool):
            val = "true" if v else "false"
        else:
            val = str(v)
        parts.append(f"{k}={val}")
    return ",".join(parts)

@register_model("steer_hf")
class SteerHFLM(HFLM):
    def __init__(self, **kwargs):
        # ---- 0) 解析并清理 model_args（兼容 str / dict）----
        raw_ma = kwargs.get("model_args", {}) or {}
        ma = self._parse_model_args(raw_ma)  # -> dict

        # ---- 从顶层 kwargs 和 model_args 两个来源“消费”自定义键 ----
        def _consume(key, caster=None, default=None):
            if key in kwargs:         # 顶层（harness 可能把 k=v 提升到这）
                val = kwargs.pop(key)
            elif key in ma:           # 落在 model_args 里
                val = ma.pop(key)
            else:
                val = default
            if caster is not None and val is not None:
                try:
                    return caster(val)
                except Exception:
                    return default
            return val

        self.steer_layer     = _consume("steer_layer", int, None)
        self.steer_lambda    = _consume("steer_lambda", float, 0.0)
        self.steer_span      = _consume("steer_span", int, 1)
        # 允许两种路径；并支持按扩展名自动识别
        _vec_path            = _consume("steer_vec_path", str, None)
        self.steer_json_path = _consume("steer_json_path", str, None)
        self.steer_key       = _consume("steer_key", str, None)

        if _vec_path and str(_vec_path).lower().endswith(".json"):
            # 用户误把 json 放在 steer_vec_path，这里自动纠正
            self.steer_json_path = _vec_path
            _vec_path = None
        self.steer_vec_path = _vec_path

        # ---- 回填给父类：必须是字符串；若为空则干脆不传 ----
        if isinstance(ma, dict) and len(ma) == 0:
            kwargs.pop("model_args", None)  # 不传，避免下游误透传
        else:
            kwargs["model_args"] = _dict_to_argstr(ma) if isinstance(ma, dict) else ma

        # ---- 1) 交给父类构建 HF 模型 ----
        super().__init__(**kwargs)

        # ---- 2) 后续初始化 ----
        self._enabled = (
            self.steer_layer is not None
            and self.steer_layer >= 0
            and self.steer_lambda != 0.0
            and (self.steer_vec_path is not None or self.steer_json_path is not None)
        )

        self._hidden_size = getattr(self.model.config, "hidden_size", None) or getattr(
            self.model.config, "n_embd", None
        )
        if self._hidden_size is None:
            raise ValueError("Cannot infer hidden size from model.config")

        self._w_cpu = None
        if self._enabled:
            w = self._load_direction_vector()
            if w.shape[0] != self._hidden_size:
                raise ValueError(f"steer vector dim {w.shape[0]} != hidden_size {self._hidden_size}")
            self._w_cpu = torch.tensor(w, dtype=torch.float32, device="cpu", requires_grad=False)

        self._hook_handle = None

    def generate_until(self, requests):
        with self._steering_ctx():
            return super().generate_until(requests)


    # --------- helpers: robust parsing for model_args ----------
    def _parse_model_args(self, ma):
        """Accept str 'k=v,k=v' or dict; return a dict with basic type coercion."""
        if isinstance(ma, dict):
            return dict(ma)  # shallow copy

        if isinstance(ma, str):
            out = {}
            s = ma.strip()
            if not s:
                return out
            parts = [p for p in s.split(",") if p.strip() != ""]
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    out[k] = self._coerce(v)
                else:
                    # 支持无值开关式参数：当作 True
                    out[p.strip()] = True
            return out

        # 其它类型，尽量转成空 dict
        return {}

    def _coerce(self, v: str):
        # 去掉引号
        if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
            v = v[1:-1]

        vl = v.lower()
        if vl == "true":
            return True
        if vl == "false":
            return False

        # 尝试 int / float
        try:
            if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
                return int(v)
            return float(v)
        except Exception:
            return v  # 保留原字符串

    # ---- 向量加载 ----
    def _load_direction_vector(self) -> np.ndarray:
        if self.steer_vec_path:
            try:
                arr = np.load(self.steer_vec_path, allow_pickle=False)
            except Exception as e:
                raise ValueError(f"Failed to load steer_vec_path '{self.steer_vec_path}': {e}")
            return np.asarray(arr, dtype=np.float32).reshape(-1)

        if self.steer_json_path:
            try:
                with open(self.steer_json_path, "r") as f:
                    data = json.load(f)
            except Exception as e:
                raise ValueError(f"Failed to load steer_json_path '{self.steer_json_path}': {e}")
            key = self.steer_key if self.steer_key is not None else str(self.steer_layer)
            if key not in data:
                raise KeyError(
                    f"Key '{key}' not found in {self.steer_json_path}. "
                    f"Available keys (truncated): {list(data)[:5]}"
                )
            vec = np.asarray(data[key], dtype=np.float32).reshape(-1)
            return vec

        raise ValueError("You must provide either steer_vec_path (.npy/.npz) or steer_json_path (+ optional steer_key)")

    # ---- Hook 安装/卸载 ----
    def _register_hook(self):
        if not self._enabled:
            return
        _, mlp = _find_block_and_mlp(self.model, self.steer_layer)
        if mlp is None:
            raise AttributeError("Selected block has no .mlp module")

        steer_lambda = self.steer_lambda
        steer_span = self.steer_span
        w_cpu = self._w_cpu

        def hook_fn(module, inputs, output):
            out = output
            if isinstance(out, tuple):
                y = out[0]
                if not torch.is_tensor(y):
                    return output
                y = y.clone()
                local_w = w_cpu.to(device=y.device, dtype=y.dtype)
                T = y.size(1)
                sl = min(steer_span, T)
                y[:, T - sl :, :] += steer_lambda * local_w
                return (y,) + out[1:]
            elif torch.is_tensor(out):
                y = out.clone()
                local_w = w_cpu.to(device=y.device, dtype=y.dtype)
                T = y.size(1)
                sl = min(steer_span, T)
                y[:, T - sl :, :] += steer_lambda * local_w
                return y
            return output

        self._hook_handle = mlp.register_forward_hook(hook_fn)

    def _remove_hook(self):
        if self._hook_handle is not None:
            try:
                self._hook_handle.remove()
            finally:
                self._hook_handle = None

    @contextmanager
    def _steering_ctx(self):
        if self._enabled:
            self._register_hook()
        try:
            yield
        finally:
            if self._enabled:
                self._remove_hook()

    def loglikelihood(self, requests):
        with self._steering_ctx():
            return super().loglikelihood(requests)

    def loglikelihood_rolling(self, requests):
        with self._steering_ctx():
            return super().loglikelihood_rolling(requests)
