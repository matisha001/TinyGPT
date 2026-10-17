"""统一配置体系：YAML + dataclass。

设计原则：模型尺寸、数据路径、训练超参全部由 YAML 驱动（configs/*.yaml），
代码里零魔法数字；后续扩大模型规模只需修改配置，不动代码。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args, get_origin, get_type_hints

import yaml


# ---------------------------------------------------------------------------
# 递归解析：dict → dataclass
# ---------------------------------------------------------------------------
def _coerce(tp: Any, value: Any) -> Any:
    """把 YAML 值按类型注解递归转换成 dataclass / list / 普通值。"""
    origin = get_origin(tp)
    if origin is None:
        if is_dataclass(tp):
            return _apply(tp, value) if isinstance(value, dict) else tp()
        return value
    if origin in (list, List):
        args = get_args(tp)
        if args and is_dataclass(args[0]):
            return [_coerce(args[0], x) for x in (value or [])]
        return list(value or [])
    if origin in (dict, Dict):
        return dict(value or {})
    # Optional[X] / Union[X, None]
    args = [a for a in get_args(tp) if a is not type(None)]
    if args and is_dataclass(args[0]) and isinstance(value, dict):
        return _apply(args[0], value)
    return value


def _apply(cls: Any, d: dict) -> Any:
    # get_type_hints 解析字符串注解（from __future__ import annotations 下必需）
    hints = get_type_hints(cls)
    known = hints
    extra = set(d or {}) - set(known)
    if extra:
        raise ValueError(f"{cls.__name__} 存在未知配置字段: {sorted(extra)}")
    kwargs = {}
    for name, tp in known.items():
        if name not in d:
            continue
        kwargs[name] = _coerce(tp, d[name])
    return cls(**kwargs)


def load_config(path: Union[str, Path], cls: Any) -> Any:
    """从 YAML 加载配置并转换为指定 dataclass。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _apply(cls, raw)


def as_dict(cfg: Any) -> dict:
    """dataclass → dict（用于 checkpoint 元信息 / 日志）。"""
    return dataclasses.asdict(cfg)


# ---------------------------------------------------------------------------
# 配置结构
# ---------------------------------------------------------------------------
DEFAULT_SPECIAL_TOKENS = [
    "<s>", "</s>", "<pad>", "<unk>",
    "<|user|>", "<|assistant|>", "<|image|>", "<|tool|>",
]
DEFAULT_LORA_TARGETS = ["q", "k", "v", "o", "gate", "up", "down"]


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    targets: List[str] = field(default_factory=lambda: list(DEFAULT_LORA_TARGETS))
    merge: bool = False


@dataclass
class VisionConfig:
    image_size: int = 224
    patch_size: int = 16
    hidden: int = 192
    depth: int = 6
    heads: int = 6
    num_vision_tokens: int = 16
    freeze: bool = True


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    hidden_size: int = 384
    num_layers: int = 8
    num_heads: int = 8
    ffn_size: int = 1024
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1.0e-6
    tie_embeddings: bool = True
    use_lora: bool = False
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    vision: Optional[VisionConfig] = None


@dataclass
class DataConfig:
    corpus_path: str = "data/corpus/pretrain.txt"
    tokenizer_path: str = "data/tokenizer/tokenizer.json"
    sft_path: Optional[str] = None
    dpo_path: Optional[str] = None
    mm_path: Optional[str] = None
    kb_path: Optional[str] = None
    seq_len: int = 256
    batch_size: int = 16
    pack: bool = True
    num_workers: int = 0
    max_samples: Optional[int] = None


@dataclass
class TrainConfig:
    steps: int = 3000
    epochs: int = 3
    grad_accum: int = 4
    lr: float = 3.0e-4
    min_lr: float = 1.0e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 1000
    out_dir: str = "checkpoints/pretrain"
    resume: Optional[str] = None
    mask_prompt_loss: bool = True
    init_from: Optional[str] = None
    # ---- DPO 专用 ----
    dpo_beta: float = 0.1
    dpo_sft_reg: float = 0.2
    ref_logprobs_path: Optional[str] = None


@dataclass
class TrainStageConfig:  # 预训练 / SFT / DPO / 多模态共用的顶层结构
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class TokenizerDataConfig:
    corpus_path: str = "data/corpus/pretrain.txt"
    min_freq: int = 2
    max_tokens: int = 1_000_000


@dataclass
class TokenizerConfig:
    data: TokenizerDataConfig = field(default_factory=TokenizerDataConfig)
    vocab_size: int = 8192
    special_tokens: List[str] = field(default_factory=lambda: list(DEFAULT_SPECIAL_TOKENS))
    out_dir: str = "data/tokenizer"


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    max_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 40


@dataclass
class APIModelConfig:
    model_path: str = "checkpoints/sft/sft.pt"
    tokenizer_path: str = "data/tokenizer/tokenizer.json"
    use_lora: bool = False
    lora_path: Optional[str] = None
    vision: Optional[VisionConfig] = None


@dataclass
class CapabilitiesConfig:
    tools: bool = True
    agent_loop: bool = True
    rag: bool = True
    mcp_servers: List[dict] = field(default_factory=list)
    kb_path: str = "data/kb_index/"


@dataclass
class APIConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    model: APIModelConfig = field(default_factory=APIModelConfig)
    capabilities: CapabilitiesConfig = field(default_factory=CapabilitiesConfig)
