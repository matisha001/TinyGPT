"""配置体系单元测试：加载全部 YAML、嵌套结构、非法字段报错。"""
import pytest
from pathlib import Path

from tinygpt.config import (
    APIConfig,
    LoRAConfig,
    TrainStageConfig,
    TokenizerConfig,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]

CONFIGS = {
    "pretrain.yaml": TrainStageConfig,
    "sft.yaml": TrainStageConfig,
    "lora_sft.yaml": TrainStageConfig,
    "dpo.yaml": TrainStageConfig,
    "mm_finetune.yaml": TrainStageConfig,
    "tokenizer.yaml": TokenizerConfig,
    "api.yaml": APIConfig,
}


@pytest.mark.parametrize("name,cls", CONFIGS.items())
def test_load_all_real_configs(name, cls):
    cfg = load_config(ROOT / "configs" / name, cls)
    assert isinstance(cfg, cls)


def test_nested_lora_is_dataclass():
    cfg = load_config(ROOT / "configs" / "pretrain.yaml", TrainStageConfig)
    assert isinstance(cfg.model.lora, LoRAConfig)
    assert cfg.model.lora.rank == 8
    assert "gate" in cfg.model.lora.targets


def test_unknown_field_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  no_such_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="未知配置字段"):
        load_config(bad, TrainStageConfig)


def test_use_lora_switch():
    full = load_config(ROOT / "configs" / "sft.yaml", TrainStageConfig)
    lora = load_config(ROOT / "configs" / "lora_sft.yaml", TrainStageConfig)
    assert full.model.use_lora is False
    assert lora.model.use_lora is True
