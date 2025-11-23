# tests/conftest.py
from pathlib import Path

import pytest

from buc import GlobalConfig
from buc import HardwareStack
from buc import SpurEngine


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="function")
def config_path() -> Path:
    """Path to the main project config."""
    return ROOT / "config.yaml"


@pytest.fixture(scope="function")
def cfg(config_path: Path) -> GlobalConfig:
    """
    Fresh GlobalConfig per test.

    This ensures tests can safely mutate cfg.tiles etc.
    """
    return GlobalConfig.load(str(config_path))


@pytest.fixture(scope="function")
def stack_def(cfg: GlobalConfig) -> dict:
    """First hardware stack definition from the YAML."""
    stacks = cfg.yaml_data["hardware_choices"]["stacks"]
    assert stacks, "Expected at least one hardware stack in config.yaml"
    return stacks[0]


@pytest.fixture(scope="function")
def hw_stack(cfg: GlobalConfig, stack_def: dict) -> HardwareStack:
    """Concrete HardwareStack for tests."""
    return HardwareStack(cfg, stack_def)


@pytest.fixture(scope="function")
def spur_engine(cfg: GlobalConfig, hw_stack: HardwareStack) -> SpurEngine:
    """
    Fully wired SpurEngine.

    Tests that need speed can shrink cfg.tiles before using this.
    """
    return SpurEngine(cfg, hw_stack)
