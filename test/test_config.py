"""Tests for configuration management."""

import os
import tempfile

import pytest

from lsh.config import (
    PipelineConfig,
    SOAPConfig,
    HashingConfig,
    IOConfig,
    load_config,
    validate_config,
    save_example_config,
)


class TestDefaults:
    def test_soap_defaults(self):
        cfg = SOAPConfig()
        assert cfg.r_cut == 6.0
        assert cfg.n_max == 4
        assert cfg.l_max == 4
        assert cfg.sigma == 1.0
        assert cfg.rbf == "gto"
        assert cfg.periodic is True

    def test_hashing_defaults(self):
        cfg = HashingConfig()
        assert cfg.n_components == 100
        assert cfg.n_hash == 100
        assert cfg.bin_width == 0.004
        assert cfg.random_seed == 42

    def test_pipeline_defaults(self):
        cfg = PipelineConfig()
        assert cfg.device == "auto"
        assert cfg.start_step == 1
        assert cfg.end_step == 7


class TestValidation:
    def test_valid_config(self):
        cfg = PipelineConfig()
        assert validate_config(cfg) == []

    def test_invalid_rcut(self):
        cfg = PipelineConfig()
        cfg.soap.r_cut = -1.0
        issues = validate_config(cfg)
        assert any("r_cut" in i for i in issues)

    def test_invalid_step_range(self):
        cfg = PipelineConfig(start_step=5, end_step=3)
        issues = validate_config(cfg)
        assert any("start_step" in i for i in issues)

    def test_invalid_device(self):
        cfg = PipelineConfig(device="tpu")
        issues = validate_config(cfg)
        assert any("device" in i for i in issues)

    def test_invalid_bin_width(self):
        cfg = PipelineConfig()
        cfg.hashing.bin_width = 0.0
        issues = validate_config(cfg)
        assert any("bin_width" in i for i in issues)


class TestLoadConfig:
    def test_load_yaml(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            "soap:\n  r_cut: 8.0\nhashing:\n  bin_width: 0.01\ndevice: cpu\n"
        )
        cfg = load_config(str(config_file))
        assert cfg.soap.r_cut == 8.0
        assert cfg.hashing.bin_width == 0.01
        assert cfg.device == "cpu"

    def test_overrides(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text("soap:\n  r_cut: 8.0\n")
        cfg = load_config(str(config_file), overrides={"soap": {"r_cut": 10.0}})
        assert cfg.soap.r_cut == 10.0

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")


class TestSaveExampleConfig:
    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "example.yaml")
        save_example_config(path)
        assert os.path.exists(path)
        cfg = load_config(path)
        assert validate_config(cfg) == []
