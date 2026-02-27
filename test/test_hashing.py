"""Tests for hashing module."""

import torch
import pytest

from lsh.hashing import hashed_values, partition


class TestHashedValues:
    def test_output_shape(self):
        data = torch.randn(50, 10)
        result = hashed_values(data, no_of_hash=20, feature_size=10)
        assert result.shape == (50, 20)

    def test_deterministic_with_seed(self):
        torch.manual_seed(42)
        data = torch.randn(10, 5)
        torch.manual_seed(0)
        r1 = hashed_values(data, no_of_hash=10, feature_size=5)
        torch.manual_seed(0)
        r2 = hashed_values(data, no_of_hash=10, feature_size=5)
        assert torch.allclose(r1, r2)


class TestPartition:
    def test_basic_partition(self):
        bin_values = torch.randn(20, 10)
        result = partition([0.5], bin_values, no_of_hash=10, random_seed=42)
        assert 0.5 in result
        assert len(result[0.5]) == 20

    def test_all_frames_assigned(self):
        bin_values = torch.randn(100, 50)
        result = partition([0.01], bin_values, no_of_hash=50)
        mapping = result[0.01]
        assert set(mapping.keys()) == set(range(100))

    def test_cluster_values_are_ints(self):
        bin_values = torch.randn(10, 5)
        result = partition([1.0], bin_values, no_of_hash=5)
        for v in result[1.0].values():
            assert isinstance(v, int)
