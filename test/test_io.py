"""Tests for I/O module."""

import os
import pytest

from lsh.io import (
    update_xyz_metadata,
    extract_frame_numbers_from_bins,
    extract_frames_from_xyz,
    split_xyz,
)


SAMPLE_XYZ = """\
3
energy=-1.0
C 0.0 0.0 0.0
H 1.0 0.0 0.0
H 0.0 1.0 0.0
3
energy=-2.0
C 1.0 1.0 1.0
H 2.0 1.0 1.0
H 1.0 2.0 1.0
3
energy=-3.0
C 2.0 2.0 2.0
H 3.0 2.0 2.0
H 2.0 3.0 2.0
"""


@pytest.fixture
def sample_xyz(tmp_path):
    path = tmp_path / "sample.xyz"
    path.write_text(SAMPLE_XYZ)
    return str(path)


class TestUpdateMetadata:
    def test_frame_count(self, sample_xyz, tmp_path):
        out = str(tmp_path / "out.xyz")
        count = update_xyz_metadata(sample_xyz, out)
        assert count == 3

    def test_metadata_format(self, sample_xyz, tmp_path):
        out = str(tmp_path / "out.xyz")
        update_xyz_metadata(sample_xyz, out)
        with open(out) as f:
            lines = f.readlines()
        # Second line should contain i = 0
        assert "i =" in lines[1]
        assert "time =" in lines[1]


class TestExtractFrameNumbers:
    def test_parse_bins(self, tmp_path):
        bins_file = tmp_path / "bins.txt"
        bins_file.write_text(
            "Total Frames: 10\nTotal Bins: 3\n\n"
            "Bin 1: [0, 3, 5]\nBin 2: [1, 4]\nBin 3: [2, 6, 7, 8]\n"
        )
        nums = extract_frame_numbers_from_bins(str(bins_file))
        assert len(nums) == 3
        assert set(nums) == {0, 1, 2}


class TestExtractFrames:
    def test_extract(self, sample_xyz, tmp_path):
        processed = str(tmp_path / "processed.xyz")
        update_xyz_metadata(sample_xyz, processed)

        out = str(tmp_path / "extracted.xyz")
        count = extract_frames_from_xyz(processed, {0, 2}, out)
        assert count == 2


class TestSplitXyz:
    def test_split(self, sample_xyz, tmp_path):
        parts_dir = str(tmp_path / "parts")
        n_files = split_xyz(sample_xyz, frames_per_file=2, output_dir=parts_dir)
        assert n_files == 2
        assert os.path.exists(os.path.join(parts_dir, "part_1.xyz"))
        assert os.path.exists(os.path.join(parts_dir, "part_2.xyz"))
