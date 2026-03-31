"""
Unit Tests for the AHDF5 chunking tool(s)
"""

import pytest
import h5py
import tempfile
import os
import shutil
import numpy as np

from test_helpers import validate_success
from rechunk_dataset import rechunk_dataset


class TestRechunkDataset:
    """Test suite for rechunk_dataset function"""

    def test_rechunk_larger(self):
        """Test making chunks larger (double)"""
        filepath = None
        output_filepath = None
        try:
            # Create temporary file with chunked dataset
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                data = np.arange(10000).reshape(100, 100)
                f.create_dataset('data', data=data, chunks=(10, 10))

            # Make chunks larger
            result = rechunk_dataset(filepath, '/data', chunk_adjustment='larger')

            # Verify success
            validate_success(result)
            assert 'output_filepath' in result
            output_filepath = result['output_filepath']
            assert os.path.exists(output_filepath)

            # Verify original chunks
            assert result['original_chunks'] == '(10, 10)'

            # Verify new chunks are larger (doubled)
            assert result['new_chunks'] == '(20, 20)'

            # Verify data integrity
            with h5py.File(filepath, 'r') as f_in:
                with h5py.File(output_filepath, 'r') as f_out:
                    original_data = f_in['data'][:]
                    rechunked_data = f_out['data'][:]
                    assert np.array_equal(original_data, rechunked_data)

            # Verify actual chunks in output file
            with h5py.File(output_filepath, 'r') as f:
                assert f['data'].chunks == (20, 20)

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)
            if output_filepath and os.path.exists(output_filepath):
                os.unlink(output_filepath)

    def test_rechunk_smaller(self):
        """Test making chunks smaller (half)"""
        filepath = None
        output_filepath = None
        try:
            # Create temporary file with chunked dataset
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                data = np.arange(10000).reshape(100, 100)
                f.create_dataset('data', data=data, chunks=(20, 20))

            # Make chunks smaller
            result = rechunk_dataset(filepath, '/data', chunk_adjustment='smaller')

            # Verify success
            validate_success(result)
            assert 'output_filepath' in result
            output_filepath = result['output_filepath']
            assert os.path.exists(output_filepath)

            # Verify original chunks
            assert result['original_chunks'] == '(20, 20)'

            # Verify new chunks are smaller (halved)
            assert result['new_chunks'] == '(10, 10)'

            # Verify data integrity
            with h5py.File(filepath, 'r') as f_in:
                with h5py.File(output_filepath, 'r') as f_out:
                    original_data = f_in['data'][:]
                    rechunked_data = f_out['data'][:]
                    assert np.array_equal(original_data, rechunked_data)

            # Verify actual chunks in output file
            with h5py.File(output_filepath, 'r') as f:
                assert f['data'].chunks == (10, 10)

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)
            if output_filepath and os.path.exists(output_filepath):
                os.unlink(output_filepath)

    def test_rechunk_invalid_chunk_spec(self):
        """Test that h5repack fails with invalid chunk specification"""
        filepath = None
        output_filepath = None
        try:
            # Create temporary file with chunked dataset
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                data = np.arange(100).reshape(10, 10)
                f.create_dataset('data', data=data, chunks=(5, 5))

            # Try to set invalid chunk dimensions (zero is not allowed)
            # h5repack should fail with a parsing/conversion error
            result = rechunk_dataset(filepath, '/data', chunk_dims='0x0')

            # Verify failure
            assert result['success'] == False
            assert 'error' in result
            # The error should mention h5repack failure
            assert 'h5repack failed' in result['error']

            # Verify output file was not created (or was cleaned up)
            if 'output_filepath' in result:
                output_filepath = result['output_filepath']
                # Should not exist because of cleanup on failure
                assert not os.path.exists(output_filepath)

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)
            if output_filepath and os.path.exists(output_filepath):
                os.unlink(output_filepath)

    def test_rechunk_nonexistent_file(self):
        """Test error when input file doesn't exist."""
        result = rechunk_dataset("/tmp/does_not_exist_ahdf5.h5", "/data", chunk_adjustment="larger")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_rechunk_nonexistent_dataset(self, tmp_path):
        """Test error when dataset path doesn't exist in file."""
        filepath = str(tmp_path / "test.h5")
        with h5py.File(filepath, "w") as f:
            f.create_dataset("data", data=[1, 2, 3])
        result = rechunk_dataset(filepath, "/nonexistent", chunk_adjustment="larger")
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_rechunk_group_not_dataset(self, tmp_path):
        """Test error when object path points to a group, not a dataset."""
        filepath = str(tmp_path / "test.h5")
        with h5py.File(filepath, "w") as f:
            f.create_group("mygroup")
        result = rechunk_dataset(filepath, "/mygroup", chunk_adjustment="larger")
        assert result["success"] is False
        assert "not a dataset" in result["error"]

    def test_rechunk_output_already_exists(self, tmp_path):
        """Test error when output file already exists."""
        filepath = str(tmp_path / "test.h5")
        output = str(tmp_path / "output.h5")
        with h5py.File(filepath, "w") as f:
            f.create_dataset("data", data=np.arange(100).reshape(10, 10), chunks=(5, 5))
        with h5py.File(output, "w") as f:
            pass
        result = rechunk_dataset(filepath, "/data", output_filepath=output, chunk_adjustment="larger")
        assert result["success"] is False
        assert "already exists" in result["error"]

    def test_rechunk_contiguous_dataset_adjustment(self, tmp_path):
        """Test error when applying adjustment to a contiguous (non-chunked) dataset."""
        filepath = str(tmp_path / "test.h5")
        with h5py.File(filepath, "w") as f:
            f.create_dataset("data", data=np.arange(100))
        result = rechunk_dataset(filepath, "/data", chunk_adjustment="larger")
        assert result["success"] is False
        assert "contiguous" in result["error"].lower()

    def test_rechunk_no_option_specified(self, tmp_path):
        """Test error when no rechunk option is specified."""
        filepath = str(tmp_path / "test.h5")
        with h5py.File(filepath, "w") as f:
            f.create_dataset("data", data=np.arange(100), chunks=(10,))
        result = rechunk_dataset(filepath, "/data")
        assert result["success"] is False
        assert "must specify" in result["error"].lower()

    def test_rechunk_make_contiguous(self, tmp_path):
        """Test converting chunked dataset to contiguous layout."""
        filepath = str(tmp_path / "test.h5")
        with h5py.File(filepath, "w") as f:
            f.create_dataset("data", data=np.arange(100).reshape(10, 10), chunks=(5, 5))
        result = rechunk_dataset(filepath, "/data", make_contiguous=True)
        assert result["success"] is True
        assert "contiguous" in result["new_chunks"].lower()
        if os.path.exists(result["output_filepath"]):
            os.unlink(result["output_filepath"])

    def test_rechunk_exact_dims(self, tmp_path):
        """Test rechunking with exact chunk dimensions."""
        filepath = str(tmp_path / "test.h5")
        with h5py.File(filepath, "w") as f:
            f.create_dataset("data", data=np.arange(100).reshape(10, 10), chunks=(5, 5))
        result = rechunk_dataset(filepath, "/data", chunk_dims="2x2")
        assert result["success"] is True
        assert "(2, 2)" in result["new_chunks"]
        with h5py.File(result["output_filepath"], "r") as f:
            assert f["data"].chunks == (2, 2)
        os.unlink(result["output_filepath"])
