"""
Unit Tests for the AHDF5 chunking tool(s)
"""

import pytest
import h5py
import tempfile
import os
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
