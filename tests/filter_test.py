"""
Unit tests for the AHDF5 filter tool(s)
"""

import pytest
import h5py
import tempfile
import os
import numpy as np

from test_helpers import validate_success
from apply_filter_dataset import apply_filter_dataset


class TestApplyFilterDataset:
    """Test suite for apply_filter_dataset function"""

    @pytest.mark.parametrize("filter_config", [
        pytest.param(
            {
                'filter_type': 'gzip',
                'params': {'compression_level': 6},
                'h5z_constant': 'FILTER_DEFLATE',
                'verify_key': 'compression',
                'verify_value': 'gzip',
                'verify_opts_key': 'compression_opts',
                'verify_opts_value': 6,
            },
            id='gzip'
        ),
        pytest.param(
            {
                'filter_type': 'shuffle',
                'params': {},
                'h5z_constant': 'FILTER_SHUFFLE',
                'verify_key': 'shuffle',
                'verify_value': True,
            },
            id='shuffle'
        ),
        pytest.param(
            {
                'filter_type': 'fletcher32',
                'params': {},
                'h5z_constant': 'FILTER_FLETCHER32',
                'verify_key': 'fletcher32',
                'verify_value': True,
            },
            id='fletcher32'
        ),
        pytest.param(
            {
                'filter_type': 'nbit',
                'params': {},
                'h5z_constant': 'FILTER_NBIT',
                'verify_key': None,  # No simple verify for nbit
            },
            id='nbit'
        ),
        pytest.param(
            {
                'filter_type': 'scaleoffset',
                'params': {'scaleoffset_params': '2,IN'},
                'h5z_constant': 'FILTER_SCALEOFFSET',
                'verify_key': 'scaleoffset',
                'verify_value': None,  # Just check not None
            },
            id='scaleoffset'
        ),
        # SZIP omitted: Most HDF5 builds only support decode (reading), not encode (writing)
        # due to patent licensing restrictions
    ])
    def test_apply_filter(self, filter_config):
        """Test applying various filters to uncompressed dataset"""
        filepath = None
        output_filepath = None
        try:
            # Check if filter is available in h5py
            import h5py.h5z as h5z
            h5z_const = getattr(h5z, filter_config['h5z_constant'])
            if not h5z.filter_avail(h5z_const):
                pytest.skip(f"{filter_config['filter_type']} filter not available in h5py")


            # Create temporary file with uncompressed dataset
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                # Use int32 for filters that work better with integers
                if filter_config['filter_type'] in ['nbit', 'scaleoffset', 'szip']:
                    data = np.arange(10000, dtype=np.int32).reshape(100, 100)
                else:
                    data = np.arange(10000).reshape(100, 100)
                f.create_dataset('data', data=data, chunks=(10, 10))

            # Apply filter
            result = apply_filter_dataset(
                filepath, '/data',
                filter_type=filter_config['filter_type'],
                **filter_config['params']
            )

            # Print diagnostics if failure
            if not result.get('success', False):
                print(f"\n=== h5repack Diagnostics for {filter_config['filter_type']} ===")
                print(f"Version: {result.get('h5repack_version', 'unknown')}")
                print(f"Command: {result.get('command', 'unknown')}")
                print(f"STDOUT: {result.get('h5repack_stdout', '(empty)')}")
                print(f"STDERR: {result.get('h5repack_stderr', '(empty)')}")
                print("============================\n")

            # Verify success
            validate_success(result)
            assert 'output_filepath' in result
            output_filepath = result['output_filepath']
            assert os.path.exists(output_filepath)

            # Verify filter was applied
            if filter_config.get('verify_key'):
                verify_key = filter_config['verify_key']
                verify_value = filter_config['verify_value']

                if verify_value is None:
                    # Just check it's not None (for scaleoffset)
                    assert result['new_filters'][verify_key] is not None
                else:
                    assert result['new_filters'][verify_key] == verify_value

                # Check opts if specified
                if 'verify_opts_key' in filter_config:
                    opts_key = filter_config['verify_opts_key']
                    opts_value = filter_config['verify_opts_value']
                    assert result['new_filters'][opts_key] == opts_value

            # Verify data integrity (except for lossy filters)
            if filter_config['filter_type'] not in ['scaleoffset']:
                with h5py.File(filepath, 'r') as f_in:
                    with h5py.File(output_filepath, 'r') as f_out:
                        assert np.array_equal(f_in['data'][:], f_out['data'][:])

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)
            if output_filepath and os.path.exists(output_filepath):
                os.unlink(output_filepath)

    def test_remove_filters(self):
        """Test removing all filters (decompress)"""
        filepath = None
        output_filepath = None
        try:
            # Check if gzip filter is available in h5py
            import h5py.h5z as h5z
            if not h5z.filter_avail(h5z.FILTER_DEFLATE):
                pytest.skip("gzip filter not available in h5py")


            # Create temporary file with compression
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                data = np.arange(10000).reshape(100, 100)
                f.create_dataset('data', data=data, chunks=(10, 10),
                               compression='gzip', compression_opts=9)

            # Remove filters
            result = apply_filter_dataset(filepath, '/data', remove_all_filters=True)

            # Print diagnostics if failure
            if not result.get('success', False):
                print("\n=== h5repack Diagnostics ===")
                print(f"Version: {result.get('h5repack_version', 'unknown')}")
                print(f"Command: {result.get('command', 'unknown')}")
                print(f"STDOUT: {result.get('h5repack_stdout', '(empty)')}")
                print(f"STDERR: {result.get('h5repack_stderr', '(empty)')}")
                print("============================\n")

            validate_success(result)
            output_filepath = result['output_filepath']
            assert os.path.exists(output_filepath)

            # Verify filters removed
            assert result['original_filters']['compression'] == 'gzip'
            assert result['new_filters']['compression'] is None

            # Verify data integrity
            with h5py.File(filepath, 'r') as f_in:
                with h5py.File(output_filepath, 'r') as f_out:
                    assert np.array_equal(f_in['data'][:], f_out['data'][:])

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)
            if output_filepath and os.path.exists(output_filepath):
                os.unlink(output_filepath)

    def test_change_compression_level(self):
        """Test changing GZIP compression level"""
        filepath = None
        output_filepath = None
        try:
            # Check if gzip filter is available in h5py
            import h5py.h5z as h5z
            if not h5z.filter_avail(h5z.FILTER_DEFLATE):
                pytest.skip("gzip filter not available in h5py")


            # Create file with gzip level 1
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                data = np.arange(10000).reshape(100, 100)
                f.create_dataset('data', data=data, chunks=(10, 10),
                               compression='gzip', compression_opts=1)

            # Change to gzip level 9
            result = apply_filter_dataset(filepath, '/data', filter_type='gzip',
                                         compression_level=9)

            # Verify success
            validate_success(result)
            output_filepath = result['output_filepath']

            # Verify compression level changed
            assert result['original_filters']['compression_opts'] == 1
            assert result['new_filters']['compression_opts'] == 9

            # Verify data integrity
            with h5py.File(filepath, 'r') as f_in:
                with h5py.File(output_filepath, 'r') as f_out:
                    assert np.array_equal(f_in['data'][:], f_out['data'][:])

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)
            if output_filepath and os.path.exists(output_filepath):
                os.unlink(output_filepath)
