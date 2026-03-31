"""
Unit tests for the AHDF5 metadata tool(s)
"""

import pytest
import h5py
import tempfile
import os
import numpy as np

from get_object_metadata import get_object_metadata
from h5py_helpers import (
    _convert_to_json_serializable,
    _get_numeric_statistics,
    _estimate_size_bytes,
    _get_attribute_metadata,
)


class TestGetObjectMetadata:
    """Test suite for get_object_metadata function"""

    def test_file_root_metadata(self):
        """Test metadata retrieval for file root"""
        filepath = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            # Create file with some groups
            with h5py.File(filepath, 'w') as f:
                f.create_group('group1')
                f.create_group('group2')
                f.attrs['file_version'] = '1.0'
                f.attrs['created_by'] = 'test'

            # Test metadata retrieval
            metadata = get_object_metadata(filepath, '/')

            # Verify file root metadata
            assert metadata['type'] == 'file_root'
            assert metadata['name'] == '/'
            assert 'members' in metadata
            assert set(metadata['members']) == {'group1', 'group2'}
            assert metadata['num_members'] == 2
            assert 'attributes' in metadata
            assert 'file_version' in metadata['attributes']
            assert metadata['attributes']['file_version']['value'] == '1.0'
            assert metadata['attributes']['created_by']['value'] == 'test'

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_group_metadata(self):
        """Test metadata retrieval for a group"""
        filepath = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            # Create file with nested groups
            with h5py.File(filepath, 'w') as f:
                grp = f.create_group('parent')
                grp.create_group('child1')
                grp.create_dataset('dataset1', data=[1, 2, 3])
                grp.attrs['group_type'] = 'test_group'
                grp.attrs['count'] = 42

            # Test metadata retrieval
            metadata = get_object_metadata(filepath, '/parent')

            # Verify group metadata
            assert metadata['type'] == 'group'
            assert metadata['name'] == '/parent'
            assert 'members' in metadata
            assert set(metadata['members']) == {'child1', 'dataset1'}
            assert metadata['num_members'] == 2
            assert 'attributes' in metadata
            assert metadata['attributes']['group_type']['value'] == 'test_group'
            assert metadata['attributes']['count']['value'] == 42

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_dataset_metadata_basic(self):
        """Test metadata retrieval for a basic dataset"""
        filepath = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            # Create file with dataset
            with h5py.File(filepath, 'w') as f:
                data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
                ds = f.create_dataset('data', data=data)
                ds.attrs['units'] = 'meters'
                ds.attrs['scale'] = 1.5

            # Test metadata retrieval
            metadata = get_object_metadata(filepath, '/data')

            # Verify dataset metadata
            assert metadata['type'] == 'dataset'
            assert metadata['name'] == '/data'
            assert metadata['shape'] == (2, 3)
            assert 'float32' in metadata['dtype']
            assert metadata['size'] == 6
            assert 'attributes' in metadata
            assert metadata['attributes']['units']['value'] == 'meters'
            assert metadata['attributes']['scale']['value'] == 1.5

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_dataset_metadata_with_compression(self):
        """Test metadata retrieval for a dataset with compression and chunking"""
        filepath = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            # Create file with compressed, chunked dataset
            with h5py.File(filepath, 'w') as f:
                data = np.random.rand(100, 100)
                ds = f.create_dataset('compressed_data',
                                      data=data,
                                      chunks=(10, 10),
                                      compression='gzip',
                                      compression_opts=9,
                                      fillvalue=-999.0)

            # Test metadata retrieval
            metadata = get_object_metadata(filepath, '/compressed_data')

            # Verify dataset metadata with storage info
            assert metadata['type'] == 'dataset'
            assert metadata['name'] == '/compressed_data'
            assert metadata['shape'] == (100, 100)
            assert metadata['chunks'] == (10, 10)
            assert metadata['compression'] == 'gzip'
            assert metadata['compression_opts'] == 9
            assert metadata['fillvalue'] == -999.0
            assert metadata['maxshape'] == (100, 100)

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_committed_datatype_metadata(self):
        """Test metadata retrieval for a committed datatype"""
        filepath = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            # Create file with committed datatype using h5py high-level API
            with h5py.File(filepath, 'w') as f:
                # Create a compound datatype
                dt = np.dtype([('x', 'f8'), ('y', 'f8'), ('name', 'S10')])
                # Commit the datatype using h5py's commit method
                tid = h5py.h5t.py_create(dt, logical=True)
                tid.commit(f.id, b'point_type')

            # Test metadata retrieval
            metadata = get_object_metadata(filepath, '/point_type')

            # Verify committed datatype metadata
            assert metadata['type'] == 'committed_datatype'
            assert metadata['name'] == '/point_type'
            assert 'dtype' in metadata
            # The dtype should contain information about the compound type
            assert 'x' in metadata['dtype'] or 'f8' in metadata['dtype']

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_soft_link_metadata(self):
        """Test metadata retrieval for a soft link"""
        filepath = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            # Create file with soft link
            with h5py.File(filepath, 'w') as f:
                f.create_dataset('original', data=[1, 2, 3])
                f['link_to_original'] = h5py.SoftLink('/original')

            # Test metadata retrieval
            metadata = get_object_metadata(filepath, '/link_to_original')

            # Verify soft link metadata
            assert metadata['type'] == 'soft_link'
            assert metadata['name'] == '/link_to_original'
            assert metadata['target'] == '/original'

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_external_link_metadata(self):
        """Test metadata retrieval for an external link"""
        filepath = None
        external_filepath = None
        try:
            # Create external file
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                external_filepath = tmp.name

            with h5py.File(external_filepath, 'w') as f:
                f.create_dataset('external_data', data=[10, 20, 30])

            # Create main file with external link
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                f['external_link'] = h5py.ExternalLink(external_filepath, '/external_data')

            # Test metadata retrieval
            metadata = get_object_metadata(filepath, '/external_link')

            # Verify external link metadata
            assert metadata['type'] == 'external_link'
            assert metadata['name'] == '/external_link'
            assert metadata['filename'] == external_filepath
            assert metadata['target'] == '/external_data'

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)
            if external_filepath and os.path.exists(external_filepath):
                os.unlink(external_filepath)

    def test_nonexistent_object(self):
        """Test error handling for nonexistent object"""
        filepath = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            # Create empty file
            with h5py.File(filepath, 'w') as f:
                pass

            # Test metadata retrieval for nonexistent object
            metadata = get_object_metadata(filepath, '/does_not_exist')

            # Verify error is returned
            assert 'error' in metadata
            assert 'not found' in metadata['error'].lower()

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        metadata = get_object_metadata("/tmp/does_not_exist_ahdf5.h5", "/data")
        assert "error" in metadata


class TestConvertToJsonSerializable:
    """Tests for _convert_to_json_serializable helper."""

    def test_numpy_bool(self):
        assert _convert_to_json_serializable(np.bool_(True)) is True

    def test_numpy_void(self):
        val = np.void(b'\x01\x02')
        result = _convert_to_json_serializable(val)
        assert isinstance(result, str)

    def test_bytes_value(self):
        assert _convert_to_json_serializable(b"hello") == "hello"

    def test_tuple_recursive(self):
        result = _convert_to_json_serializable((np.int64(1), np.float64(2.5)))
        assert result == (1, 2.5)
        assert isinstance(result, tuple)

    def test_list_recursive(self):
        result = _convert_to_json_serializable([np.int64(1), b"test"])
        assert result == [1, "test"]
        assert isinstance(result, list)

    def test_dict_recursive(self):
        result = _convert_to_json_serializable({"a": np.int64(1), "b": b"test"})
        assert result == {"a": 1, "b": "test"}

    def test_plain_value_passthrough(self):
        assert _convert_to_json_serializable("hello") == "hello"
        assert _convert_to_json_serializable(42) == 42


class TestGetNumericStatistics:
    """Tests for _get_numeric_statistics helper."""

    def test_non_numeric_array(self):
        data = np.array(["a", "b", "c"])
        assert _get_numeric_statistics(data) == {}

    def test_not_an_array(self):
        assert _get_numeric_statistics("not an array") == {}

    def test_numeric_array(self):
        data = np.array([1.0, 5.0, 3.0])
        stats = _get_numeric_statistics(data)
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0


class TestEstimateSizeBytes:
    """Tests for _estimate_size_bytes helper."""

    def test_numpy_array(self):
        arr = np.zeros((10,), dtype=np.float64)
        assert _estimate_size_bytes(arr) == 80  # 10 * 8 bytes

    def test_numpy_scalar(self):
        val = np.float64(1.0)
        assert _estimate_size_bytes(val) == 8

    def test_python_int(self):
        size = _estimate_size_bytes(42)
        assert size > 0

    def test_string(self):
        assert _estimate_size_bytes("hello") == 5

    def test_bytes(self):
        assert _estimate_size_bytes(b"hello") == 5

    def test_unknown_type(self):
        assert _estimate_size_bytes(object()) == 0


class TestGetAttributeMetadata:
    """Tests for _get_attribute_metadata helper with edge cases."""

    def test_large_string_attribute(self):
        """Large strings should be omitted with a note."""
        big_string = "x" * 2000
        result = _get_attribute_metadata(big_string)
        assert "note" in result
        assert "Large string" in result["note"]

    def test_large_array_attribute(self):
        """Large arrays should be omitted with a note."""
        big_array = np.arange(200000)
        result = _get_attribute_metadata(big_array)
        assert "note" in result
        assert "Large array" in result["note"]

    def test_numeric_scalar_attribute(self):
        result = _get_attribute_metadata(np.float64(3.14))
        assert result["value"] == 3.14

    def test_small_bytes_attribute(self):
        result = _get_attribute_metadata(b"test")
        assert result["value"] == "test"

