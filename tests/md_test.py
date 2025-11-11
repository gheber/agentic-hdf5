"""
Unit tests for the AHDF5 metadata tool(s)
"""

import pytest
import h5py
import tempfile
import os
import numpy as np

from get_object_metadata import get_object_metadata


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

