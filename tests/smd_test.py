"""
Tests for Semantic Metadata (SMD) functionality.
"""

import pytest
import h5py
import numpy as np
import tempfile
import os
import stat
from pathlib import Path

# Import functions under test (to be implemented)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "servers" / "h5py"))

from read_semantic_metadata import read_semantic_metadata
from write_semantic_metadata import write_semantic_metadata
from collect_objects_for_smd import collect_objects_for_smd
from write_smd_batch import write_smd_batch
from h5py_constants import SMD_GENERATION_BATCH_SIZE

from vectorize_semantic_metadata import vectorize_semantic_metadata
from query_semantic_metadata import query_semantic_metadata


class TestSemanticMetadata:
    """Test suite for semantic metadata read/write functions"""

    def test_write_and_read_smd_dataset(self):
        """Test writing and reading semantic metadata for a dataset"""
        filepath = None
        try:
            # Create temporary file with dataset
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                f.create_dataset('temperature', data=[20.5, 21.3, 19.8])

            # Write semantic metadata (is_best_guess=False for definite user-provided data)
            smd_content = "Temperature readings in Celsius\nCollected from sensor A\nSampling rate: 1 Hz"
            result = write_semantic_metadata(filepath, '/temperature', smd_content, is_best_guess=False)
            assert "successfully" in result.lower()

            # Read semantic metadata
            read_result = read_semantic_metadata(filepath, '/temperature')
            assert read_result == smd_content

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_write_and_read_smd_group(self):
        """Test writing and reading semantic metadata for a group"""
        filepath = None
        try:
            # Create temporary file with group
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                f.create_group('experiment_data')

            # Write semantic metadata (is_best_guess=False for definite user-provided data)
            smd_content = "Experimental data from trial #5\nDate: 2024-01-15"
            result = write_semantic_metadata(filepath, '/experiment_data', smd_content, is_best_guess=False)
            assert "successfully" in result.lower()

            # Read semantic metadata
            read_result = read_semantic_metadata(filepath, '/experiment_data')
            assert read_result == smd_content

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_write_and_read_smd_root(self):
        """Test writing and reading semantic metadata for root group"""
        filepath = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                f.create_dataset('data', data=[1, 2, 3])

            # Write semantic metadata for root (is_best_guess=False for definite user-provided data)
            smd_content = "HDF5 file containing experimental data\nVersion: 1.0"
            result = write_semantic_metadata(filepath, '/', smd_content, is_best_guess=False)
            assert "successfully" in result.lower()

            # Read semantic metadata
            read_result = read_semantic_metadata(filepath, '/')
            assert read_result == smd_content

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_write_smd_best_guess(self):
        """Test writing best-guess semantic metadata"""
        filepath = None
        try:
            # Create temporary file with dataset
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                f.create_dataset('velocity', data=[10.5, 11.2, 9.8])

            # Write best-guess semantic metadata
            smd_content = "Velocity measurements\nUnit: m/s"
            result = write_semantic_metadata(filepath, '/velocity', smd_content, is_best_guess=True)
            assert "successfully" in result.lower()

            # Read and verify it has BEST GUESS prefix
            read_result = read_semantic_metadata(filepath, '/velocity')
            assert "BEST GUESS: Velocity measurements" in read_result
            assert "BEST GUESS: Unit: m/s" in read_result

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_read_nonexistent_smd(self):
        """Test reading semantic metadata that doesn't exist"""
        filepath = None
        try:
            # Create temporary file with dataset but no SMD
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                f.create_dataset('data', data=[1, 2, 3])

            # Try to read non-existent SMD
            result = read_semantic_metadata(filepath, '/data')
            assert "No semantic metadata found" in result

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_write_smd_nonexistent_object(self):
        """Test writing semantic metadata to non-existent object"""
        filepath = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                pass

            # Try to write SMD to non-existent object
            result = write_semantic_metadata(filepath, '/nonexistent', "some metadata")
            assert "Error" in result
            assert "not found" in result

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_read_smd_nonexistent_object(self):
        """Test reading semantic metadata from non-existent object"""
        filepath = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                pass

            # Try to read SMD from non-existent object
            result = read_semantic_metadata(filepath, '/nonexistent')
            assert "Error" in result
            assert "not found" in result

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_overwrite_smd(self):
        """Test overwriting existing semantic metadata"""
        filepath = None
        try:
            # Create temporary file with dataset
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                f.create_dataset('pressure', data=[101.3, 100.5])

            # Write initial SMD
            smd_v1 = "Pressure in kPa"
            write_semantic_metadata(filepath, '/pressure', smd_v1, is_best_guess=False)

            # Overwrite with new SMD
            smd_v2 = "Atmospheric pressure in kPa\nCalibrated sensor"
            result = write_semantic_metadata(filepath, '/pressure', smd_v2, is_best_guess=False)
            assert "successfully" in result.lower()

            # Read and verify it's the new version
            read_result = read_semantic_metadata(filepath, '/pressure')
            assert read_result == smd_v2
            assert "Calibrated sensor" in read_result

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_smd_nested_group(self):
        """Test semantic metadata for nested group"""
        filepath = None
        try:
            # Create temporary file with nested groups
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                parent = f.create_group('experiments')
                parent.create_group('trial_1')

            # Write SMD for nested group (is_best_guess=False for definite user-provided data)
            smd_content = "First trial of the experiment series"
            result = write_semantic_metadata(filepath, '/experiments/trial_1', smd_content, is_best_guess=False)
            assert "successfully" in result.lower()

            # Read and verify
            read_result = read_semantic_metadata(filepath, '/experiments/trial_1')
            assert read_result == smd_content

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_smd_multiline_with_empty_lines(self):
        """Test semantic metadata with multiple lines including empty lines"""
        filepath = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                f.create_dataset('results', data=[1, 2, 3])

            # Write multi-line SMD with empty lines (is_best_guess=False for definite user-provided data)
            smd_content = "Experimental results\n\nSection A: Initial measurements\nSection B: Final measurements"
            result = write_semantic_metadata(filepath, '/results', smd_content, is_best_guess=False)
            assert "successfully" in result.lower()

            # Read and verify formatting is preserved
            read_result = read_semantic_metadata(filepath, '/results')
            assert read_result == smd_content
            assert "\n\n" in read_result  # Empty line preserved

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_write_smd_default_best_guess(self):
        """Test that is_best_guess defaults to True (for inferred metadata)"""
        filepath = None
        try:
            # Create temporary file with dataset
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                f.create_dataset('distance', data=[100.0, 250.3, 175.8])

            # Write metadata without specifying is_best_guess (should default to True)
            # This simulates inferring metadata from the object name and data
            smd_content = "Distance measurements\nInferred unit: meters based on object name"
            result = write_semantic_metadata(filepath, '/distance', smd_content)
            assert "successfully" in result.lower()

            # Read and verify it has BEST GUESS prefix (default behavior)
            read_result = read_semantic_metadata(filepath, '/distance')
            assert "BEST GUESS: Distance measurements" in read_result
            assert "BEST GUESS: Inferred unit: meters based on object name" in read_result

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)


class TestBatchSMDGeneration:
    """Test suite for batch SMD generation functions"""

    def test_collect_objects_basic(self):
        """Test collecting objects that need SMD"""
        filepath = None
        try:
            # Create temporary file with multiple objects
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                f.create_dataset('data1', data=[1, 2, 3])
                f.create_dataset('data2', data=[4, 5, 6])
                grp = f.create_group('group1')
                grp.create_dataset('nested_data', data=[7, 8, 9])

            # Collect objects needing SMD
            result = collect_objects_for_smd(filepath)

            # Verify result structure
            assert 'batch' in result
            assert 'total_in_batch' in result
            assert 'remaining_estimate' in result
            assert 'batch_complete' in result
            assert result['batch_complete'] == False
            assert result['total_in_batch'] > 0

            # Verify objects have metadata
            for obj in result['batch']:
                assert 'path' in obj
                assert 'metadata' in obj
                assert 'metadata_summary' in obj

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_collect_objects_skips_with_smd(self):
        """Test that objects with existing SMD are skipped"""
        filepath = None
        try:
            # Create file with objects
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                f.create_dataset('data1', data=[1, 2, 3])
                f.create_dataset('data2', data=[4, 5, 6])

            # Write SMD for data1
            write_semantic_metadata(filepath, '/data1', "Test metadata", is_best_guess=False)

            # Collect objects
            result = collect_objects_for_smd(filepath)

            # Verify data1 is not in batch (already has SMD)
            batch_paths = [obj['path'] for obj in result['batch']]
            assert '/data1' not in batch_paths
            assert '/data2' in batch_paths or result['total_in_batch'] > 0

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_collect_objects_empty_when_complete(self):
        """Test that empty batch is returned when all objects have SMD"""
        filepath = None
        try:
            # Create file with one object
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                f.create_dataset('data', data=[1, 2, 3])

            # Write SMD for both data and root
            write_semantic_metadata(filepath, '/data', "Test metadata", is_best_guess=False)
            write_semantic_metadata(filepath, '/', "Root metadata", is_best_guess=False)

            # Collect objects
            result = collect_objects_for_smd(filepath)

            # Verify batch is empty
            assert result['total_in_batch'] == 0
            assert result['batch_complete'] == True
            assert len(result['batch']) == 0

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_collect_objects_respects_batch_size(self):
        """Test that collect respects SMD_GENERATION_BATCH_SIZE"""
        filepath = None
        try:
            # Create file with more objects than batch size
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                for i in range(10):
                    f.create_dataset(f'data{i}', data=[i])

            # Collect objects
            result = collect_objects_for_smd(filepath)

            # Verify batch size is respected
            assert result['total_in_batch'] <= SMD_GENERATION_BATCH_SIZE
            assert result['remaining_estimate'] >= 0

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_write_smd_batch_success(self):
        """Test successful batch write of SMD"""
        filepath = None
        try:
            # Create file with objects
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                f.create_dataset('data1', data=[1, 2, 3])
                f.create_dataset('data2', data=[4, 5, 6])

            # Write batch
            smd_map = {
                '/data1': 'First dataset metadata',
                '/data2': 'Second dataset metadata'
            }
            result = write_smd_batch(filepath, smd_map, is_best_guess=False)

            # Verify success
            assert result['success_count'] == 2
            assert result['failure_count'] == 0
            assert '/data1' in result['successes']
            assert '/data2' in result['successes']

            # Verify SMD was written
            smd1 = read_semantic_metadata(filepath, '/data1')
            assert smd1 == 'First dataset metadata'

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_write_smd_batch_with_best_guess(self):
        """Test batch write with best_guess prefix"""
        filepath = None
        try:
            # Create file
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                f.create_dataset('data', data=[1, 2, 3])

            # Write with best_guess=True
            smd_map = {'/data': 'Inferred metadata\nAnother line'}
            result = write_smd_batch(filepath, smd_map, is_best_guess=True)

            # Verify success
            assert result['success_count'] == 1

            # Verify BEST GUESS prefix
            smd = read_semantic_metadata(filepath, '/data')
            assert 'BEST GUESS: Inferred metadata' in smd
            assert 'BEST GUESS: Another line' in smd

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_write_smd_batch_partial_failure(self):
        """Test that batch write continues on individual failures"""
        filepath = None
        try:
            # Create file with one object
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                f.create_dataset('data1', data=[1, 2, 3])

            # Write batch with one valid and one invalid path
            smd_map = {
                '/data1': 'Valid metadata',
                '/nonexistent': 'This should fail'
            }
            result = write_smd_batch(filepath, smd_map, is_best_guess=False)

            # Verify partial success
            assert result['success_count'] == 1
            assert result['failure_count'] == 1
            assert '/data1' in result['successes']
            assert len(result['failures']) == 1
            assert result['failures'][0]['path'] == '/nonexistent'

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_iterative_workflow(self):
        """Test complete iterative workflow: collect -> write -> collect"""
        filepath = None
        try:
            # Create file with objects
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                f.create_dataset('data1', data=[1, 2, 3])
                f.create_dataset('data2', data=[4, 5, 6])

            # First batch
            batch1 = collect_objects_for_smd(filepath)
            assert batch1['total_in_batch'] > 0
            assert batch1['batch_complete'] == False

            # Write SMD for first batch
            smd_map = {obj['path']: f"Metadata for {obj['path']}" 
                      for obj in batch1['batch']}
            write_result = write_smd_batch(filepath, smd_map, is_best_guess=False)
            assert write_result['success_count'] == len(smd_map)

            # Collect again - should get different objects or empty batch
            batch2 = collect_objects_for_smd(filepath)
            
            # Verify no overlap between batches
            batch1_paths = {obj['path'] for obj in batch1['batch']}
            batch2_paths = {obj['path'] for obj in batch2['batch']}
            assert len(batch1_paths & batch2_paths) == 0

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)

    def test_collect_objects_max_depth(self):
        """Test max_depth parameter limits recursion"""
        filepath = None
        try:
            # Create nested structure
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                filepath = tmp.name

            with h5py.File(filepath, 'w') as f:
                grp1 = f.create_group('level1')
                grp2 = grp1.create_group('level2')
                grp2.create_dataset('deep_data', data=[1, 2, 3])

            # Collect with max_depth=1
            result = collect_objects_for_smd(filepath, max_depth=1)

            # Verify deep objects are not included
            batch_paths = [obj['path'] for obj in result['batch']]
            assert '/level1/level2/deep_data' not in batch_paths
            assert '/level1' in batch_paths or result['total_in_batch'] > 0

        finally:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)


class TestSMDErrorPaths:
    """Tests for error/edge-case paths in SMD read/write functions."""

    def test_write_readonly_file(self, tmp_path):
        """write_semantic_metadata OSError path for read-only files."""
        path = str(tmp_path / "readonly.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=[1, 2, 3])
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        try:
            result = write_semantic_metadata(path, "/data", "test")
            assert "error" in result.lower()
        finally:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)

    def test_write_nonexistent_file(self):
        """write_semantic_metadata OSError path for missing file."""
        result = write_semantic_metadata("/tmp/does_not_exist_ahdf5.h5", "/data", "test")
        assert "error" in result.lower()

    def test_read_nonexistent_file(self):
        """read_semantic_metadata exception path for missing file."""
        result = read_semantic_metadata("/tmp/does_not_exist_ahdf5.h5", "/data")
        assert "error" in result.lower()

    def test_read_bytes_smd(self, tmp_path):
        """read_semantic_metadata bytes decoding path."""
        path = str(tmp_path / "bytes_smd.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=[1, 2, 3])
            f["/"].attrs["ahdf5-smd-data"] = b"bytes metadata content"
        result = read_semantic_metadata(path, "/data")
        assert result == "bytes metadata content"

    def test_batch_readonly_file(self, tmp_path):
        """write_smd_batch OSError path for read-only file."""
        path = str(tmp_path / "readonly.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=[1, 2, 3])
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        try:
            result = write_smd_batch(path, {"/data": "test metadata"})
            assert result["success_count"] == 0
            assert result["failure_count"] == 1
        finally:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)

    def test_batch_nonexistent_file(self):
        """write_smd_batch OSError path for missing file."""
        result = write_smd_batch("/tmp/does_not_exist_ahdf5.h5", {"/data": "test"})
        assert result["success_count"] == 0
        assert result["failure_count"] == 1
        assert len(result["failures"]) == 1

    def test_batch_mixed_valid_invalid(self, tmp_path):
        """write_smd_batch per-item exception path."""
        path = str(tmp_path / "test.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=[1, 2, 3])
        smd_map = {"/data": "valid metadata", "/nonexistent_obj": "should fail"}
        result = write_smd_batch(path, smd_map, is_best_guess=False)
        assert result["success_count"] == 1
        assert result["failure_count"] == 1
        assert any(f["path"] == "/nonexistent_obj" for f in result["failures"])
