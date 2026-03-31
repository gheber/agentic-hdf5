import sys
from pathlib import Path
import pytest
import h5py
import tempfile
import os

# Add the tools/h5py directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools' / 'h5py'))


@pytest.fixture
def test_h5_file():
    """Fixture that creates a temporary HDF5 file with test data"""
    # Use project dir — snap Docker can't access /tmp
    project_root = str(Path(__file__).resolve().parent.parent)
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False, dir=project_root) as tmp:
        filepath = tmp.name

    with h5py.File(filepath, 'w') as f:
        # Create test structure
        grp = f.create_group('test_group')
        ds = grp.create_dataset('data', data=[1, 2, 3, 4, 5])
        ds.attrs['units'] = 'km'
        ds.attrs['description'] = 'Test dataset'

    yield filepath

    # Cleanup
    if os.path.exists(filepath):
        os.unlink(filepath)
