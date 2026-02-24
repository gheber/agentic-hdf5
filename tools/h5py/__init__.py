import sys
from pathlib import Path

# Add this directory to sys.path so bare imports (h5py_helpers, h5py_constants)
# used within tool modules continue to work when imported as a package.
_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from tools.h5py.get_object_metadata import get_object_metadata
from tools.h5py.rechunk_dataset import rechunk_dataset
from tools.h5py.apply_filter_dataset import apply_filter_dataset
from tools.h5py.visualize import visualize
from tools.h5py.read_semantic_metadata import read_semantic_metadata
from tools.h5py.write_semantic_metadata import write_semantic_metadata
from tools.h5py.collect_objects_for_smd import collect_objects_for_smd
from tools.h5py.write_smd_batch import write_smd_batch
from tools.h5py.vectorize_semantic_metadata import vectorize_semantic_metadata
from tools.h5py.query_semantic_metadata import query_semantic_metadata

__all__ = [
    "get_object_metadata",
    "rechunk_dataset",
    "apply_filter_dataset",
    "visualize",
    "read_semantic_metadata",
    "write_semantic_metadata",
    "collect_objects_for_smd",
    "write_smd_batch",
    "vectorize_semantic_metadata",
    "query_semantic_metadata",
]
