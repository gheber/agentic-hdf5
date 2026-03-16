import h5py
from h5py_helpers import _get_attribute_metadata, _get_dataset_metadata, _get_group_metadata, _get_committed_datatype_metadata, _has_smd
from h5py_constants import SMD_GENERATION_BATCH_SIZE

try:
    from tools.h5py.registry import hdf5_tool
except ImportError:
    def hdf5_tool(**_kw):
        return lambda f: f


# TBD - Impose bottom-up ordering for context optimization (e.g. file summary is more knowledgeable)
@hdf5_tool(
    category="semantic-metadata",
    keywords=["collect", "batch", "scan", "missing", "smd", "semantic metadata", "generate", "iterate"],
    use_cases=[
        "Finding all objects without SMD",
        "Batch SMD generation workflows",
        "Iteratively documenting large files",
        "Identifying undocumented datasets",
    ],
)
def collect_objects_for_smd(filepath: str, object_path: str = "/",
                            max_depth: int = -1) -> dict:
    """
    Scan file for objects missing SMD. Returns batch with structural metadata for auto-generating descriptions.

    Iterative workflow: call this → generate SMD → write_smd_batch() → repeat until empty batch.

    This enables an iterative workflow:
    1. Agent calls this function to get a batch of objects
    2. Agent generates SMD for each object based on metadata
    3. Agent calls write_smd_batch() to save the SMD
    4. Agent repeats until this function returns an empty batch

    Args:
        filepath: Path to the HDF5 file
        object_path: Starting point for collection (default: root)
        max_depth: Maximum recursion depth (-1 = unlimited, 0 = only specified path)

    Returns:
        Dictionary containing:
        - batch: List of objects with their metadata
        - remaining_estimate: Approximate count of objects still needing SMD (or -1 if unknown)
        - total_in_batch: Number of objects in this batch
        - batch_complete: True if batch is empty (all objects have SMD)
    """
    try:
        with h5py.File(filepath, 'r') as f:
            if object_path not in f:
                return {
                    "error": f"Object '{object_path}' not found in file '{filepath}'",
                    "batch": [],
                    "total_in_batch": 0,
                    "batch_complete": True
                }

            batch = []
            objects_without_smd = []

            # Helper function to recursively collect objects
            def collect_recursive(path: str, current_depth: int):
                if max_depth != -1 and current_depth > max_depth:
                    return

                try:
                    obj = f[path]

                    # Check if this object needs SMD
                    if not _has_smd(f, path):
                        # Get metadata for this object
                        metadata = {}
                        metadata["name"] = path

                        # Add type-specific metadata
                        if isinstance(obj, h5py.Dataset):
                            metadata.update(_get_dataset_metadata(obj))
                        elif isinstance(obj, h5py.Group):
                            metadata.update(_get_group_metadata(obj, path))
                        elif isinstance(obj, h5py.Datatype):
                            metadata.update(_get_committed_datatype_metadata(obj))

                        # Add attributes if present
                        if hasattr(obj, 'attrs') and len(obj.attrs) > 0:
                            attr_metadata = {name: _get_attribute_metadata(obj.attrs[name])
                                           for name in obj.attrs}
                            metadata["attributes"] = attr_metadata

                        # Create a human-readable summary for the LLM
                        summary_parts = [f"Object: {path}"]
                        summary_parts.append(f"Type: {metadata.get('type', 'unknown')}")

                        if metadata.get('type') == 'dataset':
                            summary_parts.append(f"Shape: {metadata.get('shape')}")
                            summary_parts.append(f"Dtype: {metadata.get('dtype')}")
                            if metadata.get('attributes'):
                                summary_parts.append(f"Attributes: {', '.join(metadata['attributes'].keys())}")
                        elif metadata.get('type') in ('group', 'file_root'):
                            summary_parts.append(f"Members: {metadata.get('num_members', 0)}")

                        metadata_summary = " | ".join(summary_parts)

                        objects_without_smd.append({
                            "path": path,
                            "metadata": metadata,
                            "metadata_summary": metadata_summary
                        })

                    # Recurse into groups
                    if isinstance(obj, h5py.Group):
                        for key in obj.keys():
                            child_path = f"{path}/{key}" if path != '/' else f"/{key}"
                            collect_recursive(child_path, current_depth + 1)

                except Exception as e:
                    # Log error but continue with other objects
                    pass

            # Start collection
            collect_recursive(object_path, 0)

            # Return up to BATCH_SIZE objects
            batch = objects_without_smd[:SMD_GENERATION_BATCH_SIZE]
            remaining = len(objects_without_smd) - len(batch)

            return {
                "batch": batch,
                "total_in_batch": len(batch),
                "remaining_estimate": remaining,
                "batch_complete": len(batch) == 0,
                "filepath": filepath
            }

    except Exception as e:
        return {
            "error": f"Error collecting objects: {str(e)}",
            "batch": [],
            "total_in_batch": 0,
            "batch_complete": True
        }
