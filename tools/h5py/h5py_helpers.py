import h5py
import numpy as np

from typing import Any
from h5py_constants import NUM_ELEMS_PREVIEW_THRESHOLD, BYTE_SIZE_PREVIEW_THRESHOLD


def _convert_to_json_serializable(value: Any) -> Any:
    """
    Convert numpy types and other non-JSON-serializable types to native Python types.

    Args:
        value: Value to convert

    Returns:
        JSON-serializable version of the value
    """
    # Handle numpy scalar types
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, np.void):
        # Handle structured/compound numpy types
        return str(value)
    elif isinstance(value, bytes):
        return value.decode('utf-8')
    elif isinstance(value, (tuple, list)):
        return type(value)(_convert_to_json_serializable(v) for v in value)
    elif isinstance(value, dict):
        return {k: _convert_to_json_serializable(v) for k, v in value.items()}
    else:
        return value


def _get_numeric_statistics(data) -> dict:
    """
    Compute summary statistics for numeric data.

    Args:
        data: Numpy array or scalar numeric value

    Returns:
        Dictionary with min and max values, or empty dict if not numeric
    """
    try:
        # Check if data is numeric
        if np.issubdtype(data.dtype, np.number):
            # Compute statistics on flattened array
            stats = {
                "min": _convert_to_json_serializable(np.min(data)),
                "max": _convert_to_json_serializable(np.max(data))
            }
            return stats
    except (AttributeError, TypeError):
        # Not a numpy array or not numeric
        pass

    return {}


def _estimate_size_bytes(obj) -> int:
    """
    Estimate size in bytes for an object.

    Args:
        obj: Dataset, array, or scalar value

    Returns:
        Size in bytes
    """
    if isinstance(obj, h5py.Dataset):
        # For datasets, multiply element size by number of elements
        return obj.dtype.itemsize * obj.size
    elif hasattr(obj, 'dtype') and hasattr(obj, 'size'):
        # Numpy array
        return obj.dtype.itemsize * obj.size
    elif isinstance(obj, (int, float, bool, np.integer, np.floating, np.bool_)):
        # Scalar numeric
        if hasattr(obj, 'itemsize'):
            return obj.itemsize
        else:
            # Python native types
            return np.dtype(type(obj)).itemsize
    elif isinstance(obj, (str, bytes)):
        return len(obj) if isinstance(obj, bytes) else len(obj.encode('utf-8'))
    else:
        # Unknown, return 0
        return 0


def _get_attribute_metadata(attr_val) -> dict:
    """
    Extract metadata for a single attribute.

    Args:
        attr_val: Value of the attribute

    Returns:
        Dictionary containing attribute metadata (without the name key)
    """
    attr_info = {
        "dtype": str(type(attr_val).__name__) if not hasattr(attr_val, 'dtype') else str(attr_val.dtype),
    }

    # Add size estimate
    size_bytes = _estimate_size_bytes(attr_val)
    if size_bytes > 0:
        attr_info["size_bytes"] = size_bytes

    # Add shape for array attributes
    if hasattr(attr_val, 'shape'):
        attr_info["shape"] = attr_val.shape

        # Add numeric statistics if applicable
        stats = _get_numeric_statistics(attr_val)
        if stats:
            attr_info.update(stats)

        # Only include value for small arrays
        if attr_val.size < NUM_ELEMS_PREVIEW_THRESHOLD:
            attr_info["value"] = attr_val.tolist() if hasattr(attr_val, 'tolist') else str(attr_val)
        else:
            attr_info["note"] = f"Large array ({attr_val.size} elements) - value omitted"
    else:
        # Scalar values - check if it's a reasonable size
        if isinstance(attr_val, (str, bytes)):
            if len(attr_val) < BYTE_SIZE_PREVIEW_THRESHOLD:  # Small strings/bytes
                attr_info["value"] = attr_val.decode() if isinstance(attr_val, bytes) else attr_val
            else:
                attr_info["note"] = f"Large string ({len(attr_val)} chars) - value omitted"
        elif isinstance(attr_val, (int, float, bool, np.integer, np.floating, np.bool_)):
            attr_info["value"] = _convert_to_json_serializable(attr_val)
        else:
            # Other types - try to serialize reasonably
            str_val = str(attr_val)
            if len(str_val) < BYTE_SIZE_PREVIEW_THRESHOLD:
                attr_info["value"] = str_val
            else:
                attr_info["note"] = "Complex value - omitted"

    return attr_info


def _get_dataset_metadata(obj: h5py.Dataset) -> dict:
    """
    Extract metadata specific to a Dataset object.

    Args:
        obj: h5py Dataset object

    Returns:
        Dictionary containing dataset-specific metadata
    """
    metadata = {
        "type": "dataset",
        "shape": obj.shape,
        "dtype": str(obj.dtype),
        "size": obj.size
    }

    # Add size in bytes
    size_bytes = _estimate_size_bytes(obj)
    if size_bytes > 0:
        metadata["size_bytes"] = size_bytes

    # Storage info
    if obj.chunks:
        metadata["chunks"] = obj.chunks
    if obj.compression:
        metadata["compression"] = obj.compression
        metadata["compression_opts"] = obj.compression_opts
    if obj.fillvalue is not None:
        metadata["fillvalue"] = _convert_to_json_serializable(obj.fillvalue)
    metadata["maxshape"] = obj.maxshape

    # Add numeric statistics for numeric datasets
    # Only compute for reasonably sized datasets to avoid excessive I/O
    MAX_ELEMENTS_FOR_STATS = 10_000_000  # 10M elements
    if obj.size <= MAX_ELEMENTS_FOR_STATS:
        try:
            if np.issubdtype(obj.dtype, np.number):
                # Read the entire dataset to compute statistics
                data = obj[:]
                stats = _get_numeric_statistics(data)
                if stats:
                    metadata.update(stats)
        except Exception as e:
            # If we can't read the data for any reason, just skip statistics
            metadata["statistics_note"] = f"Could not compute statistics: {str(e)}"
    else:
        metadata["statistics_note"] = f"Dataset too large ({obj.size} elements) for automatic statistics"

    return metadata


def _get_group_metadata(obj: h5py.Group, object_path: str) -> dict:
    """
    Extract metadata specific to a Group or File object.

    Args:
        obj: h5py Group object
        object_path: Path to the object within the file

    Returns:
        Dictionary containing group-specific metadata
    """
    metadata = {}
    if object_path == '/':
        metadata["type"] = "file_root"
    else:
        metadata["type"] = "group"
    metadata["members"] = list(obj.keys())
    metadata["num_members"] = len(obj)

    return metadata


def _get_committed_datatype_metadata(obj: h5py.Datatype) -> dict:
    """
    Extract metadata specific to a committed datatype object.

    Args:
        obj: h5py Datatype object

    Returns:
        Dictionary containing committed datatype metadata
    """
    return {
        "type": "committed_datatype",
        "dtype": str(obj.dtype)
    }


def _get_link_metadata(link_info, object_path: str) -> dict:
    """
    Extract metadata for soft or external links.

    Args:
        link_info: h5py link info object (SoftLink or ExternalLink)
        object_path: Path to the link within the file

    Returns:
        Dictionary containing link metadata, or None if not a link
    """
    if isinstance(link_info, h5py.SoftLink):
        return {
            "type": "soft_link",
            "name": object_path,
            "target": link_info.path
        }
    elif isinstance(link_info, h5py.ExternalLink):
        return {
            "type": "external_link",
            "name": object_path,
            "filename": link_info.filename,
            "target": link_info.path
        }
    return None


def _parse_object_path(object_path: str) -> tuple[str, str]:
    """
    Parse an object path into parent path and object name.

    Args:
        object_path: Full path to the object (e.g., "/group/dataset")

    Returns:
        Tuple of (parent_path, object_name)
    """
    if object_path == '/':
        return '/', ''

    # Split path and get components
    parts = object_path.rstrip('/').split('/')
    object_name = parts[-1]

    # Reconstruct parent path
    if len(parts) == 2:  # e.g., "/dataset"
        parent_path = '/'
    else:
        parent_path = '/'.join(parts[:-1])

    return parent_path, object_name


def _construct_smd_attribute_name(object_name: str) -> str:
    """
    Construct the semantic metadata attribute name for an object.

    Args:
        object_name: Name of the HDF5 object

    Returns:
        Semantic metadata attribute name
    """
    if not object_name:  # Root group case
        return "ahdf5-smd-root"
    return f"ahdf5-smd-{object_name}"


def _prefix_best_guess(smd_value: str) -> str:
    """
    Prefix each line of semantic metadata with "BEST GUESS: ".

    Args:
        smd_value: The semantic metadata string

    Returns:
        String with each line prefixed
    """
    lines = smd_value.split('\n')
    prefixed_lines = [f"BEST GUESS: {line}" if line.strip() else line
                      for line in lines]
    return '\n'.join(prefixed_lines)


def _has_smd(h5_file: h5py.File, object_path: str) -> bool:
    """
    Check if an object already has semantic metadata.

    Args:
        h5_file: Open h5py File object
        object_path: Path to the object within the file

    Returns:
        True if SMD exists, False otherwise
    """
    try:
        parent_path, object_name = _parse_object_path(object_path)

        if object_path == '/':
            parent = h5_file['/']
        else:
            parent = h5_file[parent_path]

        smd_attr_name = _construct_smd_attribute_name(object_name)
        return smd_attr_name in parent.attrs
    except Exception:
        return False
