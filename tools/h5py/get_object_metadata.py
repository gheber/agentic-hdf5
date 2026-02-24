import h5py
from h5py_helpers import (_get_attribute_metadata, _get_dataset_metadata,
                          _get_group_metadata, _get_committed_datatype_metadata,
                          _get_link_metadata)

try:
    from tools.h5py.registry import hdf5_tool
except ImportError:
    def hdf5_tool(**_kw):
        return lambda f: f


@hdf5_tool(
    category="inspection",
    keywords=["metadata", "inspect", "info", "properties", "shape", "dtype", "chunks", "compression", "attributes"],
    use_cases=[
        "Understanding dataset structure before visualization",
        "Checking current chunk configuration before rechunking",
        "Inspecting compression settings",
        "Exploring file structure",
    ],
)
def get_object_metadata(filepath: str, object_path: str) -> dict:
    """
    Retrieve comprehensive metadata for a specific HDF5 object.

    Returns information about type, shape, datatype, chunking, compression filters,
    and other storage properties. For datasets: shape, dtype, chunks, compression
    settings, fill value. For groups: member count and names. Essential for
    understanding object properties before operations like visualization or optimization.

    Args:
        filepath: Path to the HDF5 file
        object_path: Path to the object within the file (e.g., "/group/dataset")

    Returns:
        Dictionary containing metadata appropriate for the object type
    """
    try:
        with h5py.File(filepath, 'r') as f:
            # Check if this is a link first
            parent_path = '/'.join(object_path.split('/')[:-1]) or '/'
            obj_name = object_path.split('/')[-1]

            if parent_path in f and obj_name:
                parent = f[parent_path]
                link_info = parent.get(obj_name, getlink=True)

                # Handle soft/external links
                link_metadata = _get_link_metadata(link_info, object_path)
                if link_metadata:
                    return link_metadata

            # Get the actual object
            obj = f[object_path]

            # Build metadata with shared logic
            metadata = {
                "name": object_path
            }

            # Add attribute metadata if the object has attributes
            if hasattr(obj, 'attrs') and len(obj.attrs) > 0:
                attr_metadata = {name: _get_attribute_metadata(obj.attrs[name]) for name in obj.attrs}
                metadata["attributes"] = attr_metadata

            # Get object-type-specific metadata and merge with shared metadata
            if isinstance(obj, h5py.Dataset):
                metadata.update(_get_dataset_metadata(obj))
            elif isinstance(obj, h5py.Group):
                metadata.update(_get_group_metadata(obj, object_path))
            elif isinstance(obj, h5py.Datatype):
                metadata.update(_get_committed_datatype_metadata(obj))

            return metadata

    except KeyError:
        return {"error": f"Object '{object_path}' not found in file '{filepath}'"}
    except Exception as e:
        return {"error": f"Error reading metadata: {str(e)}"}
