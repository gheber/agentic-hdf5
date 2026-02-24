import h5py
from h5py_helpers import _parse_object_path, _construct_smd_attribute_name

try:
    from tools.h5py.registry import hdf5_tool
except ImportError:
    def hdf5_tool(**_kw):
        return lambda f: f


@hdf5_tool(
    category="semantic-metadata",
    keywords=["semantic", "metadata", "smd", "read", "description", "documentation", "meaning", "context"],
    use_cases=[
        "Understanding what a dataset represents",
        "Reading human-readable descriptions",
        "Checking if SMD exists for an object",
        "Getting context before visualization",
    ],
)
def read_semantic_metadata(filepath: str, object_path: str) -> str:
    """
    Read the semantic metadata attribute associated with a given object.

    Returns the SMD text content if it exists, or an error/info message if the
    object or its SMD attribute is not found. Acts on a single object only
    (not recursive).

    Args:
        filepath: Path to the HDF5 file
        object_path: Path to the object within the file (e.g., "/group/dataset")

    Returns:
        The semantic metadata string if it exists, or an error/info message
    """
    try:
        with h5py.File(filepath, 'r') as f:
            # Verify the object exists
            if object_path not in f:
                return f"Error: Object '{object_path}' not found in file '{filepath}'"

            # Parse the object path
            parent_path, object_name = _parse_object_path(object_path)

            # Get the parent group (where the SMD attribute is stored)
            if object_path == '/':
                # Root group stores its own SMD
                parent = f['/']
            else:
                parent = f[parent_path]

            # Construct the SMD attribute name
            smd_attr_name = _construct_smd_attribute_name(object_name)

            # Check if the SMD attribute exists
            if smd_attr_name not in parent.attrs:
                return f"No semantic metadata found for object '{object_path}'"

            # Read and return the SMD value
            smd_value = parent.attrs[smd_attr_name]

            # Decode if it's bytes
            if isinstance(smd_value, bytes):
                smd_value = smd_value.decode('utf-8')

            return smd_value

    except KeyError:
        return f"Error: Object '{object_path}' not found in file '{filepath}'"
    except Exception as e:
        return f"Error reading semantic metadata: {str(e)}"
