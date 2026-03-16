import h5py
from h5py_helpers import _parse_object_path, _construct_smd_attribute_name, _prefix_best_guess

try:
    from tools.h5py.registry import hdf5_tool
except ImportError:
    def hdf5_tool(**_kw):
        return lambda f: f


@hdf5_tool(
    category="semantic-metadata",
    keywords=["semantic", "metadata", "smd", "write", "document", "annotate", "describe", "label"],
    use_cases=[
        "Documenting what a dataset contains",
        "Adding human-readable descriptions",
        "Annotating imported data",
        "Recording provenance information",
    ],
)
def write_semantic_metadata(filepath: str, object_path: str,
                            smd_value: str, is_best_guess: bool = True) -> str:
    """
    Write/update SMD text description for one HDF5 object. Document, annotate, or label datasets/groups.

    Prefixes with "BEST GUESS: " by default (disable for verified info). Single object only.

    Args:
        filepath: Path to the HDF5 file
        object_path: Path to the object within the file (e.g., "/group/dataset")
        smd_value: The semantic metadata string to write
        is_best_guess: If True, prefix each line with "BEST GUESS: " (default: True)
            Use True when:
            - Working with external/imported HDF5 files where provenance is unknown
            - Inferring metadata from object names, structure, or data values
            - Making educated guesses about units, purpose, or context
            - Analyzing data patterns to infer semantic meaning
            Use False only when:
            - User has explicitly provided definite information
            - Metadata comes from verified ground truth sources
            - Information is certain and should not be questioned

    Returns:
        Success message or error message
    """
    try:
        with h5py.File(filepath, 'r+') as f:
            # Verify the object exists
            if object_path not in f:
                return f"Error: Object '{object_path}' not found in file '{filepath}'"

            # Parse the object path
            parent_path, object_name = _parse_object_path(object_path)

            # Get the parent group (where the SMD attribute will be stored)
            if object_path == '/':
                # Root group stores its own SMD
                parent = f['/']
            else:
                parent = f[parent_path]

            # Construct the SMD attribute name
            smd_attr_name = _construct_smd_attribute_name(object_name)

            # Apply best-guess prefix if requested
            if is_best_guess:
                smd_value = _prefix_best_guess(smd_value)

            # Write the SMD attribute (overwrites if exists)
            parent.attrs[smd_attr_name] = smd_value

            return f"Semantic metadata written successfully for '{object_path}'"

    except KeyError:
        return f"Error: Object '{object_path}' not found in file '{filepath}'"
    except OSError as e:
        if 'read-only' in str(e).lower() or 'permission' in str(e).lower():
            return f"Error: File '{filepath}' is read-only or lacks write permissions"
        return f"Error writing semantic metadata: {str(e)}"
    except Exception as e:
        return f"Error writing semantic metadata: {str(e)}"
