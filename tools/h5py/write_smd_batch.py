import h5py
from h5py_helpers import _parse_object_path, _construct_smd_attribute_name, _prefix_best_guess

try:
    from tools.h5py.registry import hdf5_tool
except ImportError:
    def hdf5_tool(**_kw):
        return lambda f: f


# TBD - Impose bottom-up ordering for context optimization (e.g. file summary is more knowledgeable)
@hdf5_tool(
    category="semantic-metadata",
    keywords=["batch", "write", "multiple", "bulk", "smd", "semantic metadata", "transaction"],
    use_cases=[
        "Writing SMD for multiple objects at once",
        "Batch documentation workflows",
        "Efficiently annotating large files",
        "Pairing with collect_objects_for_smd()",
    ],
)
def write_smd_batch(filepath: str, smd_map: dict, is_best_guess: bool = True) -> dict:
    """
    Write SMD for multiple objects in one transaction. Pair with collect_objects_for_smd() for bulk annotation.

    Continues on individual failures, returns per-object success/failure details.

    Args:
        filepath: Path to the HDF5 file
        smd_map: Dictionary mapping object paths to SMD content
                 Example: {"/dataset1": "Temperature data...", "/group1": "Experiment group..."}
        is_best_guess: If True, prefix each line with "BEST GUESS: " (default: True)

    Returns:
        Dictionary containing:
        - success_count: Number of objects successfully written
        - failure_count: Number of objects that failed
        - failures: List of dicts with 'path' and 'error' for each failure
        - successes: List of object paths successfully written
    """
    successes = []
    failures = []

    try:
        with h5py.File(filepath, 'r+') as f:
            for object_path, smd_content in smd_map.items():
                try:
                    # Verify the object exists
                    if object_path not in f:
                        failures.append({
                            "path": object_path,
                            "error": f"Object not found in file"
                        })
                        continue

                    # Parse the object path
                    parent_path, object_name = _parse_object_path(object_path)

                    # Get the parent group (where the SMD attribute will be stored)
                    if object_path == '/':
                        parent = f['/']
                    else:
                        parent = f[parent_path]

                    # Construct the SMD attribute name
                    smd_attr_name = _construct_smd_attribute_name(object_name)

                    # Apply best-guess prefix if requested
                    final_smd_content = smd_content
                    if is_best_guess:
                        final_smd_content = _prefix_best_guess(smd_content)

                    # Write the SMD attribute
                    parent.attrs[smd_attr_name] = final_smd_content

                    successes.append(object_path)

                except Exception as e:
                    failures.append({
                        "path": object_path,
                        "error": str(e)
                    })

        return {
            "success_count": len(successes),
            "failure_count": len(failures),
            "successes": successes,
            "failures": failures
        }

    except OSError as e:
        if 'read-only' in str(e).lower() or 'permission' in str(e).lower():
            return {
                "success_count": 0,
                "failure_count": len(smd_map),
                "successes": [],
                "failures": [{"path": "ALL", "error": f"File is read-only or lacks write permissions"}]
            }
        return {
            "success_count": 0,
            "failure_count": len(smd_map),
            "successes": [],
            "failures": [{"path": "ALL", "error": f"Error opening file: {str(e)}"}]
        }
    except Exception as e:
        return {
            "success_count": 0,
            "failure_count": len(smd_map),
            "successes": [],
            "failures": [{"path": "ALL", "error": f"Unexpected error: {str(e)}"}]
        }
