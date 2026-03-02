import h5py
from typing import Optional, Literal

try:
    from tools.h5py.registry import hdf5_tool
except ImportError:
    def hdf5_tool(**_kw):
        return lambda f: f


@hdf5_tool(
    category="optimization",
    keywords=["compress", "compression", "filter", "gzip", "szip", "shuffle", "fletcher32", "reduce size", "decompress"],
    use_cases=[
        "Compressing datasets to reduce file size",
        "Adding data integrity checksums",
        "Removing compression",
        "Applying shuffle filter to improve compression",
    ],
)
def apply_filter_dataset(
    filepath: str,
    object_path: str,
    output_filepath: Optional[str] = None,
    filter_type: Optional[Literal['gzip', 'szip', 'shuffle', 'fletcher32', 'nbit',
                                   'scaleoffset', 'none']] = None,
    compression_level: Optional[int] = None,
    szip_options: Optional[str] = None,
    scaleoffset_params: Optional[str] = None,
    remove_all_filters: bool = False
) -> dict:
    """
    Apply, modify, or remove compression and filter settings on an HDF5 dataset.

    Supports gzip, szip, shuffle, fletcher32, nbit, and scaleoffset filters. Creates
    a NEW file using h5repack (original NOT modified). Call get_object_metadata() first
    to understand current filter configuration.

    Filter descriptions:
    - gzip: General-purpose compression (level 1-9, higher=better compression but slower)
    - szip: NASA-developed compression, patent-restricted (format: "pixels_per_block,coding" e.g. "8,NN")
    - shuffle: Rearranges byte order to improve compression (use with gzip)
    - fletcher32: Adds checksum for data integrity verification
    - nbit: Packs data to use minimum bits
    - scaleoffset: Lossy compression (format: "scale_factor,scale_type" e.g. "2,IN")
    - none: Remove all filters (decompress)

    Args:
        filepath: Path to input HDF5 file
        object_path: Path to dataset to filter (e.g., "/data/measurements")
        output_filepath: Path for output file (default: adds "_filtered" suffix)
        filter_type: Type of filter to apply
        compression_level: For gzip only, compression level 1-9 (default: 6)
        szip_options: For szip, format "pixels_per_block,coding" (e.g., "8,NN" or "16,EC")
        scaleoffset_params: For scaleoffset, format "scale_factor,scale_type" (e.g., "2,IN" or "3,DS")
        remove_all_filters: If True, remove all existing filters (decompress)

    Returns:
        Dictionary with:
        - success: bool
        - output_filepath: str (path to new file)
        - original_filters: dict of current filters
        - new_filters: dict of filters in output file
        - message: description of what was done
        - error: str (only if success=False)
    """
    import subprocess
    import os

    try:
        # Validate inputs
        if not os.path.exists(filepath):
            return {
                "success": False,
                "error": f"Input file not found: {filepath}"
            }

        # Check filter availability
        import h5py.h5z as h5z
        filter_availability = {
            'gzip': h5z.filter_avail(h5z.FILTER_DEFLATE),
            'shuffle': h5z.filter_avail(h5z.FILTER_SHUFFLE),
            'fletcher32': h5z.filter_avail(h5z.FILTER_FLETCHER32),
            'szip': h5z.filter_avail(h5z.FILTER_SZIP),
            'nbit': h5z.filter_avail(h5z.FILTER_NBIT),
            'scaleoffset': h5z.filter_avail(h5z.FILTER_SCALEOFFSET),
        }

        # Get current dataset metadata and filters
        with h5py.File(filepath, 'r') as f:
            if object_path not in f:
                return {
                    "success": False,
                    "error": f"Dataset '{object_path}' not found in file"
                }

            obj = f[object_path]
            if not isinstance(obj, h5py.Dataset):
                return {
                    "success": False,
                    "error": f"Object '{object_path}' is not a dataset"
                }

            # Get current filter information
            original_filters = {
                'compression': obj.compression,
                'compression_opts': obj.compression_opts,
                'shuffle': obj.shuffle,
                'fletcher32': obj.fletcher32,
                'scaleoffset': obj.scaleoffset,
            }
            dataset_shape = obj.shape

        # Generate output filepath if not provided
        if output_filepath is None:
            base, ext = os.path.splitext(filepath)
            output_filepath = f"{base}_filtered{ext}"

        # Check if output file already exists
        if os.path.exists(output_filepath):
            return {
                "success": False,
                "error": f"Output file already exists: {output_filepath}. Please specify a different output path or remove the existing file."
            }

        # Determine the filter argument for h5repack
        filter_arg = None
        filter_desc = None

        if remove_all_filters:
            # Remove all filters
            filter_arg = f"{object_path}:NONE"
            filter_desc = "remove all filters"

        elif filter_type == 'none':
            # Explicit none
            filter_arg = f"{object_path}:NONE"
            filter_desc = "remove all filters"

        elif filter_type == 'gzip':
            # Check availability
            if not filter_availability['gzip']:
                return {
                    "success": False,
                    "error": "GZIP filter is not available on this system",
                    "filter_availability": filter_availability
                }

            # Default compression level
            if compression_level is None:
                compression_level = 6
            elif not (1 <= compression_level <= 9):
                return {
                    "success": False,
                    "error": f"Invalid gzip compression level: {compression_level}. Must be 1-9."
                }

            filter_arg = f"{object_path}:GZIP={compression_level}"
            filter_desc = f"GZIP compression level {compression_level}"

        elif filter_type == 'szip':
            # Check availability
            if not filter_availability['szip']:
                return {
                    "success": False,
                    "error": "SZIP filter is not available on this system (patent-restricted, requires special build)",
                    "filter_availability": filter_availability
                }

            if not szip_options:
                return {
                    "success": False,
                    "error": "SZIP requires szip_options parameter (format: 'pixels_per_block,coding' e.g., '8,NN' or '16,EC')"
                }

            filter_arg = f"{object_path}:SZIP={szip_options}"
            filter_desc = f"SZIP compression ({szip_options})"

        elif filter_type == 'shuffle':
            # Check availability
            if not filter_availability['shuffle']:
                return {
                    "success": False,
                    "error": "Shuffle filter is not available on this system",
                    "filter_availability": filter_availability
                }

            filter_arg = f"{object_path}:SHUF"
            filter_desc = "Shuffle filter"

        elif filter_type == 'fletcher32':
            # Check availability
            if not filter_availability['fletcher32']:
                return {
                    "success": False,
                    "error": "Fletcher32 filter is not available on this system",
                    "filter_availability": filter_availability
                }

            filter_arg = f"{object_path}:FLET"
            filter_desc = "Fletcher32 checksum filter"

        elif filter_type == 'nbit':
            # Check availability
            if not filter_availability['nbit']:
                return {
                    "success": False,
                    "error": "N-bit filter is not available on this system",
                    "filter_availability": filter_availability
                }

            filter_arg = f"{object_path}:NBIT"
            filter_desc = "N-bit filter"

        elif filter_type == 'scaleoffset':
            # Check availability
            if not filter_availability['scaleoffset']:
                return {
                    "success": False,
                    "error": "Scale-offset filter is not available on this system",
                    "filter_availability": filter_availability
                }

            if not scaleoffset_params:
                return {
                    "success": False,
                    "error": "Scale-offset requires scaleoffset_params parameter (format: 'scale_factor,scale_type' e.g., '2,IN' or '3,DS')"
                }

            filter_arg = f"{object_path}:SOFF={scaleoffset_params}"
            filter_desc = f"Scale-offset filter ({scaleoffset_params})"

        else:
            return {
                "success": False,
                "error": "Must specify either filter_type or remove_all_filters=True"
            }

        # First, verify h5repack is available and get version info
        version_result = subprocess.run(
            ["h5repack", "-V"],
            capture_output=True,
            text=True,
            timeout=10
        )
        h5repack_version = version_result.stdout.strip() if version_result.returncode == 0 else "unknown"

        # Verify input file is readable by h5repack using h5dump
        dump_check = subprocess.run(
            ["h5dump", "-H", filepath],
            capture_output=True,
            text=True,
            timeout=30
        )
        if dump_check.returncode != 0:
            return {
                "success": False,
                "error": f"Input file appears corrupted or unreadable by HDF5 tools: {dump_check.stderr}"
            }

        # Build h5repack command with verbose flag
        cmd = [
            "h5repack",
            "-v",  # Verbose output to see what h5repack is doing
            "-i", filepath,
            "-o", output_filepath,
            "-f", filter_arg
        ]

        # Execute h5repack
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        # Capture full output for diagnostics
        h5repack_stdout = result.stdout if result.stdout else ""
        h5repack_stderr = result.stderr if result.stderr else ""

        if result.returncode != 0:
            # h5repack failed
            error_msg = h5repack_stderr if h5repack_stderr else h5repack_stdout

            # Clean up partial output file if it exists
            if os.path.exists(output_filepath):
                try:
                    os.remove(output_filepath)
                except:
                    pass

            return {
                "success": False,
                "error": f"h5repack failed (exit code {result.returncode}): {error_msg}",
                "command": ' '.join(cmd),
                "h5repack_version": h5repack_version
            }

        # Verify the output file was created
        if not os.path.exists(output_filepath):
            return {
                "success": False,
                "error": "h5repack completed but output file was not created"
            }

        # Verify the new filters in the output file
        try:
            with h5py.File(output_filepath, 'r') as f:
                # First check what's actually in the output file
                def list_all_objects(name, obj):
                    return name
                available_paths = []
                f.visititems(lambda name, obj: available_paths.append(name))

                new_obj = f[object_path]
                new_filters = {
                    'compression': new_obj.compression,
                    'compression_opts': new_obj.compression_opts,
                    'shuffle': new_obj.shuffle,
                    'fletcher32': new_obj.fletcher32,
                    'scaleoffset': new_obj.scaleoffset,
                }
        except Exception as e:
            # Include diagnostic info about what's in the file
            try:
                with h5py.File(output_filepath, 'r') as f:
                    available_paths = []
                    f.visititems(lambda name, obj: available_paths.append(name))
                    diagnostic_info = f"Available paths in output file: {available_paths}, Expected: {object_path}"
            except:
                diagnostic_info = "Could not list output file contents"

            return {
                "success": False,
                "error": f"Output file created but could not verify filters: {repr(e)}. {diagnostic_info}",
                "output_filepath": output_filepath,
                "h5repack_version": h5repack_version,
                "h5repack_stdout": h5repack_stdout,
                "h5repack_stderr": h5repack_stderr,
                "command": ' '.join(cmd)
            }

        return {
            "success": True,
            "output_filepath": output_filepath,
            "dataset_path": object_path,
            "dataset_shape": dataset_shape,
            "original_filters": original_filters,
            "new_filters": new_filters,
            "message": f"Successfully applied {filter_desc} to dataset '{object_path}'",
            "h5repack_version": h5repack_version,
            "h5repack_stdout": h5repack_stdout,
            "h5repack_stderr": h5repack_stderr
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "h5repack operation timed out (>5 minutes)"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }
