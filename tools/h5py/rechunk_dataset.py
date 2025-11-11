import h5py
import subprocess
import os
from typing import Optional, Literal


def rechunk_dataset(
    filepath: str,
    object_path: str,
    output_filepath: Optional[str] = None,
    chunk_adjustment: Optional[Literal['larger', 'smaller', 'half', 'double']] = None,
    chunk_dims: Optional[str] = None,
    make_contiguous: bool = False
) -> dict:
    """
    Modify the chunking layout of an HDF5 dataset by creating a new file.

    Supports high-level adjustments ('larger', 'smaller', 'half', 'double') or exact
    chunk dimension specifications. Can also convert chunked datasets to contiguous layout.
    Creates a NEW file using h5repack (original NOT modified). Call get_object_metadata()
    first to understand current chunk configuration.

    Usage patterns:
    1. Simple adjustments: chunk_adjustment='larger' or 'smaller' (doubles/halves all dims)
    2. Specific multipliers: chunk_adjustment='double' or 'half'
    3. Exact dimensions: chunk_dims='10x20x30' (for 3D dataset)
    4. Remove chunking: make_contiguous=True

    Args:
        filepath: Path to input HDF5 file
        object_path: Path to dataset to rechunk (e.g., "/data/temperature")
        output_filepath: Path for output file (default: adds "_rechunked" suffix to input)
        chunk_adjustment: High-level chunk size adjustment ('larger', 'smaller', 'half', 'double')
        chunk_dims: Exact chunk dimensions as string (e.g., "100x100" or "10x20x30")
        make_contiguous: If True, convert to contiguous layout (removes chunking)

    Returns:
        Dictionary with:
        - success: bool
        - output_filepath: str (path to new file)
        - original_chunks: current chunk dimensions (or None if contiguous)
        - new_chunks: new chunk dimensions (or "contiguous")
        - message: description of what was done
        - error: str (only if success=False)

    Example workflow:
        # Agent first gets metadata to see current chunking
        metadata = get_object_metadata("data.h5", "/temperature")
        # Agent sees chunks are (10, 10, 10) and dataset is (1000, 1000, 1000)
        # User says "make chunks larger"
        # Agent calls: rechunk_dataset("data.h5", "/temperature", chunk_adjustment="larger")
        # Result: new file with chunks (20, 20, 20)
    """

    try:
        # Validate inputs
        if not os.path.exists(filepath):
            return {
                "success": False,
                "error": f"Input file not found: {filepath}"
            }

        # Get current dataset metadata
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

            current_chunks = obj.chunks
            dataset_shape = obj.shape

        # Generate output filepath if not provided
        if output_filepath is None:
            base, ext = os.path.splitext(filepath)
            output_filepath = f"{base}_rechunked{ext}"

        # Check if output file already exists
        if os.path.exists(output_filepath):
            return {
                "success": False,
                "error": f"Output file already exists: {output_filepath}. Please specify a different output path or remove the existing file."
            }

        # Determine the new chunk configuration
        layout_arg = None
        new_chunks_desc = None

        if make_contiguous:
            # Convert to contiguous layout
            layout_arg = f"{object_path}:CONTI"
            new_chunks_desc = "contiguous"

        elif chunk_dims:
            # User provided exact dimensions
            layout_arg = f"{object_path}:CHUNK={chunk_dims}"
            new_chunks_desc = f"CHUNK={chunk_dims}"

        elif chunk_adjustment:
            # Calculate new chunks based on adjustment
            if current_chunks is None:
                return {
                    "success": False,
                    "error": f"Dataset is currently contiguous (not chunked). Cannot apply adjustment '{chunk_adjustment}'. Please specify exact chunk_dims instead."
                }

            # Apply the adjustment
            if chunk_adjustment in ['larger', 'double']:
                multiplier = 2
            elif chunk_adjustment in ['smaller', 'half']:
                multiplier = 0.5
            else:
                return {
                    "success": False,
                    "error": f"Unknown chunk_adjustment: {chunk_adjustment}"
                }

            # Calculate new chunk dimensions
            new_chunks = []
            for i, (chunk_dim, dataset_dim) in enumerate(zip(current_chunks, dataset_shape)):
                new_dim = max(1, int(chunk_dim * multiplier))
                # Don't make chunks larger than the dataset dimension
                new_dim = min(new_dim, dataset_dim)
                new_chunks.append(new_dim)

            # Format for h5repack
            chunk_spec = 'x'.join(str(d) for d in new_chunks)
            layout_arg = f"{object_path}:CHUNK={chunk_spec}"

        else:
            return {
                "success": False,
                "error": "Must specify either chunk_adjustment, chunk_dims, or make_contiguous=True"
            }

        # Build h5repack command
        cmd = [
            "h5repack",
            "-i", filepath,
            "-o", output_filepath,
            "-l", layout_arg
        ]

        # Execute h5repack
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            # h5repack failed
            error_msg = result.stderr if result.stderr else result.stdout

            return {
                "success": False,
                "error": f"h5repack failed (exit code {result.returncode}): {error_msg}",
                "command": ' '.join(cmd)
            }

        # Verify the output file was created
        if not os.path.exists(output_filepath):
            return {
                "success": False,
                "error": "h5repack completed but output file was not created"
            }

        # Verify the new chunking in the output file
        try:
            with h5py.File(output_filepath, 'r') as f:
                new_obj = f[object_path]
                actual_new_chunks = new_obj.chunks
        except Exception as e:
            return {
                "success": False,
                "error": f"Output file created but could not verify chunking: {str(e)}",
                "output_filepath": output_filepath
            }

        # Format current chunks for display
        if current_chunks is None:
            current_chunks_desc = "contiguous (not chunked)"
        else:
            current_chunks_desc = f"({', '.join(str(d) for d in current_chunks)})"

        # Format actual new chunks
        if actual_new_chunks is None:
            actual_new_chunks_desc = "contiguous (not chunked)"
        else:
            actual_new_chunks_desc = f"({', '.join(str(d) for d in actual_new_chunks)})"

        return {
            "success": True,
            "output_filepath": output_filepath,
            "dataset_path": object_path,
            "dataset_shape": dataset_shape,
            "original_chunks": current_chunks_desc,
            "new_chunks": actual_new_chunks_desc,
            "message": f"Successfully rechunked dataset '{object_path}' from {current_chunks_desc} to {actual_new_chunks_desc}",
            "h5repack_output": result.stdout if result.stdout else "(no output)"
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
