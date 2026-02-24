#!/usr/bin/env python3
"""
MCP server exposing agentic-hdf5 tools.

Run directly:
    python -m tools.mcp_server

Or via console script (after pip install):
    hdf5-mcp-server
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Literal, Optional

from mcp.server.fastmcp import FastMCP

# Ensure tools/h5py bare imports resolve
_h5py_dir = str(Path(__file__).resolve().parent / "h5py")
if _h5py_dir not in sys.path:
    sys.path.insert(0, _h5py_dir)

# Import all tool functions (also populates the registry)
from tools.h5py import (
    get_object_metadata as _get_object_metadata,
    rechunk_dataset as _rechunk_dataset,
    apply_filter_dataset as _apply_filter_dataset,
    visualize as _visualize,
    read_semantic_metadata as _read_semantic_metadata,
    write_semantic_metadata as _write_semantic_metadata,
    collect_objects_for_smd as _collect_objects_for_smd,
    write_smd_batch as _write_smd_batch,
    vectorize_semantic_metadata as _vectorize_semantic_metadata,
    query_semantic_metadata as _query_semantic_metadata,
)

server = FastMCP("hdf5-tools")


def _check_h5repack() -> dict | None:
    """Return an error dict if h5repack is not available, else None."""
    if shutil.which("h5repack") is None:
        return {
            "status": "error",
            "message": (
                "h5repack not found on PATH. Install HDF5 command-line tools: "
                "apt install hdf5-tools (Debian/Ubuntu), "
                "brew install hdf5 (macOS), "
                "or conda install hdf5."
            ),
        }
    return None


# ---------- Tool wrappers ----------


@server.tool()
def get_object_metadata(filepath: str, object_path: str) -> dict:
    """Retrieve comprehensive metadata for a specific HDF5 object (shape, dtype, chunks, compression, attributes)."""
    return _get_object_metadata(filepath, object_path)


@server.tool()
def rechunk_dataset(
    filepath: str,
    object_path: str,
    output_filepath: Optional[str] = None,
    chunk_adjustment: Optional[Literal["larger", "smaller", "half", "double"]] = None,
    chunk_dims: Optional[str] = None,
    make_contiguous: bool = False,
) -> dict:
    """Modify the chunking layout of an HDF5 dataset. Creates a NEW file (original not modified). Requires h5repack."""
    err = _check_h5repack()
    if err:
        return err
    return _rechunk_dataset(
        filepath, object_path, output_filepath, chunk_adjustment, chunk_dims, make_contiguous
    )


@server.tool()
def apply_filter_dataset(
    filepath: str,
    object_path: str,
    output_filepath: Optional[str] = None,
    filter_type: Optional[
        Literal["gzip", "szip", "shuffle", "fletcher32", "nbit", "scaleoffset", "none"]
    ] = None,
    compression_level: Optional[int] = None,
    szip_options: Optional[str] = None,
    scaleoffset_params: Optional[str] = None,
    remove_all_filters: bool = False,
) -> dict:
    """Apply, modify, or remove compression/filter settings on an HDF5 dataset. Creates a NEW file. Requires h5repack."""
    err = _check_h5repack()
    if err:
        return err
    return _apply_filter_dataset(
        filepath,
        object_path,
        output_filepath,
        filter_type,
        compression_level,
        szip_options,
        scaleoffset_params,
        remove_all_filters,
    )


@server.tool()
def visualize(
    filepath: str,
    object_path: str,
    plot_type: Optional[
        Literal["auto", "line", "scatter", "hist", "pcolormesh", "imshow", "contour", "contourf"]
    ] = "auto",
    hdf5_slices: Optional[str] = None,
    x: Optional[str] = None,
    y: Optional[str] = None,
    xscale: str = "linear",
    yscale: str = "linear",
    xlim: Optional[str] = None,
    ylim: Optional[str] = None,
    save_path: Optional[str] = None,
) -> dict:
    """Generate a visualization (PNG) of an HDF5 dataset. Auto-selects plot type based on data shape and metadata."""
    # Type coercion: hdf5_slices comes as JSON string with string keys → Dict[int, Any]
    parsed_slices = None
    if hdf5_slices is not None:
        raw = json.loads(hdf5_slices) if isinstance(hdf5_slices, str) else hdf5_slices
        parsed_slices = {int(k): v for k, v in raw.items()}

    # Type coercion: xlim/ylim come as JSON arrays → tuple[float, float]
    parsed_xlim = None
    if xlim is not None:
        arr = json.loads(xlim) if isinstance(xlim, str) else xlim
        parsed_xlim = tuple(arr)

    parsed_ylim = None
    if ylim is not None:
        arr = json.loads(ylim) if isinstance(ylim, str) else ylim
        parsed_ylim = tuple(arr)

    return _visualize(
        filepath,
        object_path,
        plot_type,
        parsed_slices,
        x,
        y,
        xscale,
        yscale,
        parsed_xlim,
        parsed_ylim,
        save_path,
    )


@server.tool()
def read_semantic_metadata(filepath: str, object_path: str) -> str:
    """Read the semantic metadata attribute for an HDF5 object."""
    return _read_semantic_metadata(filepath, object_path)


@server.tool()
def write_semantic_metadata(
    filepath: str, object_path: str, smd_value: str, is_best_guess: bool = True
) -> str:
    """Write or update semantic metadata for a single HDF5 object."""
    return _write_semantic_metadata(filepath, object_path, smd_value, is_best_guess)


@server.tool()
def collect_objects_for_smd(
    filepath: str, object_path: str = "/", max_depth: int = -1
) -> dict:
    """Scan HDF5 file structure and collect objects that lack semantic metadata."""
    return _collect_objects_for_smd(filepath, object_path, max_depth)


@server.tool()
def write_smd_batch(filepath: str, smd_map: str, is_best_guess: bool = True) -> dict:
    """Write semantic metadata for multiple HDF5 objects in a single transaction."""
    # Type coercion: smd_map arrives as JSON string
    parsed_map = json.loads(smd_map) if isinstance(smd_map, str) else smd_map
    return _write_smd_batch(filepath, parsed_map, is_best_guess)


@server.tool()
def vectorize_semantic_metadata(
    filepath: str,
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    rebuild: bool = False,
    object_paths: Optional[str] = None,
) -> dict:
    """Convert text-based semantic metadata into vector embeddings for semantic search. Requires sentence-transformers."""
    # Type coercion: object_paths arrives as JSON array string
    parsed_paths = None
    if object_paths is not None:
        parsed_paths = json.loads(object_paths) if isinstance(object_paths, str) else object_paths
    return _vectorize_semantic_metadata(filepath, embedder_model, rebuild, parsed_paths)


@server.tool()
def query_semantic_metadata(
    filepath: str,
    query_text: str,
    top_k: int = 5,
    object_filter: Optional[str] = None,
    min_score: float = 0.0,
    embedder_model: Optional[str] = None,
) -> dict:
    """Perform natural language semantic search over vectorized HDF5 metadata. Requires sentence-transformers."""
    return _query_semantic_metadata(
        filepath, query_text, top_k, object_filter, min_score, embedder_model
    )


def main():
    server.run()


if __name__ == "__main__":
    main()
