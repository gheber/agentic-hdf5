# Agentic HDF5 Plugin

You have access to HDF5 expertise through this plugin's skills and MCP tools. Use them when working with HDF5 files (.h5, .hdf5, .nc/NetCDF backed by HDF5).

## When to use skills

Skills contain deep domain knowledge. Load a skill when:
- The user asks about HDF5 concepts (chunking, compression, parallel I/O, VFDs, VOL connectors, SWMR, VDS, etc.)
- You need to make architectural decisions about HDF5 file layout, storage strategy, or access patterns
- The task involves scientific data workflows: publishing, replication, cloud optimization, visualization
- You're unsure about HDF5 best practices for a specific scenario

Don't load skills for straightforward h5py read/write operations you already know how to do.

## When to use MCP tools

MCP tools (`hdf5-tools` server) perform actions on actual HDF5 files. Use them when:
- **Inspecting files**: `get_object_metadata` — get structure, datasets, attributes, shapes, dtypes
- **Optimizing storage**: `rechunk_dataset`, `apply_filter_dataset` — change chunk layout or compression
- **Visualizing data**: `visualize` — generate plots directly from HDF5 datasets
- **Semantic metadata**: `read_semantic_metadata`, `write_semantic_metadata`, `collect_objects_for_smd`, `write_smd_batch` — manage human-readable descriptions attached to HDF5 objects
- **Semantic search**: `vectorize_semantic_metadata`, `query_semantic_metadata` — embed and search metadata via natural language

## Skill vs MCP tool decision

- **Need to understand how/why?** → Load a skill (e.g., which compression filter to choose)
- **Need to do something to a file?** → Use an MCP tool (e.g., actually apply that filter)
- **Both?** → Load the skill first for guidance, then use MCP tools to execute

## General HDF5 guidance

- Always use context managers (`with h5py.File(...) as f:`) to ensure files are closed
- Prefer chunked datasets when compression, partial reads, or resizing are needed
- Batch writes — avoid element-by-element I/O
- When exploring an unfamiliar HDF5 file, start with `get_object_metadata` to understand its structure before reading data
