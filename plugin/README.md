# HDF5 Plugin for Claude Code

This plugin provides HDF5 tools and expert knowledge for agentic workflows.

## Features

- **8+ MCP tools** for HDF5 inspection, optimization, visualization, and semantic metadata
- **10+ skills** providing expert knowledge on chunking, compression, parallel I/O, cloud optimization, and more

## Installation

```bash
claude --plugin-dir ./plugin
```

## MCP Server

The plugin connects to an MCP server that exposes the HDF5 tools. For local development:

```json
{
  "hdf5-tools": {
    "command": "python",
    "args": ["/absolute/path/to/tools/mcp_server.py"]
  }
}
```

## Tools

| Tool | Category | Description |
|------|----------|-------------|
| `get_object_metadata` | inspection | Retrieve metadata for HDF5 objects |
| `rechunk_dataset` | optimization | Modify dataset chunking layout |
| `apply_filter_dataset` | optimization | Apply/remove compression filters |
| `visualize` | analysis | Generate plots of HDF5 datasets |
| `read_semantic_metadata` | semantic-metadata | Read semantic metadata attributes |
| `write_semantic_metadata` | semantic-metadata | Write semantic metadata attributes |
| `collect_objects_for_smd` | semantic-metadata | Find objects lacking metadata |
| `write_smd_batch` | semantic-metadata | Batch-write semantic metadata |
| `vectorize_semantic_metadata` | semantic-search | Create vector embeddings for search |
| `query_semantic_metadata` | semantic-search | Natural language search over metadata |
