# AGENTS.md — Agentic HDF5

## Project Overview

Agentic HDF5 provides tools and skills that enable AI agents to work with HDF5 data files. It has three layers: **tools** (Python functions for file I/O operations), an **MCP server** (exposes tools over the Model Context Protocol), and **skills** (curated knowledge documents for HDF5 best practices). It is also packaged as a **Claude Code plugin** in `plugin/`.

## Repository Structure

```
tools/                  # Python tool implementations
  h5py/                 # h5py-based tools (core logic)
    registry.py         # @hdf5_tool decorator and TOOL_REGISTRY
    __init__.py         # Package init — imports all tools, adds h5py/ to sys.path
  mcp_server.py         # MCP server with explicit wrappers for all 10 tools
  generate_catalog.py   # Generates tool_catalog.json from the registry
  tool_catalog.json     # Auto-generated tool catalog (do not edit manually)
  search_tools.py       # Tool search/selection utilities
plugin/                 # Claude Code plugin scaffold
  .claude-plugin/       # Plugin manifest (plugin.json)
  .mcp.json             # MCP server configuration
  skills/               # Copied HDF5 skills (self-contained)
docs/                   # Specifications (SMD, VSMD)
examples/               # Usage examples (filters, semantic-metadata, visualization)
tests/                  # pytest test suite
  test_registry.py      # Registry and decorator tests
  test_mcp_server.py    # MCP server tests
  agent_tool_selection/  # Eval tests for agent tool selection accuracy
pyproject.toml          # Package definition with dependencies and console script
```

## Conventions

- **Temporary files**: Prefix with `_tmp_` (e.g., `_tmp_analyze_filters.py`). Production files use normal names.
- **Tool registration**: Each tool function is decorated with `@hdf5_tool(category=..., keywords=[...], use_cases=[...])` from `tools/h5py/registry.py`. The decorator is the source of truth for tool metadata.
- **Catalog generation**: `tools/tool_catalog.json` is auto-generated from the registry — do not edit it manually. Regenerate with `python -m tools.generate_catalog`.
- **Non-destructive operations**: Rechunking and filter tools create NEW output files via `h5repack` — they do not modify the original.
- **Semantic metadata prefix**: All SMD attributes use the `ahdf5-smd-*` prefix. Vectorized SMD is stored under `/ahdf5-vsmd` in the HDF5 file.
- **Lazy imports**: `sentence-transformers` is imported inside function bodies (not at module level) so the MCP server and registry can load without the heavy ML dependency.

## Testing

```bash
# Run standard tests (excludes agent/API tests)
python -m pytest tests/

# Run agent tool selection evals (requires claude CLI + ANTHROPIC_API_KEY)
pytest -m agent tests/agent_tool_selection/ -v
```

- Tests require `h5repack` and `h5dump` on the system PATH.
- Agent tests are excluded by default via the `agent` pytest marker.
- Install: `pip install -e .` (or `pip install -e ".[search]"` for sentence-transformers)

## Key Dependencies

- `h5py`, `numpy` — HDF5 file operations
- `sentence-transformers` — Vector embeddings for semantic search (optional, install via `pip install -e ".[search]"`)
- `matplotlib`, `xarray` — Visualization
- `mcp` — Model Context Protocol SDK (for the MCP server)
- System: `h5repack`, `h5dump` (HDF5 command-line tools)

## Working with Tools

Each tool in `tools/h5py/` is a standalone Python module with a single entry-point function decorated with `@hdf5_tool(...)`. Before modifying datasets, call `get_object_metadata` first to understand the current state. When adding or modifying tools:

1. Add the `@hdf5_tool(...)` decorator with metadata to the function
2. Add the import to `tools/h5py/__init__.py`
3. Add an explicit MCP wrapper in `tools/mcp_server.py`
4. Run `python -m tools.generate_catalog` to regenerate the catalog

## MCP Server

The MCP server (`tools/mcp_server.py`) exposes all 10 tools over the Model Context Protocol. Each tool has an explicit thin wrapper that handles type coercion (JSON strings → Python dicts/tuples/lists) and graceful `h5repack` detection.

```bash
# Run directly
python -m tools.mcp_server

# Or via console script (after pip install -e .)
hdf5-mcp-server
```

## Working with Skills

**Source of truth**: `plugin/skills/` contains the canonical skill files. They are synced to `.claude/skills/` (where Claude Code discovers them) via a cross-platform script. The `skill-development/` meta-skill lives only in `.claude/skills/` (not part of the plugin).

```bash
# Sync skills from plugin/skills/ → .claude/skills/ (run after cloning or editing skills)
python scripts/sync_skills.py
```

Skills are loaded on-demand based on user intent and provide expert-level guidance on HDF5 topics (chunking, compression, parallel I/O, cloud optimization, etc.).

When adding a new HDF5 skill:
1. Create the skill directory in `plugin/skills/<name>/`
2. Run `python scripts/sync_skills.py`
