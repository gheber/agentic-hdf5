# Agentic HDF5

![Test Agentic HDF5 Tools](https://github.com/mattjala/agentic-hdf5/actions/workflows/test-ahdf5-tools.yml/badge.svg)
![GitHub tag](https://img.shields.io/github/v/tag/mattjala/agentic-hdf5)

A set of expansions and tools for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) that enable AI agents to work at a high level with HDF5 data and files. Provides 10+ tools, 14+ skills, support for semantic metadata, and natural language search of vectorized semantic metadata.

## Prerequisites

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) must be installed. The MCP server is fetched and run via [`uvx`](https://docs.astral.sh/uv/), so `uv` must also be installed — all Python dependencies (h5py, numpy, matplotlib, etc.) are resolved automatically.

## Installation

Run these [slash commands](https://docs.anthropic.com/en/docs/claude-code/cli-reference#slash-commands) inside a Claude Code session to register the plugin marketplace and install the plugin:

```bash
# Add the marketplace (one-time)
/plugin marketplace add mattjala/agentic-hdf5

# Install the plugin
/plugin install ahdf5-plugin@agentic-hdf5
```

This gives you all 14+ skills and 10 MCP tools automatically.

For development/testing, clone the repo and load the plugin directly:

```bash
git clone https://github.com/mattjala/agentic-hdf5.git
claude --plugin-dir ./agentic-hdf5/plugin
```

## Architecture

Agentic HDF5 is composed of two complementary layers — **tools** and **skills** — that can be used independently or together.

### Tools

Tools are Python functions that agents call to perform concrete operations on HDF5 files. They handle the actual file I/O: reading metadata, rechunking datasets, applying compression filters, writing semantic metadata, generating visualizations, and running semantic searches. Tools live in the `tools/` directory and are registered in `tools/tool_catalog.json`.

| Tool | Description |
|------|-------------|
| `get_object_metadata` | Inspect dataset/group properties (shape, dtype, chunks, compression) |
| `rechunk_dataset` | Modify chunk layout (larger, smaller, exact dimensions, contiguous) |
| `apply_filter_dataset` | Apply or remove compression filters (gzip, szip, shuffle, etc.) |
| `visualize` | Generate plots from datasets (line, heatmap, histogram, contour, etc.) |
| `read_semantic_metadata` | Read semantic metadata (SMD) from an HDF5 object |
| `write_semantic_metadata` | Write or update SMD on a single object |
| `collect_objects_for_smd` | Scan a file for objects missing SMD |
| `write_smd_batch` | Write SMD to multiple objects in a single transaction |
| `vectorize_semantic_metadata` | Embed all SMD into vector representations for search |
| `query_semantic_metadata` | Natural language semantic search over vectorized SMD |

### Skills

Skills are curated knowledge documents that teach the agent *how* and *when* to apply HDF5 best practices. They are loaded on-demand when a user's request matches the skill's domain, giving the agent expert-level guidance without bloating its context on every interaction. Skills live in `.claude/skills/`.

| Skill | Domain |
|-------|--------|
| `hdf5-chunking` | Chunk layout strategies and optimization |
| `hdf5-filters` | Compression and filter selection |
| `hdf5-io` | General I/O performance tuning |
| `hdf5-cloud-optimized` | Cloud/S3 access, paged aggregation, ros3 VFD |
| `hdf5-core-vfd` | In-memory file driver |
| `hdf5-parallel` | MPI-IO and parallel HDF5 |
| `hdf5-swmr` | Single Writer Multiple Reader access |
| `hdf5-vds` | Virtual datasets across multiple files |
| `hdf5-vol-usage` | Using VOL connectors (DAOS, Async, Cache, REST) |
| `hdf5-vol-dev` | Developing custom VOL connectors |
| `hdf5-visualization` | Plot type selection and matplotlib guidance |
| `hdf5-scientific-publishing` | DOIs, Zenodo/Dataverse, FAIR data practices |
| `hdf5-omni-selective` | OMNI file creation for selective data download |
| `hdf5-optimization` | General HDF5 optimization scripts |

### How They Work Together

Tools and skills are designed to complement each other but neither requires the other:

- **Skills alone** — An agent can use skill knowledge to advise on HDF5 best practices (e.g., recommending a chunk layout) without modifying any files.
- **Tools alone** — An agent can call tools to inspect, optimize, or annotate files using the tool's built-in logic, without loading any skill context.
- **Skills + Tools** — The most powerful mode. A skill provides the agent with expert knowledge (e.g., chunking strategies for cloud access patterns), and the agent then uses tools to apply that knowledge to specific files (e.g., rechunking a dataset with the recommended layout).

## Semantic Metadata (SMD)

Semantic metadata attributes (`ahdf5-smd-*`) attach human-readable, structured descriptions to HDF5 objects — describing what data represents, its provenance, units, and scientific significance. SMD bridges the gap between raw array data and human understanding.

See `docs/semantic-metadata.md` for the full specification.

## Vectorized Semantic Metadata (VSMD)

VSMD converts text-based SMD into vector embeddings stored directly in the HDF5 file, enabling natural language search over datasets. An agent (or user) can query "temperature measurements in Celsius" and retrieve the most semantically relevant objects — without needing to know paths or attribute names.

See `docs/vectorized-semantic-metadata.md` for the design document.

## Testing

```bash
python -m pytest tests/
```

### Agent Tool Selection Evaluation

The `tests/agent_tool_selection/` suite evaluates whether Claude models correctly identify the right HDF5 tool from natural language prompts. Run from a normal terminal (not inside Claude Code):

```bash
pytest -m agent tests/agent_tool_selection/ -v --model haiku
```

| Date | Model | Parameters | Score |
|------|-------|------------|-------|
| 2026-03-16 | Claude Opus 4.6 | Not disclosed | 7/7 (100%) |
| 2026-03-16 | Claude Sonnet 4.6 | Not disclosed | 7/7 (100%) |
| 2026-03-16 | Claude Haiku 4.5 | Not disclosed | 7/7 (100%) |
| 2026-03-16 | Claude 3 Haiku | ~20B (est.) | 7/7 (100%) |

See `tests/agent_tool_selection/RESULTS.md` for full methodology and detailed results across prompt modes.
