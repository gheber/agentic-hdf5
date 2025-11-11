# Agentic HDF5

![Test Agentic HDF5 Tools](https://github.com/mattjala/agentic-hdf5/actions/workflows/test-ahdf5-tools.yml/badge.svg)
![GitHub tag](https://img.shields.io/github/v/tag/mattjala/agentic-hdf5)

A collection of tools and skills that enable AI agents to work at a high level with HDF5 data and files.

Included Skills:
- HDF-optimization Skill: Skill teaching agents best-practice for HDF5 chunking, chunk cache usage, filter/compression usage, and read/write practices

Included Tools:
- Tools for creation and parsing of Semantic Metadata (SMD) attributes to track data provenance, significance, and other high-level scientific attributes. See `docs/semantic-metadata.md` for details
- Tools for creation and querying of Vectorized Semantic Metadata (VSMD), enabling natural-language/keyword based queries over data within SMD attributes. See `docs/vectorized_smd_design.md` for details
