== Vectorized Semantic Metadata Specification

Vectorized semantic metadata (VSMD) extends the semantic metadata system by converting text-based SMD into searchable vector embeddings. This enables natural language queries over HDF5 file contents based on semantic meaning rather than exact text matching.

=== Purpose

Vectorized semantic metadata is introduced to enable several key capabilities:

- Enable natural language search across HDF5 files (e.g., "temperature in Celsius", "pressure measurements")
- Find semantically related objects even when terminology differs
- Support large-scale metadata exploration without reading all SMD attributes linearly
- Provide fast similarity-based ranking of objects by relevance to a query

=== Architecture

VSMD is stored within the HDF5 file itself in a dedicated group structure at `/ahdf5-vsmd`. This co-location ensures that the vector index travels with the data and remains synchronized with the underlying semantic metadata.

==== Storage Structure

The `/ahdf5-vsmd` group contains three main components:

**/ahdf5-vsmd/meta** - Metadata about the vectorization process:
- `created_at`: ISO timestamp of when VSMD was generated
- `embedder`: Name of the sentence transformer model used (e.g., "sentence-transformers/all-MiniLM-L6-v2")
- `embed_dim`: Dimension of embedding vectors (e.g., 384)
- `version`: VSMD format version (currently "1.0")
- `smd_hash`: SHA-256 hash of all SMD content for staleness detection
- `note`: Implementation notes (e.g., "Embeddings are L2-normalized for cosine similarity via dot product")

**/ahdf5-vsmd/chunks** - Parallel datasets containing the vectorized content:
- `text`: Variable-length UTF-8 strings containing the original SMD text (gzip-4 compressed)
- `object_path`: Variable-length UTF-8 strings containing HDF5 object paths (gzip-4 compressed)
- `embedding`: Float32 array of L2-normalized embedding vectors (gzip-4 compressed, shape: N × embed_dim)

**/ahdf5-vsmd/index** - Structured dataset mapping objects to chunks:
- `object_path`: Path to the HDF5 object
- `object_type`: Type of object ("dataset", "group", "file_root", "datatype")
- `chunk_start`: Starting index in chunks arrays (inclusive)
- `chunk_end`: Ending index in chunks arrays (exclusive)
- `smd_length`: Character length of the SMD text

==== Chunking Strategy

Version 1.0 uses a simple chunking strategy: one chunk per object. Each object's complete SMD text is embedded as a single vector. The chunk_start and chunk_end indices in the index are therefore always consecutive (chunk_end = chunk_start + 1).

Future versions may support splitting large SMD texts into multiple chunks to improve search granularity and manage very long metadata descriptions.

=== Vectorization Process

The vectorization process (implemented in `vectorize_semantic_metadata()`) follows these steps:

1. **Collection**: Scan the HDF5 file for all `ahdf5-smd-*` attributes, collecting SMD text and object metadata
2. **Embedding**: Process SMD texts in batches (batch size: 1000) through a sentence transformer model
3. **Normalization**: L2-normalize all embedding vectors to enable cosine similarity via dot product
4. **Hashing**: Compute SHA-256 hash of concatenated SMD texts for staleness detection
5. **Storage**: Write the `/ahdf5-vsmd` structure to the HDF5 file with all embeddings and metadata

The vectorization process can be run in full rebuild mode (deleting existing VSMD) or incremental mode (updating only changed objects, not yet implemented in v1.0).

=== Query Process

Natural language queries (implemented in `query_semantic_metadata()`) operate as follows:

1. **Validation**: Verify that `/ahdf5-vsmd` exists in the file and check model compatibility
2. **Query Embedding**: Embed the query text using the same sentence transformer model, with L2-normalization
3. **Filtering**: Optionally filter candidate objects by path prefix (e.g., only search within "/data/experiments")
4. **Similarity Computation**: Compute dot product between query embedding and all candidate embeddings (equivalent to cosine similarity due to normalization)
5. **Ranking**: Sort results by similarity score in descending order
6. **Thresholding**: Apply minimum similarity score filter if specified
7. **Selection**: Return top-k results with scores, object paths, types, and SMD text

For large files, embeddings are loaded in blocks (block size: 128,000 vectors) to manage memory usage.

=== Model Compatibility

The embedding model used for vectorization is stored in the `/ahdf5-vsmd/meta` group. Queries must use the same model to ensure vector compatibility. The query function validates model compatibility and raises an error if a different model is requested.

Changing the embedding model requires a full rebuild of the VSMD structure, as embeddings from different models are not comparable.

=== Staleness Detection

The `smd_hash` attribute in `/ahdf5-vsmd/meta` contains a SHA-256 hash of all SMD text content at the time of vectorization. This enables detection of staleness:

- If SMD attributes are added, removed, or modified after vectorization, the hash will no longer match
- Agents can recompute the hash and compare to detect when VSMD needs updating
- Stale VSMD can still be queried but may return incomplete or outdated results

Future versions may implement automatic staleness detection and incremental updates.
