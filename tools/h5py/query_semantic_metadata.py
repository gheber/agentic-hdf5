"""
Query vectorized semantic metadata using natural language.

This module provides semantic search capabilities over HDF5 files with
vectorized semantic metadata (VSMD).
"""

import h5py
import numpy as np

try:
    from tools.h5py.registry import hdf5_tool
except ImportError:
    def hdf5_tool(**_kw):
        return lambda f: f


BLOCK_SIZE = 128000  # Load embeddings in blocks for large files


@hdf5_tool(
    category="semantic-search",
    keywords=["search", "query", "find", "semantic", "natural language", "similarity", "discover", "locate"],
    use_cases=[
        "Finding datasets by natural language query",
        "Discovering relevant data without knowing paths",
        "Semantic search for 'temperature data'",
        "Locating datasets by description",
    ],
)
def query_semantic_metadata(
    filepath: str,
    query_text: str,
    top_k: int = 5,
    object_filter: str | None = None,
    min_score: float = 0.0,
    embedder_model: str | None = None
) -> dict:
    """
    Perform natural language semantic search over vectorized semantic metadata.

    Takes a query string (e.g., "temperature in Celsius") and returns the top-k
    most semantically similar objects from the file based on their SMD. Requires
    that vectorize_semantic_metadata() has been run first. Supports filtering by
    path prefix and minimum similarity score thresholds.

    Args:
        filepath: Path to the HDF5 file
        query_text: Natural language query (e.g., "temperature in Celsius")
        top_k: Number of results to return
        object_filter: Optional path prefix to restrict search (e.g., "/data")
        min_score: Minimum similarity score (0.0-1.0)
        embedder_model: Override embedding model (defaults to model used in file)

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - query: Original query text
        - results: List of dicts with:
            - rank: 1-indexed rank
            - score: Cosine similarity score (0.0-1.0)
            - object_path: Path to the HDF5 object
            - object_type: "dataset", "group", or "file_root"
            - smd_text: The semantic metadata text
            - smd_preview: First 200 chars of SMD (for display)
    """
    # Step 1: Validate VSMD exists and check model compatibility
    try:
        with h5py.File(filepath, 'r') as f:
            if '/ahdf5-vsmd' not in f:
                return {
                    "status": "error",
                    "message": "No VSMD found in file. Run vectorize_semantic_metadata first.",
                    "query": query_text,
                    "results": []
                }

            # Load metadata
            stored_model = f['/ahdf5-vsmd/meta'].attrs['embedder']
            if isinstance(stored_model, bytes):
                stored_model = stored_model.decode('utf-8')
    except FileNotFoundError:
        return {
            "status": "error",
            "message": f"File not found: {filepath}",
            "query": query_text,
            "results": []
        }

    # Step 2: Validate model compatibility (raises on mismatch - validation error)
    model_to_use = embedder_model if embedder_model is not None else stored_model

    if embedder_model is not None and embedder_model != stored_model:
        raise ValueError(
            f"Model mismatch: file uses '{stored_model}' but query requested '{embedder_model}'"
        )

    try:
        # Step 3: Load embedding model and embed query
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_to_use)
        query_embedding = model.encode(
            [query_text],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True  # L2-normalize for cosine similarity
        )[0]  # Extract single vector

        # Step 4: Filter candidates and compute similarity
        with h5py.File(filepath, 'r') as f:
            # Load index to get object metadata
            index = f['/ahdf5-vsmd/index'][:]

            # Filter by object_filter if provided
            if object_filter is not None:
                # Filter index entries by path prefix
                mask = np.array([
                    _starts_with(obj_path, object_filter)
                    for obj_path in index['object_path']
                ])
                filtered_indices = np.where(mask)[0]

                if len(filtered_indices) == 0:
                    return {
                        "status": "success",
                        "query": query_text,
                        "results": []
                    }
            else:
                filtered_indices = np.arange(len(index))

            # Step 5: Compute similarity scores
            # For v1.0, chunk indices = object indices (1 chunk per object)
            embeddings = f['/ahdf5-vsmd/chunks/embedding']

            # Load relevant embeddings
            if len(filtered_indices) <= BLOCK_SIZE:
                # Small enough to load all at once
                relevant_embeddings = embeddings[filtered_indices]
            else:
                # Block-wise loading for large files
                relevant_embeddings = _load_embeddings_blockwise(
                    embeddings,
                    filtered_indices
                )

            # Compute dot product (cosine similarity since normalized)
            similarity_scores = np.dot(relevant_embeddings, query_embedding)

            # Step 6: Rank and filter
            # Find indices sorted by score (descending)
            sorted_indices = np.argsort(similarity_scores)[::-1]

            # Apply min_score filter
            valid_mask = similarity_scores[sorted_indices] >= min_score
            sorted_indices = sorted_indices[valid_mask]

            # Take top-K
            top_indices = sorted_indices[:top_k]

            # Map back to original file indices
            result_file_indices = filtered_indices[top_indices]

            # Step 7: Load corresponding data and format results
            texts = f['/ahdf5-vsmd/chunks/text'][:]
            object_paths = f['/ahdf5-vsmd/chunks/object_path'][:]

            results = []
            for rank, file_idx in enumerate(result_file_indices, start=1):
                # Get score (map top_indices back to similarity_scores)
                score_idx = top_indices[rank - 1]
                score = float(similarity_scores[score_idx])

                # Get metadata from index
                obj_metadata = index[file_idx]

                # Decode text fields
                object_path = _decode_if_bytes(object_paths[file_idx])
                object_type = _decode_if_bytes(obj_metadata['object_type'])
                smd_text = _decode_if_bytes(texts[file_idx])

                # Create preview (first 200 chars)
                smd_preview = smd_text[:200]
                if len(smd_text) > 200:
                    smd_preview += "..."

                results.append({
                    'rank': rank,
                    'score': score,
                    'object_path': object_path,
                    'object_type': object_type,
                    'smd_text': smd_text,
                    'smd_preview': smd_preview
                })

        return {
            "status": "success",
            "query": query_text,
            "results": results
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during query: {str(e)}",
            "query": query_text,
            "results": []
        }


def _starts_with(path, prefix):
    """
    Check if path starts with prefix.

    Handles both string and bytes types.
    """
    if isinstance(path, bytes):
        path = path.decode('utf-8')
    if isinstance(prefix, bytes):
        prefix = prefix.decode('utf-8')

    return path.startswith(prefix)


def _decode_if_bytes(value):
    """Decode bytes to string if needed."""
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value


def _load_embeddings_blockwise(dataset, indices):
    """
    Load embeddings in blocks to avoid memory issues with large datasets.

    Args:
        dataset: HDF5 dataset containing embeddings
        indices: Array of indices to load

    Returns:
        NumPy array of selected embeddings
    """
    embed_dim = dataset.shape[1]
    result = np.zeros((len(indices), embed_dim), dtype=np.float32)

    for i in range(0, len(indices), BLOCK_SIZE):
        block_end = min(i + BLOCK_SIZE, len(indices))
        block_indices = indices[i:block_end]
        result[i:block_end] = dataset[block_indices]

    return result
