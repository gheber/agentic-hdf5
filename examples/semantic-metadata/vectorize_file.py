#!/usr/bin/env python3
"""
Vectorize semantic metadata in an HDF5 file.

This script reads all SMD attributes from an HDF5 file, generates vector embeddings,
and stores them in the /ahdf5-vsmd group for semantic search.
"""

import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Vectorize semantic metadata in an HDF5 file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s example_with_smd.h5
  %(prog)s /path/to/data.h5 --model sentence-transformers/all-mpnet-base-v2
        """
    )
    parser.add_argument(
        'filepath',
        help='Path to the HDF5 file to vectorize'
    )
    parser.add_argument(
        '--model',
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Embedding model to use (default: sentence-transformers/all-MiniLM-L6-v2)'
    )

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.filepath).exists():
        print(f"Error: File not found: {args.filepath}", file=sys.stderr)
        sys.exit(1)

    # Import vectorization function only when needed (avoids loading model for --help)
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "servers" / "h5py"))
    from vectorize_semantic_metadata import vectorize_semantic_metadata

    print(f"Vectorizing semantic metadata in: {args.filepath}")
    print(f"Using embedding model: {args.model}")
    print("-" * 60)

    # Vectorize the file
    result = vectorize_semantic_metadata(
        args.filepath,
        embedder_model=args.model,
        rebuild=True
    )

    # Display results
    print()
    if result['status'] == 'success':
        print("✓ Vectorization successful!")
        print()
        print(f"  Objects vectorized: {result['objects_vectorized']}")
        print(f"  Total chunks: {result['total_chunks']}")
        print(f"  Embedding dimension: {result['embed_dim']}")
        print(f"  VSMD stored at: {result['vsmd_path']}")
        print()
        print(f"Message: {result['message']}")
        print()
        print("You can now query this file using query_file.py")
        sys.exit(0)
    else:
        print("✗ Vectorization failed!", file=sys.stderr)
        print(f"Error: {result['message']}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
