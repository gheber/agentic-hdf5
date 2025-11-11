#!/usr/bin/env python3
"""
Analyze HDF5 file and recommend optimizations.

This script examines datasets in an HDF5 file and provides recommendations
for compression, chunking, and storage layout optimization.

Usage:
    python analyze-and-optimize.py <file.h5> [dataset_path]
"""

import argparse
import h5py
import numpy as np
import os
import sys


def analyze_dataset(dset):
    """Analyze a single dataset and return optimization recommendations."""

    info = {
        'name': dset.name,
        'shape': dset.shape,
        'dtype': dset.dtype,
        'size_mb': dset.size * dset.dtype.itemsize / (1024**2),
        'chunks': dset.chunks,
        'compression': dset.compression,
        'compression_opts': dset.compression_opts,
        'shuffle': dset.shuffle,
        'fletcher32': dset.fletcher32,
        'scaleoffset': dset.scaleoffset,
    }

    # Calculate chunk size if chunked
    if dset.chunks:
        chunk_bytes = np.prod(dset.chunks) * dset.dtype.itemsize
        info['chunk_size_kb'] = chunk_bytes / 1024
    else:
        info['chunk_size_kb'] = None
        info['layout'] = 'contiguous'

    recommendations = []

    # Recommendation 1: Storage layout
    if dset.chunks is None:
        if info['size_mb'] > 100:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Layout',
                'issue': 'Large contiguous dataset',
                'recommendation': 'Convert to chunked storage to enable compression and partial access',
                'command': f'h5repack -l {dset.name}:CHUNK=...'
            })
        else:
            recommendations.append({
                'priority': 'INFO',
                'category': 'Layout',
                'issue': 'Small contiguous dataset',
                'recommendation': 'Contiguous layout is fine for small datasets accessed entirely'
            })

    # Recommendation 2: Compression
    if dset.chunks and not dset.compression:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Compression',
            'issue': 'No compression applied',
            'recommendation': 'Apply GZIP=3 with shuffle filter for ~3-5x size reduction',
            'command': f'h5repack -f {dset.name}:SHUF -f {dset.name}:GZIP=3 input.h5 output.h5'
        })
    elif dset.compression == 'gzip':
        level = dset.compression_opts or 6
        if level > 4 and not dset.shuffle:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Compression',
                'issue': f'GZIP level {level} without shuffle filter',
                'recommendation': 'Add shuffle filter to improve compression ratio',
                'command': f'h5repack -f {dset.name}:SHUF -f {dset.name}:GZIP={level} input.h5 output.h5'
            })
        elif level > 4:
            recommendations.append({
                'priority': 'LOW',
                'category': 'Compression',
                'issue': f'High GZIP level ({level})',
                'recommendation': 'Consider GZIP=3 for better write performance with minimal compression loss',
                'command': f'h5repack -f {dset.name}:SHUF -f {dset.name}:GZIP=3 input.h5 output.h5'
            })

    # Recommendation 3: Chunk size
    if dset.chunks:
        chunk_kb = info['chunk_size_kb']
        if chunk_kb < 10:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Chunking',
                'issue': f'Very small chunks ({chunk_kb:.1f} KB)',
                'recommendation': 'Increase chunk size to 10-100 KB range to reduce overhead',
                'command': f'h5repack -l {dset.name}:CHUNK=... input.h5 output.h5'
            })
        elif chunk_kb > 1024:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Chunking',
                'issue': f'Very large chunks ({chunk_kb:.1f} KB)',
                'recommendation': 'Reduce chunk size to 100-1000 KB range for better partial access',
                'command': f'h5repack -l {dset.name}:CHUNK=... input.h5 output.h5'
            })
        else:
            recommendations.append({
                'priority': 'INFO',
                'category': 'Chunking',
                'issue': f'Chunk size is reasonable ({chunk_kb:.1f} KB)',
                'recommendation': 'Current chunk size is in optimal range'
            })

    # Recommendation 4: Data integrity
    if not dset.fletcher32 and info['size_mb'] > 1000:
        recommendations.append({
            'priority': 'LOW',
            'category': 'Integrity',
            'issue': 'No checksums for large dataset',
            'recommendation': 'Consider adding Fletcher32 checksums for data integrity verification',
            'command': f'h5repack -f {dset.name}:FLET input.h5 output.h5'
        })

    return info, recommendations


def analyze_file(filepath, dataset_path=None):
    """Analyze entire file or specific dataset."""

    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return

    file_size_mb = os.path.getsize(filepath) / (1024**2)

    print(f"\n{'='*70}")
    print(f"HDF5 File Analysis: {filepath}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"{'='*70}\n")

    with h5py.File(filepath, 'r') as f:
        if dataset_path:
            # Analyze specific dataset
            if dataset_path not in f:
                print(f"Error: Dataset '{dataset_path}' not found in file")
                return

            dset = f[dataset_path]
            if not isinstance(dset, h5py.Dataset):
                print(f"Error: '{dataset_path}' is not a dataset")
                return

            analyze_and_print_dataset(dset)

        else:
            # Analyze all datasets
            datasets = []

            def collect_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets.append(obj)

            f.visititems(collect_datasets)

            if not datasets:
                print("No datasets found in file")
                return

            print(f"Found {len(datasets)} dataset(s)\n")

            for dset in datasets:
                analyze_and_print_dataset(dset)
                print()


def analyze_and_print_dataset(dset):
    """Analyze and print results for a dataset."""

    info, recommendations = analyze_dataset(dset)

    # Print dataset info
    print(f"Dataset: {info['name']}")
    print(f"  Shape: {info['shape']}")
    print(f"  Type: {info['dtype']}")
    print(f"  Size: {info['size_mb']:.2f} MB")

    if info['chunks']:
        print(f"  Chunks: {info['chunks']} ({info['chunk_size_kb']:.1f} KB/chunk)")
    else:
        print(f"  Layout: Contiguous (not chunked)")

    if info['compression']:
        print(f"  Compression: {info['compression']}", end='')
        if info['compression_opts']:
            print(f" (level {info['compression_opts']})", end='')
        print()

    if info['shuffle']:
        print(f"  Shuffle: Enabled")

    if info['fletcher32']:
        print(f"  Fletcher32: Enabled")

    # Print recommendations
    if recommendations:
        print(f"\n  Recommendations:")
        for rec in recommendations:
            priority_marker = {
                'HIGH': '❗',
                'MEDIUM': '⚠️ ',
                'LOW': 'ℹ️ ',
                'INFO': '  '
            }.get(rec['priority'], '  ')

            print(f"    {priority_marker} [{rec['priority']}] {rec['category']}: {rec['issue']}")
            print(f"       → {rec['recommendation']}")
            if 'command' in rec:
                print(f"       Command: {rec['command']}")
    else:
        print(f"\n  ✓ No optimization recommendations")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze HDF5 file and recommend optimizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze entire file
  python analyze-and-optimize.py data.h5

  # Analyze specific dataset
  python analyze-and-optimize.py data.h5 /measurements/temperature

  # Analyze and save report
  python analyze-and-optimize.py data.h5 > analysis_report.txt
        """
    )

    parser.add_argument('filepath', help='Path to HDF5 file')
    parser.add_argument('dataset', nargs='?', help='Optional: specific dataset path')

    args = parser.parse_args()

    try:
        analyze_file(args.filepath, args.dataset)
    except Exception as e:
        print(f"Error analyzing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
