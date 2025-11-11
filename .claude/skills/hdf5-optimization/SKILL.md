---
name: hdf5-optimization
description: This skill should be used when the user asks to "optimize HDF5 file", "compress dataset", "reduce file size", "improve HDF5 performance", "rechunk dataset", "apply compression", "make HDF5 faster", mentions "slow HDF5 reads/writes", or requests optimization of storage space or access speed for HDF5 files. Provides guidance on compression, chunking, storage layout, and I/O optimization strategies.
version: 0.1.0
---

# HDF5 File Optimization

## Purpose

This skill provides guidance for optimizing HDF5 files for storage efficiency and I/O performance. Optimization involves choosing appropriate compression filters, chunk layouts, storage strategies, and runtime I/O settings based on data characteristics and access patterns.

## Core Concepts

### Optimization Goals

HDF5 optimization typically targets one or more objectives:

1. **Reduce file size**: Apply compression filters
2. **Improve read performance**: Align chunks with access patterns
3. **Improve write performance**: Tune chunk cache and buffer sizes
4. **Enable partial access**: Use chunked storage layout

### Key Optimization Levers

| Lever | Affects | When to Modify |
|-------|---------|----------------|
| Compression | File size, read/write speed | When storage is constrained |
| Chunk shape | Access speed | When access pattern is known |
| Chunk size | I/O efficiency, memory | When dealing with large datasets |
| Storage layout | Sequential vs. random access | At dataset creation |
| Chunk cache | Read performance | At file open time |

## Optimization Workflow

Follow this systematic approach:

1. **Analyze current state**: Check existing compression, chunking, and file size. Use h5dump if available.
2. **Identify bottleneck**: Storage space or access speed?
3. **Choose strategy**: Select appropriate optimization technique
4. **Apply changes**: Use h5repack or modify creation code
5. **Validate**: Verify improvements meet goals

## Quick Decision Guide

### Should I Use Compression?

```
Is storage space a concern?
├─ Yes → Apply GZIP level 1-3 with shuffle filter
└─ No → Skip compression, use contiguous storage for sequential access
```

**Recommended compression**: GZIP level 3 + shuffle filter. Increase compression level if storage is more important, decrease is speed is more important.

### Should I Use Chunking?

```
How is data accessed?
├─ Entire dataset sequentially → Consider contiguous storage
├─ Larger dataset -> Consider chunking
├─ Subsets/slices → Use chunking
├─ Need compression → Must use chunking
└─ Dataset extensible → Must use chunking
```

### What Chunk Shape?

```
Access pattern:
├─ Row-wise access → Row-oriented chunks
├─ Column-wise access → Column-oriented chunks
├─ Block access → Cubic/square chunks
└─ Unknown → Start with balanced chunks
```

### What Chunk Size?

**Target**: 10 KB to 1 MB per chunk (at default chunk cache size)

```
Current chunk size:
├─ < 10 KB → Too small, increase (excessive overhead)
├─ 10 KB - 1 MB → Good range
└─ > 1 MB → Too large, decrease (wastes bandwidth)
```

## Common Optimization Patterns

### Pattern 1: Reduce File Size

For large uncompressed files:

```bash
# Apply GZIP level 3 + shuffle to specific dataset
h5repack -f /dataset:SHUF -f /dataset:GZIP=3 input.h5 output.h5

# Apply to all datasets in file
h5repack -f SHUF -f GZIP=3 input.h5 output.h5
```

**Expected results**: 2-10x size reduction for numeric data

### Pattern 2: Optimize for Column Access

For datasets accessed by columns:

```bash
# Current: (100, 1000, 1000) with chunks (10, 10, 10)
# Problem: Column access reads many small chunks

# Solution: Make chunks column-oriented
h5repack -l /dataset:CHUNK=100x10x10 input.h5 output.h5
```

### Pattern 3: Convert to Contiguous for Sequential Access

For small datasets read entirely:

```bash
# Remove chunking, disable compression
h5repack -l /dataset:CONTI input.h5 output.h5
```

**Use when**: Dataset < 100 MB, always read completely

## Using h5repack

h5repack creates optimized copies of HDF5 files.

### Basic Syntax

```bash
h5repack -i input.h5 -o output.h5 [options]
```

### Common Options

| Option | Purpose | Example |
|--------|---------|---------|
| `-f` | Apply filter | `-f /dset:GZIP=3` |
| `-l` | Set layout/chunks | `-l /dset:CHUNK=100x100` |
| `-e` | Extend to all datasets | `-f GZIP=3` (no path) |

Use `h5repack --help` for further information on syntax and other specific options.

## Runtime I/O Optimization

Beyond file structure, optimize I/O at access time.

### Chunk Cache Tuning

Configure cache when opening file:

```python
import h5py

# Increase chunk cache for better read performance
f = h5py.File('data.h5', 'r', rdcc_nbytes=10*1024**2,  # 10 MB cache
              rdcc_w0=0.75,  # Preemption policy
              rdcc_nslots=10009)  # Number of slots (prime number)

dset = f['/dataset']
# Access data...
f.close()
```

**Parameters:**
- `rdcc_nbytes`: Cache size in bytes (default: 1 MB)
- `rdcc_w0`: Preemption policy (0.0-1.0, default: 0.75)
- `rdcc_nslots`: Hash table slots (prime number, default: 521)

**When to increase cache:**
- Reading same chunks multiple times
- Random access patterns
- Large chunk sizes

### Buffer Sizes

For h5repack operations:

```bash
# Set buffer size for h5repack (default: 1 MB)
export H5TOOLS_BUFSIZE=10MB
h5repack -f GZIP=3 input.h5 output.h5
```

## Optimization Best Practices

1. **Always analyze first**: Measure current performance and size
2. **Test on representative data**: Use real datasets when possible, not synthetic
3. **Benchmark changes**: Compare before/after objectively
4. **Match access patterns**: Align chunks with how data is read
5. **Start conservative**: Use GZIP 1-3, not higher levels
6. **Combine shuffle + GZIP**: Nearly always beneficial for numeric data
7. **Don't over-chunk**: Avoid creating millions of tiny chunks
8. **Document choices**: Record why specific settings were chosen

## Common Pitfalls

- **Over-compression**: GZIP 9 rarely worth the CPU cost
- **Misaligned chunks**: Row chunks for column access (poor performance)
- **Too-small chunks**: Creates metadata bloat, slow I/O
- **Too-large chunks**: Wastes bandwidth for partial access
- **Missing shuffle**: Forgetting shuffle filter with compression
- **Wrong layout**: Chunked storage for pure sequential access

## Additional Resources

### Reference Files

Detailed guidance on specific topics:

- **`references/compression-and-filters.md`** - Comprehensive filter guide, h5repack usage, performance trade-offs
- **`references/chunking-strategies.md`** - Chunk size calculations, access pattern analysis, layout decisions
- **`references/io-optimization.md`** - Runtime I/O tuning, cache configuration, parallel I/O strategies

### Examples

Working optimization workflows in `examples/`:

- **`examples/analyze-and-optimize.py`** - Script to analyze file and recommend optimizations
- **`examples/optimization-workflow.md`** - Complete step-by-step optimization example

## When to Optimize

**Optimize early when:**
- Creating large datasets (>1 GB)
- Access patterns are well-known
- Storage constraints are strict

**Optimize later when:**
- Access patterns are exploratory
- Dataset size is moderate (<100 MB)
- Rapid iteration is needed

**Don't optimize when:**
- Files are temporary
- Datasets are very small (<10 MB)
- Time to optimize exceeds time saved
