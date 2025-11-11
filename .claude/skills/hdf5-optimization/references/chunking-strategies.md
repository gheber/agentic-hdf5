# Chunking Strategies Reference

This reference provides comprehensive guidance on HDF5 chunking: choosing chunk sizes, shapes, and storage layouts for optimal performance.

## Overview

Chunking is the single most important factor affecting HDF5 performance. Chunks are the atomic unit of I/O—reading a single value requires loading its entire chunk. Proper chunking aligns storage with access patterns and enables compression.

## Storage Layouts

HDF5 supports two main storage layouts:

### Contiguous Storage

**Description**: Data stored in single continuous block in file

**Characteristics**:
- Optimal for sequential access of entire dataset
- No compression or extensibility
- Minimal overhead

**Use when**:
- Dataset is small (< 100 MB)
- Entire dataset always accessed together
- Dataset size is fixed

**Avoid when**:
- Want compression
- Dataset may grow
- Random, small accesses

### Chunked Storage

**Description**: Data divided into fixed-size chunks stored independently

**Characteristics**:
- Enables partial access, compression, extension
- Overhead from chunk index/metadata
- Performance depends on chunk configuration
- Required for filters

**Use when**:
- Need partial dataset access
- Want compression
- Dataset may be extended
- Data access is non-sequential

**Avoid when**:
- Always read entire dataset sequentially
- Dataset very small (overhead dominates)

## Chunk Size Fundamentals

### Optimal Size Range

**Target**: 10 KB to 1 MB per chunk

**Why this range?**
- **< 10 KB**: Excessive I/O overhead, metadata bloat
- **10 KB - 100 KB**: Good for random access, many small reads
- **100 KB - 1 MB**: Good for sequential access, larger reads
- **> 1 MB**: Wastes bandwidth on partial access, memory pressure

### Chunk Cache Size

HDF5 2.0.0 uses an 8MiB chunk cache per open dataset by default. On newer systems with more resources, it often makes sense to increase the size of this cache.
In h5py, the chunk cache size is controlled by the `rdcc_nbytes` keyword argument to `h5py.File()`.

HDF5 2.0.0 has 521 slots in the chunk cache by default.
The number of slots should be set to a prime number higher than the number of chunks you expect to access.
The number of slots is controleld by the `rdcc_nslots` keyword argument to `h5py.File()`.

### Preemption Policy (`rdcc_w0`)

**Purpose**: Controls chunk eviction strategy
**Range**: 0.0 to 1.0
**Default**: 0.75

**Interpretation**:
- **w0 = 0.0**: Pure LRU (least recently used)
- **w0 = 1.0**: Pure LFU (least frequently used)
- **w0 = 0.75**: Balanced (default, usually optimal)

**When to adjust**:
- **w0 → 0.0**: Access pattern is strictly sequential (favor recency)
- **w0 → 1.0**: Some chunks accessed repeatedly (favor frequency)
- **Usually leave at 0.75**: Default works well for most patterns

## Chunk Shape Optimization

Chunk shape determines which access patterns are efficient.

### Access Pattern Alignment

**Critical principle**: Chunks should align with how data is accessed. If no specific access pattern is known, default to roughly balancing chunks along each dimension, with a weighting towards total dataset size along each dimension.

**Example: 2D array (1000, 1000)**

**Row-wise access** (e.g., `data[i, :]`):
```python
# Good: Row-oriented chunks
chunks = (1, 1000)  # Each row is one chunk
# Reading row i loads 1 chunk

# Bad: Column-oriented chunks
chunks = (1000, 1)  # Each column is one chunk
# Reading row i loads 1000 chunks!
```

**Block access** (e.g., `data[0:100, 0:100]`):
```python
# Good: Square chunks
chunks = (100, 100)  # Matches block size

# Bad: Non-aligned chunks
chunks = (1, 1000)  # Loads many chunks for small block
```

The same principles apply in higher dimensions.

**Example: 3D dataset (100, 500, 500)**

**Volume access** (full 3D blocks):
```python
chunks = (10, 50, 50)  # Cubic chunks, balanced
```

**Slice access** (2D slices along first axis):
```python
chunks = (1, 100, 100)  # Thin along sliced dimension - good if many entries along axes 2 and 3 are desired for r/w relative to first dimension.
```

## Using h5repack for Rechunking

### Basic Rechunking

Change chunk dimensions:

```bash
# Explicit dimensions
h5repack -l /dataset:CHUNK=100x200 input.h5 output.h5

# For 3D data
h5repack -l /dataset:CHUNK=10x100x100 input.h5 output.h5
```

### Convert to Contiguous

Remove chunking:

```bash
h5repack -l /dataset:CONTI input.h5 output.h5
```

### Rechunk Multiple Datasets

```bash
# Different chunks for different datasets
h5repack \
  -l /data/raw:CHUNK=100x1x1 \
  -l /data/grid:CHUNK=1x100x100 \
  input.h5 output.h5
```

### Rechunk + Compress

Combine rechunking with compression:

```bash
h5repack \
  -l /dataset:CHUNK=100x100 \
  -f /dataset:SHUF \
  -f /dataset:GZIP=3 \
  input.h5 output.h5
```

## Chunk Cache Considerations


**When to increase cache**:
- Random access patterns
- Repeatedly accessing same chunks
- Reading same data multiple times

## Workflow for Choosing Chunks

1. **Analyze access pattern**: How will data be read?
2. **Determine target size**: 10 KB - 1 MB range
3. **Calculate dimensions**: Align with access, meet size target
4. **Create test file**: Write with proposed chunks
5. **Benchmark**: Test realistic access patterns
6. **Iterate**: Adjust based on measurements
7. **Document choice**: Record rationale for future reference

## Best Practices

1. **Default to 100 KB chunks** for unknown access patterns
2. **Align with access patterns** when known
3. **Use balanced dimensions** for multi-dimensional data
4. **Avoid extremes**: Not too small (< 10 KB) or too large (> 1 MB)
5. **Test with real access patterns**, not synthetic
6. **Consider compression**: Chunks needed for filters
7. **Use h5repack** to experiment with existing files
8. **Document decisions**: Why these chunks? What access pattern?
9. **Monitor performance**: Measure, don't guess
10. **Rechunk if access changes**: Not permanent decisions

## Additional Resources

- HDF Group: [Chunking in HDF5](https://support.hdfgroup.org/documentation/hdf5-latest/hdf5_chunking.html)
- Davis Lab: [Chunking in HDF5 Manual](https://davis.lbl.gov/Manuals/HDF5-1.8.7/Advanced/Chunking/index.html)
- BASTet: [HDF5 I/O Performance Documentation](https://biorack.github.io/BASTet/HDF5_format_performance.html)
