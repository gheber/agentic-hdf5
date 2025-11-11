# I/O Optimization Reference

This reference covers runtime strategies for optimizing HDF5 I/O performance: chunk cache configuration, buffer tuning, access patterns, and parallel I/O considerations.

## Write Performance

### Buffered Writes

Write in large batches:

```python
import h5py
import numpy as np

# Good: Single large write
with h5py.File('data.h5', 'w') as f:
    data = np.random.randn(10000, 10000)
    f.create_dataset('data', data=data, chunks=(100, 100))

# Bad: Many small writes
with h5py.File('data.h5', 'w') as f:
    dset = f.create_dataset('data', shape=(10000, 10000), chunks=(100, 100))
    for i in range(10000):
        dset[i, :] = np.random.randn(10000)  # Slow!
```

### Chunk Cache for Writes

Increase cache for write-heavy workloads:

```python
# Writing many small updates
f = h5py.File('data.h5', 'a',  # Append mode
              rdcc_nbytes=20*1024**2)  # Large cache for write buffering

dset = f['/dataset']
for i in range(1000):
    dset[i*10:(i+1)*10, :] = compute_chunk(i)  # Updates buffered

f.close()  # Flush all writes
```

## File-Level Optimization

### Alignment and Threshold

Control how HDF5 aligns data blocks:

```python
import h5py

# Set alignment for better filesystem performance
fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
fapl.set_alignment(threshold=1024**2, alignment=4096)
# Align objects > 1 MB to 4 KB boundaries

# Use with file creation (low-level API)
# Usually automatic optimization, rarely need manual tuning
```

## Buffer Sizes

### h5repack Buffer

Control memory usage during h5repack:

```bash
# Default: 1 MB buffer
h5repack -f GZIP=3 input.h5 output.h5

# Larger buffer: Faster for large files
export H5TOOLS_BUFSIZE=10MB
h5repack -f GZIP=3 input.h5 output.h5

# Very large buffer for huge files
export H5TOOLS_BUFSIZE=100MB
h5repack -f GZIP=3 huge_file.h5 optimized.h5
```

**Impact**:
- Larger buffers → faster repacking
- More memory usage
- Diminishing returns above ~50 MB

### Application Buffers

Use buffering in application code:

```python
import h5py
import numpy as np

# Buffer writes in memory, then flush
write_buffer = []
buffer_size = 1000

with h5py.File('data.h5', 'a') as f:
    dset = f['/dataset']

    for i in range(10000):
        write_buffer.append(compute_row(i))

        if len(write_buffer) >= buffer_size:
            # Flush buffer
            start = i - len(write_buffer) + 1
            end = i + 1
            dset[start:end, :] = np.array(write_buffer)
            write_buffer = []

    # Flush remaining
    if write_buffer:
        dset[-len(write_buffer):, :] = np.array(write_buffer)
```