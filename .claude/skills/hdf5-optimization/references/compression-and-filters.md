# Compression and Filters Reference

This reference provides comprehensive information about HDF5 compression filters, their characteristics, usage patterns, and trade-offs.

## Overview

HDF5 filters are processing pipelines applied to dataset chunks before storage. Filters can compress data, add checksums, or transform data layout. Multiple filters can be chained together in a specific order.

**Key principle**: Filters operate per-chunk on chunked datasets only. Contiguous datasets cannot use filters.

## Available Filters

### GZIP (Deflate) Compression

**Type**: Lossless compression
**Availability**: Universal (always available)
**h5repack syntax**: `GZIP=N` where N is 1-9

**Description**:
GZIP is the most widely supported compression filter in HDF5. It uses the DEFLATE algorithm (same as ZIP files) and provides a good balance of compression ratio and speed.

**Compression levels**: Go from 1 (fastest, but least amount of compression, 2-3x) to 9 (max compression, ~4-6x, slower access). Higher compression levels provide more compression, but with diminishing returns. Default to level 3.

**Use when**:
- Need portable, widely-supported compression
- Balanced read/write performance needed

**Avoid when**:
- Ultra-high speed writes are critical
- Data is already compressed (images, video)

**h5repack examples**:
```bash
# Apply GZIP level 1 (fast)
h5repack -f /dataset:GZIP=1 input.h5 output.h5

# Apply to all datasets
h5repack -f GZIP=3 input.h5 output.h5
```

### Shuffle Filter

**Type**: Byte reordering (preprocessing)
**Availability**: Universal
**h5repack syntax**: `SHUF`

**Description**:
Shuffle rearranges byte order within chunks to group similar byte positions together. For example, transforms:
```
[01234567][01234567][01234567]  # Original: interleaved bytes
→
[000][111][222][333][444][555][666][777]  # Shuffled: grouped by position
```

This dramatically improves compression ratios for numeric data because similar byte values cluster together.

**Impact**:
- Improves GZIP compression by 2-5x
- Minimal CPU overhead (<5%)
- No effect on non-numeric data
- **Must be combined with compression filter**

**Use when**:
- Compressing integer or floating-point arrays
- Using GZIP or other compression
- Data has low byte-level entropy

**h5repack example**:
```bash
# Shuffle + GZIP (shuffle applied first automatically)
h5repack -f SHUF -f GZIP=3 input.h5 output.h5
```

**Best practice**: Always pair shuffle with GZIP for numeric data.

### SZIP Compression

**Type**: Lossless compression
**Availability**: Optional (patent-restricted)
**h5repack syntax**: `SZIP=pixels_per_block,coding`

**Description**:
SZIP is NASA-developed compression optimized for scientific data, particularly satellite imagery. It's faster than GZIP but has patent restrictions.

**Parameters**:
- `pixels_per_block`: Must be even, 2-32 (typically 8 or 16)
- `coding`: `NN` (nearest neighbor) or `EC` (entropy coding)

**Compression characteristics**:
- Similar ratio to GZIP 1-3
- Faster compression/decompression
- Patent-restricted (requires license for some uses)

**Use when**:
- SZIP is available (check filter availability)
- Need faster compression than GZIP
- Data is scientific measurements

**Avoid when**:
- Portability is required (SZIP not always available)
- Patent licensing is concern

**h5repack examples**:
```bash
# SZIP with 8 pixels per block, nearest neighbor
h5repack -f /dataset:SZIP=8,NN input.h5 output.h5

# SZIP with entropy coding
h5repack -f /dataset:SZIP=16,EC input.h5 output.h5
```

### LZF Compression

**Type**: Lossless compression
**Availability**: h5py only
**h5repack syntax**: Not directly supported

**Description**:
LZF is a very fast compression algorithm with moderate compression ratios. Available through h5py but not standard h5repack.

**Characteristics**:
- Very fast (faster than GZIP 1)
- Moderate compression (~2-3x)
- Not widely supported outside h5py

**Use when**:
- Speed is paramount
- Using h5py for all access
- Moderate compression acceptable

### Fletcher32 Checksum

**Type**: Data integrity verification
**Availability**: Universal
**h5repack syntax**: `FLET`

**Description**:
Adds 32-bit checksum to each chunk for detecting corruption. Not compression. Increases size by adding ~4 bytes per chunk overhead, and is likely to have a minor negative impact on speed.

**Impact**:
- Slight performance penalty (~5-10%)
- Detects data corruption
- Adds small storage overhead

**Use when**:
- Data integrity is critical
- Storage on unreliable media
- Long-term archival

**h5repack examples**:
```bash
# Add checksums
h5repack -f /dataset:FLET input.h5 output.h5

# Combine with compression
h5repack -f /dataset:SHUF -f /dataset:GZIP=3 -f /dataset:FLET input.h5 output.h5
```

### N-Bit Filter

**Type**: Lossless compression (bit packing)
**Availability**: Universal
**h5repack syntax**: `NBIT`

**Description**:
Packs integer data to use only the minimum necessary bits. For example, stores 12-bit sensor data in 12 bits instead of 16.

**Compression ratio**:
- Depends on data precision
- Example: 12-bit data in 16-bit type → 1.33x compression
- Example: 10-bit data in 32-bit type → 3.2x compression

**Use when**:
- Data uses fewer bits than storage type
- Sensor data with known precision
- Need lossless compression

**Limitations**:
- Requires specifying bit precision
- Only for integer types
- Not widely used (complex configuration)

**h5repack examples**:
```bash
# Apply N-bit filter
h5repack -f /dataset:NBIT input.h5 output.h5
```

### Scale-Offset Filter

**Type**: Lossy compression (quantization)
**Availability**: Universal
**h5repack syntax**: `SOFF=scale_factor,scale_type`

**Description**:
Quantizes floating-point data to specified precision, discarding insignificant digits. Reduces storage by truncating precision.

**Parameters**:
- `scale_factor`: Number of decimal digits to preserve
- `scale_type`: `IN` (integer) or `DS` (decimal significant figures)

**Compression characteristics**:
- Lossy (data modified)
- Can achieve 2-4x compression
- Useful when full precision unnecessary

**Use when**:
- Full floating-point precision not needed
- Data has known significant figures
- Acceptable to lose precision

**Avoid when**:
- Exact values required
- Downstream analysis sensitive to small errors

**h5repack examples**:
```bash
# Keep 2 decimal digits
h5repack -f /dataset:SOFF=2,DS input.h5 output.h5

# Integer quantization with scale factor 10
h5repack -f /dataset:SOFF=10,IN input.h5 output.h5
```

## Filter Combinations

Filters are applied in order and form a pipeline. **Order matters**.

### Recommended Combinations

**Standard compression (numeric data)**:
```bash
h5repack -f SHUF -f GZIP=3 input.h5 output.h5
```
- Shuffle first (improves GZIP ratio)
- GZIP second (compresses shuffled data)
- Best general-purpose option

**Maximum compression**:
```bash
h5repack -f SHUF -f GZIP=9 input.h5 output.h5
```
- Highest compression ratio
- Slow writes, acceptable reads
- Use for archival data

**Fast compression**:
```bash
h5repack -f SHUF -f GZIP=1 input.h5 output.h5
```
- Fast writes and reads
- Moderate compression
- Use for working datasets

## Using h5repack for Filtering

### Basic h5repack Operations

**Syntax**:
```bash
h5repack -i input.h5 -o output.h5 [filter options]
```

Or simplified:
```bash
h5repack [filter options] input.h5 output.h5
```

h5repach --help can be used to acquire more information on the syntax and optional arguments.

## Checking Filter Availability

Use Python to check which filters are available to h5py:

```python
import h5py.h5z as h5z

filters = {
    'GZIP': h5z.filter_avail(h5z.FILTER_DEFLATE),
    'Shuffle': h5z.filter_avail(h5z.FILTER_SHUFFLE),
    'Fletcher32': h5z.filter_avail(h5z.FILTER_FLETCHER32),
    'SZIP': h5z.filter_avail(h5z.FILTER_SZIP),
    'N-bit': h5z.filter_avail(h5z.FILTER_NBIT),
    'Scale-offset': h5z.filter_avail(h5z.FILTER_SCALEOFFSET),
}

for name, available in filters.items():
    status = "Available" if available else "Not available"
    print(f"{name}: {status}")
```

## Trade-offs and Recommendations

### Storage vs. Speed

| Priority | Recommendation | Typical Ratio | Write Speed | Read Speed |
|----------|----------------|---------------|-------------|------------|
| Speed | GZIP=1 + SHUF | 2-3x | Fast | Fast |
| Balanced | GZIP=3 + SHUF | 4-5x | Good | Good |
| Storage | GZIP=6 + SHUF | 5-6x | Moderate | Good |
| Maximum | GZIP=9 + SHUF | 5-7x | Slow | Moderate |

### Data Type Recommendations

**Integer arrays**: GZIP + Shuffle
**Floating-point arrays**: GZIP + Shuffle
**Approximate floats**: Scale-offset + GZIP + Shuffle
**Binary/categorical**: GZIP alone (no shuffle benefit)
**Image data**: Already compressed, use GZIP=1 or none
**Text/strings**: GZIP alone

### Use Case Recommendations

**Active research data**: GZIP=1 or GZIP=3 + Shuffle
**Archived data**: GZIP=6 + Shuffle + Fletcher32
**Distributed data**: GZIP=3 + Shuffle (portable)
**High-speed writes**: No compression or GZIP=1
**Network transfer**: GZIP=3 + Shuffle (balance size/decompress speed)

## Common Issues and Solutions

**Problem**: h5repack reports "filter not available"
**Solution**: Check filter availability, may need to rebuild HDF5 library with filter support. Alternatively, switch to a different filter.

**Problem**: Compressed file larger than original
**Solution**: Data may already be compressed, or not well-suited to this filter; remove filter, potentially try another.

**Problem**: Very slow compression
**Solution**: Reduce GZIP level (use 1-3 instead of 6-9), switch filters, or accept penalty - consult user.

**Problem**: Cannot read file on other system
**Solution**: Ensure same filters available; GZIP/Shuffle most portable

## Best Practices

1. **Default to GZIP=3 + Shuffle** for numeric data
2. **Test on representative data** before committing to settings
3. **Measure before optimizing** - know your baseline
4. **Consider access patterns** - compression affects read performance
5. **Use h5repack** for existing files rather than recreating
6. **Document filter choices** in dataset attributes or metadata
7. **Check portability** if sharing files across systems
8. **Benchmark realistic workloads**, not synthetic tests
9. **Don't over-optimize** - GZIP=3 vs GZIP=9 rarely matters
10. **Combine with chunking** optimization for best results

## Additional Reading

- HDF Group: [HDF5 Data Compression Demystified](https://www.hdfgroup.org/2017/05/24/hdf5-data-compression-demystified-2-performance-tuning/)
- HDF Group: [Improving I/O Performance When Working with HDF5 Compressed Datasets](https://support.hdfgroup.org/documentation/hdf5/latest/improve_compressed_perf.html)
- HDF Group: [h5repack Tool Guide](https://portal.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html)
