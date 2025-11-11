# Complete HDF5 Optimization Workflow

This example demonstrates a complete optimization workflow, from analyzing an unoptimized file through applying improvements and validating results.

## Scenario

Optimize a simulation output file containing temperature and pressure fields for storage and access performance.

## Step 1: Initial Analysis

### Analyze Datasets

h5dump can be used from the command line to get detailed information about the structure of a file and the objects within it.
analyze-and-optimize.py script can provide recommendations based on a limited set of objective criteria.

### Identify Potential Issues

1. Lack of compression
2. Small chunks
3. Misaligned chunks

## Step 2: Determine Optimization Strategy

### Access Pattern Analysis

Determine what the primary access pattern is likely to be (e.g. along a time series), and arrange the data within the file so that access along that dimension is optimized. e.g. Resize chunks so their largest dimension is along a time series, and their dimensions along other axes are smaller.

### Optimization Goals

1. **Reduce file size**: Apply compression (e.g. gzip + shuffle)
2. **Improve slice access**: Align chunks with spatial slices
3. **Reasonable chunk size**: Target 100-500 KB per chunk

## Step 3: Apply Optimizations

### Option A: Using h5repack

These numbers, filters, and value are just for demonstration. Each file may need particular consideration.

```bash
# Create optimized copy
h5repack \
  -l /results/temperature:CHUNK=1x100x100 \
  -l /results/pressure:CHUNK=1x100x100 \
  -f /results/temperature:SHUF \
  -f /results/temperature:GZIP=3 \
  -f /results/pressure:SHUF \
  -f /results/pressure:GZIP=3 \
  simulation_output.h5 \
  simulation_optimized.h5
```

### Option B: Recreate with h5py

For new files, set parameters during creation:

```python
import h5py
import numpy as np

# Create optimized file
with h5py.File('simulation_optimized.h5', 'w') as f:
    # Load original data
    with h5py.File('simulation_output.h5', 'r') as f_orig:
        temp_data = f_orig['/results/temperature'][:]
        press_data = f_orig['/results/pressure'][:]

    # Write with optimization
    f.create_dataset(
        '/results/temperature',
        data=temp_data,
        chunks=(1, 100, 100),
        compression='gzip',
        compression_opts=3,
        shuffle=True
    )

    f.create_dataset(
        '/results/pressure',
        data=press_data,
        chunks=(1, 100, 100),
        compression='gzip',
        compression_opts=3,
        shuffle=True
    )
```

## Step 4: Validate Improvements

- Compare file sizes with `ls` or other command-line tool
- Verify data is the same in both versions via h5py
- Benchmark access performance (performance same read patterns on both files)
- Document optimizations performed and their results
- If expected optimizations are small or non-existent, consider alternative approaches

## Alternative Scenarios

### Maximum Compression (Archival)

For long-term archival, make the tradeoff in favor of file size and against R/W speed (e.g. higher compression levels)
