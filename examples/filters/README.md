# Filter Examples

This directory demonstrates agentic HDF5's ability to detect, analyze, and manipulate compression filters on HDF5 datasets.

## Test Files

Run `generate_test_files.py` to create two test files: `compressed_data.h5` to demonstrate filter detection decompression, and `raw_data.h5` to demonstrate filter choice and compression.

## Example Prompts for ahdf5

**For compressed_data.h5:**
- "What compression is used on the datasets in this file?"
- "Show me the compression ratio achieved"
- "Decompress the temperature dataset"

**For raw_data.h5:**
- "What's the size of this file and could it benefit from compression?"
- "Apply appropriate compression to all datasets"
- "Compare different compression levels on the spatial_field dataset"
