# Semantic Metadata Example

Demonstration of HDF5 files with semantic metadata (SMD) annotations and vectorized search.

## Files

- `generate_smd_file.py` - Creates an example HDF5 file with semantic metadata attributes
- `vectorize_file.py` - Generates vector embeddings from the semantic metadata for search
- `example_with_smd.h5` - Pre-generated example file with environmental sensor data

## Usage

Generate a new example file:
```bash
./generate_smd_file.py
```

Vectorize the semantic metadata:
```bash
./vectorize_file.py example_with_smd.h5
```
