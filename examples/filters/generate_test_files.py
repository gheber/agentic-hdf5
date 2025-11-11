#!/usr/bin/env python3
"""
Generate test HDF5 files for demonstrating filter capabilities.

Creates two files:
1. compressed_data.h5 - Already compressed with gzip+shuffle (name doesn't reveal filter)
2. raw_data.h5 - Uncompressed data ready for compression
"""

import h5py
import numpy as np
import os

def generate_files():
    """Generate test HDF5 files with and without compression."""

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate sample data that compresses well
    # Temperature measurements over time
    time_steps = 2000
    spatial_points = 50

    # Create realistic temperature data with patterns
    base_temp = 20.0  # Base temperature in Celsius
    seasonal_variation = 10.0 * np.sin(np.linspace(0, 4*np.pi, time_steps))  # Seasonal
    daily_variation = 5.0 * np.sin(np.linspace(0, 100*np.pi, time_steps))  # Daily cycles
    noise = np.random.normal(0, 1.0, time_steps)

    temperature = base_temp + seasonal_variation + daily_variation + noise

    # Spatial field data (2D grid that changes over time)
    spatial_data = np.zeros((time_steps, spatial_points, spatial_points))
    x, y = np.meshgrid(np.linspace(-5, 5, spatial_points), np.linspace(-5, 5, spatial_points))

    for t in range(time_steps):
        # Create evolving pattern
        phase = t / time_steps * 2 * np.pi
        spatial_data[t] = np.sin(x + phase) * np.cos(y + phase) + np.random.normal(0, 0.1, (spatial_points, spatial_points))

    # Pressure measurements (correlated with temperature)
    pressure = 1013.25 + temperature * 0.5 + np.random.normal(0, 2.0, time_steps)

    print("Generated data:")
    print(f"  Temperature: {temperature.shape}, {temperature.nbytes / 1024 / 1024:.2f} MB")
    print(f"  Spatial field: {spatial_data.shape}, {spatial_data.nbytes / 1024 / 1024:.2f} MB")
    print(f"  Pressure: {pressure.shape}, {pressure.nbytes / 1024 / 1024:.2f} MB")
    print()

    # File 1: Compressed with gzip + shuffle
    # Name doesn't reveal the filter used
    compressed_file = "compressed_data.h5"
    print(f"Creating {compressed_file} with compression (gzip level 6 + shuffle)...")

    with h5py.File(compressed_file, 'w') as f:
        # Create datasets with compression
        ds_temp = f.create_dataset(
            'measurements/temperature',
            data=temperature,
            compression='gzip',
            compression_opts=6,
            shuffle=True,
            chunks=(1000,)
        )
        ds_temp.attrs['units'] = 'Celsius'
        ds_temp.attrs['description'] = 'Temperature measurements over time'
        ds_temp.attrs['measurement_interval'] = '1 hour'

        ds_spatial = f.create_dataset(
            'measurements/spatial_field',
            data=spatial_data,
            compression='gzip',
            compression_opts=6,
            shuffle=True,
            chunks=(50, 25, 25)
        )
        ds_spatial.attrs['units'] = 'arbitrary'
        ds_spatial.attrs['description'] = 'Spatial field measurements over time'
        ds_spatial.attrs['grid_size'] = '50x50'

        ds_pressure = f.create_dataset(
            'measurements/pressure',
            data=pressure,
            compression='gzip',
            compression_opts=6,
            shuffle=True,
            chunks=(1000,)
        )
        ds_pressure.attrs['units'] = 'hPa'
        ds_pressure.attrs['description'] = 'Atmospheric pressure measurements'

        # Add metadata
        f.attrs['experiment'] = 'Climate monitoring station'
        f.attrs['location'] = 'Mountain observatory'
        f.attrs['start_date'] = '2024-01-01'
        f.attrs['instrument'] = 'Multi-sensor array'

    compressed_size = os.path.getsize(compressed_file) / 1024 / 1024
    print(f"  Created {compressed_file} ({compressed_size:.2f} MB)")
    print()

    # File 2: Uncompressed (raw data)
    uncompressed_file = "raw_data.h5"
    print(f"Creating {uncompressed_file} without compression...")

    with h5py.File(uncompressed_file, 'w') as f:
        # Create datasets without compression
        ds_temp = f.create_dataset(
            'measurements/temperature',
            data=temperature,
            chunks=(1000,)  # Chunked but not compressed
        )
        ds_temp.attrs['units'] = 'Celsius'
        ds_temp.attrs['description'] = 'Temperature measurements over time'
        ds_temp.attrs['measurement_interval'] = '1 hour'

        ds_spatial = f.create_dataset(
            'measurements/spatial_field',
            data=spatial_data,
            chunks=(50, 25, 25)
        )
        ds_spatial.attrs['units'] = 'arbitrary'
        ds_spatial.attrs['description'] = 'Spatial field measurements over time'
        ds_spatial.attrs['grid_size'] = '50x50'

        ds_pressure = f.create_dataset(
            'measurements/pressure',
            data=pressure,
            chunks=(1000,)
        )
        ds_pressure.attrs['units'] = 'hPa'
        ds_pressure.attrs['description'] = 'Atmospheric pressure measurements'

        # Add metadata
        f.attrs['experiment'] = 'Climate monitoring station'
        f.attrs['location'] = 'Mountain observatory'
        f.attrs['start_date'] = '2024-01-01'
        f.attrs['instrument'] = 'Multi-sensor array'

    uncompressed_size = os.path.getsize(uncompressed_file) / 1024 / 1024
    print(f"  Created {uncompressed_file} ({uncompressed_size:.2f} MB)")
    print()

    # Summary
    compression_ratio = uncompressed_size / compressed_size
    space_saved = ((uncompressed_size - compressed_size) / uncompressed_size) * 100

    print("Summary:")
    print(f"  Compressed file:   {compressed_size:.2f} MB")
    print(f"  Uncompressed file: {uncompressed_size:.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Space saved:       {space_saved:.1f}%")
    print()
    print("Files ready for agentic HDF5 demonstration:")
    print(f"  1. {compressed_file} - Detect and optionally decompress")
    print(f"  2. {uncompressed_file} - Analyze and apply optimal compression")

if __name__ == '__main__':
    generate_files()
