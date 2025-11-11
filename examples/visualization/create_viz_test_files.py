#!/usr/bin/env python3
"""
Create three HDF5 test files with ground-truth SMD that should allow an agent
to infer the appropriate visualization approach from contextual clues.

File 1: timeseries.h5 - 1D temporal data (agent should infer: LINE plot)
File 2: spatial_field.h5 - 2D spatial field (agent should infer: PCOLORMESH/IMSHOW)
File 3: distribution_analysis.h5 - Statistical distribution (agent should infer: HISTOGRAM)

SMD does NOT explicitly mention plot types - agent must infer from data characteristics.
"""

import h5py
import numpy as np


def create_timeseries_file():
    """
    Create a file with time series data that should trigger LINE plot visualization.
    Contains temperature sensor data over time with clear temporal patterns.
    """
    filename = "test_timeseries.h5"

    with h5py.File(filename, 'w') as f:
        # Create realistic time series data
        n_points = 2000
        time = np.linspace(0, 48, n_points)  # 48 hours

        # Temperature with daily cycle + noise + trend
        daily_cycle = 20 + 5 * np.sin(2 * np.pi * time / 24)  # 24-hour period
        trend = 0.1 * time  # Gradual warming
        noise = np.random.normal(0, 0.5, n_points)
        temperature = daily_cycle + trend + noise

        # Create dataset
        ds = f.create_dataset('temperature_data', data=temperature)
        ds.attrs['units'] = 'celsius'
        ds.attrs['sampling_rate'] = '90 seconds'
        ds.attrs['location'] = 'Lab Room 3B'

        # Add time coordinate
        time_ds = f.create_dataset('time', data=time)
        time_ds.attrs['units'] = 'hours'
        time_ds.attrs['description'] = 'Time since start of experiment'

        # Ground truth SMD for the dataset
        smd = """Temperature measurements from a laboratory environmental sensor.
Data type: Time series (1-dimensional temporal data)
Physical quantity: Temperature in degrees Celsius
Temporal characteristics: 48-hour continuous recording, sampled every 90 seconds
Expected patterns: Daily thermal cycles (24-hour period), gradual temperature drift
Data structure: Sequential measurements, preserving temporal ordering is critical
Units: Celsius (°C)
Value range: Approximately 18-28°C
Special notes: Shows clear diurnal variation and slight warming trend over observation period"""

        f.attrs['ahdf5-smd-temperature_data'] = smd

        # Root group SMD
        root_smd = """Laboratory temperature monitoring dataset.
Contains time-series temperature measurements for environmental monitoring.
Primary dataset is 1D temporal data showing temperature evolution over time."""
        f.attrs['ahdf5-smd-root'] = root_smd

    print(f"Created {filename}")
    print(f"  - Dataset shape: {temperature.shape}")
    print(f"  - Data type: 1D time series")
    print(f"  - Expected visualization: LINE PLOT")
    print()


def create_spatial_field_file():
    """
    Create a file with 2D spatial field data that should trigger PCOLORMESH/IMSHOW heatmap.
    Contains temperature distribution across a surface.
    """
    filename = "test_spatial_field.h5"

    with h5py.File(filename, 'w') as f:
        # Create realistic 2D spatial temperature field
        nx, ny = 80, 60
        x = np.linspace(0, 10, nx)  # cm
        y = np.linspace(0, 7.5, ny)  # cm
        X, Y = np.meshgrid(x, y)

        # Temperature field with hot spots and gradients
        # Hot spot in center
        center_x, center_y = 5, 3.75
        hot_spot = 100 * np.exp(-((X - center_x)**2 + (Y - center_y)**2) / 2)

        # Corner cold region
        corner_cold = -20 * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 1)

        # Background gradient
        background = 50 + 2 * X - 1.5 * Y

        # Combine with noise
        noise = np.random.normal(0, 2, (ny, nx))
        temperature_field = background + hot_spot + corner_cold + noise

        # Create dataset
        ds = f.create_dataset('surface_temperature', data=temperature_field)
        ds.attrs['units'] = 'celsius'
        ds.attrs['dimensions'] = 'y,x'
        ds.attrs['x_units'] = 'centimeters'
        ds.attrs['y_units'] = 'centimeters'
        ds.attrs['measurement_method'] = 'Infrared thermography'

        # Coordinate arrays
        x_ds = f.create_dataset('x_coords', data=x)
        x_ds.attrs['units'] = 'cm'
        x_ds.attrs['description'] = 'Horizontal position across surface'

        y_ds = f.create_dataset('y_coords', data=y)
        y_ds.attrs['units'] = 'cm'
        y_ds.attrs['description'] = 'Vertical position across surface'

        # Ground truth SMD
        smd = """Infrared thermography measurement of surface temperature distribution.
Data type: Two-dimensional spatial field (gridded spatial data)
Physical quantity: Temperature in degrees Celsius across a rectangular surface
Spatial characteristics: 80x60 regular grid covering 10cm × 7.5cm area
Expected patterns: Localized hot regions, thermal gradients, cold spots
Data structure: 2D array where position indices correspond to physical coordinates
Dimensionality: 2D (rows = y-position, columns = x-position)
Units: Celsius (°C)
Value range: Approximately 30-150°C with localized extremes
Special notes: Hot spot visible near center, cold region in corner, general left-to-right warming gradient
Analysis priorities: Spatial structure is paramount, temperature values vary continuously across the surface"""

        f.attrs['ahdf5-smd-surface_temperature'] = smd

        # Root group SMD
        root_smd = """Surface temperature mapping dataset from infrared thermography.
Contains 2D spatial field data showing temperature distribution across a surface.
Primary dataset is 2D gridded data with spatial coordinates."""
        f.attrs['ahdf5-smd-root'] = root_smd

    print(f"Created {filename}")
    print(f"  - Dataset shape: {temperature_field.shape}")
    print(f"  - Data type: 2D spatial field")
    print(f"  - Expected visualization: PCOLORMESH or IMSHOW HEATMAP")
    print()


def create_distribution_file():
    """
    Create a file with measurement data that should trigger HISTOGRAM visualization.
    Contains particle size measurements where distribution analysis is key.
    """
    filename = "test_distribution.h5"

    with h5py.File(filename, 'w') as f:
        # Create particle size distribution (bimodal)
        n_samples = 5000

        # Two populations of particles
        small_particles = np.random.normal(2.5, 0.8, int(n_samples * 0.6))  # 60% small
        large_particles = np.random.normal(8.0, 1.5, int(n_samples * 0.4))  # 40% large

        # Combine and add some outliers
        particle_sizes = np.concatenate([small_particles, large_particles])
        outliers = np.random.uniform(15, 20, 50)
        particle_sizes = np.concatenate([particle_sizes, outliers])

        # Remove negative values (physical constraint)
        particle_sizes = particle_sizes[particle_sizes > 0]

        # Shuffle to remove order
        np.random.shuffle(particle_sizes)

        # Create dataset
        ds = f.create_dataset('particle_diameters', data=particle_sizes)
        ds.attrs['units'] = 'micrometers'
        ds.attrs['measurement_method'] = 'Dynamic light scattering'
        ds.attrs['sample_id'] = 'P-2024-11-17-A'
        ds.attrs['n_measurements'] = len(particle_sizes)

        # Ground truth SMD
        smd = """Particle size measurements from dynamic light scattering analysis.
Data type: Statistical distribution (1-dimensional measurement collection)
Physical quantity: Particle diameter in micrometers
Statistical characteristics: Bimodal distribution with two distinct populations
Expected patterns: Two peaks (small particles ~2.5μm, large particles ~8μm), some outliers
Data structure: Unordered collection of individual measurements, no inherent sequence
Order significance: NONE - temporal or spatial order is not meaningful
Analysis focus: Distribution shape, central tendency, dispersion, and multimodality
Value range: Primarily 0.5-12 μm with occasional outliers up to 20μm
Sample size: ~5000 individual particle measurements
Special notes: Clear bimodal distribution indicates two distinct particle populations in sample
Physical interpretation: Likely mixture of fine and coarse particles, possibly two material types"""

        f.attrs['ahdf5-smd-particle_diameters'] = smd

        # Root group SMD
        root_smd = """Particle size analysis dataset from light scattering measurements.
Contains statistical distribution data of particle diameters.
Primary dataset is unordered measurement collection for distribution analysis."""
        f.attrs['ahdf5-smd-root'] = root_smd

        # Add some summary statistics as additional datasets
        stats_group = f.create_group('statistics')
        stats_group.attrs['description'] = 'Computed statistical summaries'

        stats = {
            'mean': np.mean(particle_sizes),
            'median': np.median(particle_sizes),
            'std': np.std(particle_sizes),
            'min': np.min(particle_sizes),
            'max': np.max(particle_sizes),
            'q25': np.percentile(particle_sizes, 25),
            'q75': np.percentile(particle_sizes, 75)
        }

        for name, value in stats.items():
            stats_group.attrs[name] = value

    print(f"Created {filename}")
    print(f"  - Dataset shape: {particle_sizes.shape}")
    print(f"  - Data type: Unordered distribution")
    print(f"  - Expected visualization: HISTOGRAM")
    print(f"  - Distribution: Bimodal (peaks at ~2.5μm and ~8μm)")
    print()


def verify_files():
    """Verify the created files and show their SMD"""
    files = [
        ('test_timeseries.h5', 'temperature_data'),
        ('test_spatial_field.h5', 'surface_temperature'),
        ('test_distribution.h5', 'particle_diameters')
    ]

    print("\n" + "="*80)
    print("VERIFICATION: SMD Content Summary")
    print("="*80 + "\n")

    for filename, dataset_path in files:
        print(f"\n{filename}:")
        print("-" * len(filename))
        with h5py.File(filename, 'r') as f:
            smd_attr = f'ahdf5-smd-{dataset_path}'
            if smd_attr in f.attrs:
                smd = f.attrs[smd_attr]
                if isinstance(smd, bytes):
                    smd = smd.decode('utf-8')
                # Print first few lines
                lines = smd.split('\n')
                for line in lines[:5]:
                    print(f"  {line}")
                print(f"  ... ({len(lines)} total lines)")

            # Show dataset info
            ds = f[dataset_path]
            print(f"\n  Dataset info:")
            print(f"    Shape: {ds.shape}")
            print(f"    Dtype: {ds.dtype}")
            if 'units' in ds.attrs:
                print(f"    Units: {ds.attrs['units']}")


if __name__ == '__main__':
    print("Creating test HDF5 files with ground-truth SMD...\n")
    print("="*80)

    create_timeseries_file()
    create_spatial_field_file()
    create_distribution_file()

    verify_files()

    print("\n" + "="*80)
    print("EXPECTED AGENT BEHAVIOR:")
    print("="*80)
    print("""
1. test_timeseries.h5 / temperature_data
   → Agent should infer: plot_type='line'
   → Key SMD clues: "Time series", "temporal characteristics", "sequential measurements",
     "preserving temporal ordering is critical"

2. test_spatial_field.h5 / surface_temperature
   → Agent should infer: plot_type='pcolormesh' or 'imshow'
   → Key SMD clues: "Two-dimensional spatial field", "gridded spatial data",
     "2D array where position indices correspond to physical coordinates"

3. test_distribution.h5 / particle_diameters
   → Agent should infer: plot_type='hist'
   → Key SMD clues: "Statistical distribution", "unordered collection",
     "Order significance: NONE", "Distribution shape, central tendency, dispersion"

NOTE: SMD does NOT explicitly mention visualization types - agent must reason from data characteristics.
""")

    print("Files created successfully! Test with your agent using these files.")
