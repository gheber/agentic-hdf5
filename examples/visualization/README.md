## Visualization Test Files

These files were created by `create_test_files.py` with ground-truth semantic metadata to test the agent's ability to choose appropriate visualizations:

- **test_timeseries.h5**: 1D time series data (temperature measurements over 48 hours)
  - Expected visualization: Line plot
  - Dataset: `/temperature_data` (2000 points)

- **test_spatial_field.h5**: 2D spatial field data (infrared thermography)
  - Expected visualization: Heatmap (pcolormesh or imshow)
  - Dataset: `/surface_temperature` (60×80 grid)

- **test_distribution.h5**: Statistical distribution data (particle sizes)
  - Expected visualization: Histogram
  - Dataset: `/particle_diameters` (5048 measurements, bimodal distribution)

All test files include semantic metadata describing the data type, physical quantity, and expected visualization approach.
