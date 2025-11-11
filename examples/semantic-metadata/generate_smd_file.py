#!/usr/bin/env python3
"""
Generate an HDF5 file with semantic metadata for VSMD testing.

This script creates a realistic HDF5 file with various datasets, groups,
and committed datatypes, all annotated with semantic metadata (SMD).
"""

import h5py
import numpy as np
from datetime import datetime, timedelta


def generate_smd_file(output_path='example_with_smd.h5'):
    """
    Generate an HDF5 file with diverse datasets and semantic metadata.

    Args:
        output_path: Path where the HDF5 file will be created
    """
    print(f"Generating HDF5 file with SMD: {output_path}")

    with h5py.File(output_path, 'w') as f:
        # Add file-level SMD
        f.attrs['ahdf5-smd-file_description'] = (
            "Environmental monitoring dataset from the Pacific Northwest Climate Observatory. "
            "Contains sensor measurements from multiple weather stations collected during the "
            "2024 summer monitoring campaign. Includes temperature, pressure, humidity, wind, "
            "and solar radiation data sampled at 5-minute intervals."
        )

        # Create committed datatype for timestamp
        timestamp_dtype = h5py.h5t.py_create(np.dtype('int64'))
        timestamp_dtype.commit(f.id, b'unix_timestamp')
        f['unix_timestamp'].attrs['ahdf5-smd-datatype_description'] = (
            "Unix timestamp datatype representing seconds since epoch (1970-01-01 00:00:00 UTC). "
            "Used throughout the file for temporal indexing of all sensor measurements."
        )

        # Generate time series data
        n_samples = 1000
        start_time = datetime(2024, 7, 1, 0, 0, 0)
        times = np.array([int((start_time + timedelta(minutes=5*i)).timestamp())
                         for i in range(n_samples)], dtype='int64')

        # === OUTDOOR SENSORS GROUP ===
        outdoor = f.create_group('outdoor_sensors')
        outdoor.attrs['ahdf5-smd-group_description'] = (
            "Collection of outdoor environmental sensors located at the main weather station "
            "installation. All sensors are mounted on a standard 10-meter meteorological tower "
            "with appropriate radiation shielding and calibrated according to WMO standards."
        )

        # Temperature dataset
        temp_data = 20 + 10 * np.sin(np.linspace(0, 4*np.pi, n_samples)) + np.random.randn(n_samples) * 2
        ds_temp = outdoor.create_dataset('temperature', data=temp_data, compression='gzip')
        ds_temp.attrs['units'] = 'Celsius'
        ds_temp.attrs['ahdf5-smd-dataset_description'] = (
            "Air temperature measurements in degrees Celsius recorded at 2 meters above ground level. "
            "Measured using a platinum resistance thermometer (PT100) with ±0.1°C accuracy. "
            "Data has been quality-controlled to remove sensor drift and spurious readings. "
            "Valid range: -40°C to 50°C. Missing values are indicated by NaN."
        )

        # Pressure dataset
        pressure_data = 1013 + 5 * np.sin(np.linspace(0, 2*np.pi, n_samples)) + np.random.randn(n_samples) * 1
        ds_pressure = outdoor.create_dataset('atmospheric_pressure', data=pressure_data, compression='gzip')
        ds_pressure.attrs['units'] = 'hPa'
        ds_pressure.attrs['ahdf5-smd-dataset_description'] = (
            "Atmospheric pressure measurements in hectopascals (hPa) at station elevation of 245 meters. "
            "Measured with a silicon capacitive pressure sensor calibrated against a mercury barometer. "
            "Values have been corrected for temperature effects and represent station pressure, "
            "not sea-level pressure. Typical range: 980-1040 hPa."
        )

        # Humidity dataset
        humidity_data = np.clip(60 + 20 * np.cos(np.linspace(0, 4*np.pi, n_samples)) + np.random.randn(n_samples) * 5, 0, 100)
        ds_humidity = outdoor.create_dataset('relative_humidity', data=humidity_data, compression='gzip')
        ds_humidity.attrs['units'] = 'percent'
        ds_humidity.attrs['ahdf5-smd-dataset_description'] = (
            "Relative humidity measurements expressed as percentage (0-100%). "
            "Measured using a capacitive polymer sensor housed in a ventilated radiation shield. "
            "Sensor accuracy is ±2% RH in the range 10-90% RH. Data represents humidity relative "
            "to saturation at the measured air temperature."
        )

        # Wind speed dataset
        wind_speed = np.abs(5 + 3 * np.sin(np.linspace(0, 6*np.pi, n_samples)) + np.random.randn(n_samples) * 1.5)
        ds_wind = outdoor.create_dataset('wind_speed', data=wind_speed, compression='gzip')
        ds_wind.attrs['units'] = 'm/s'
        ds_wind.attrs['ahdf5-smd-dataset_description'] = (
            "Wind speed measurements in meters per second at 10 meters above ground level. "
            "Measured with a cup anemometer with starting threshold of 0.5 m/s and accuracy of "
            "±0.3 m/s or ±3%, whichever is larger. Values represent 5-minute averages. "
            "Calm conditions (< 0.5 m/s) are recorded as zero."
        )

        # Solar radiation dataset
        solar_rad = np.maximum(0, 800 * np.sin(np.linspace(0, 4*np.pi, n_samples)) + np.random.randn(n_samples) * 50)
        ds_solar = outdoor.create_dataset('solar_radiation', data=solar_rad, compression='gzip')
        ds_solar.attrs['units'] = 'W/m²'
        ds_solar.attrs['ahdf5-smd-dataset_description'] = (
            "Incoming solar radiation (global horizontal irradiance) in watts per square meter. "
            "Measured using a thermopile pyranometer with spectral range 300-2800 nm. "
            "Values represent total shortwave radiation from both direct sun and diffuse sky. "
            "Nighttime values are zero. Maximum expected value is approximately 1200 W/m²."
        )

        # Timestamps
        ds_time = outdoor.create_dataset('timestamp', data=times, compression='gzip')
        ds_time.attrs['ahdf5-smd-dataset_description'] = (
            "Unix timestamps corresponding to the measurement times for all outdoor sensor data. "
            "Each timestamp represents the end of the 5-minute sampling interval. "
            "All sensors are synchronized to UTC time using GPS receivers."
        )

        # === INDOOR SENSORS GROUP ===
        indoor = f.create_group('indoor_sensors')
        indoor.attrs['ahdf5-smd-group_description'] = (
            "Indoor environmental monitoring sensors located inside the climate-controlled "
            "instrument shelter adjacent to the main weather tower. Used for equipment "
            "health monitoring and indoor air quality assessment."
        )

        # Indoor temperature
        indoor_temp = 22 + 2 * np.sin(np.linspace(0, 2*np.pi, n_samples)) + np.random.randn(n_samples) * 0.5
        ds_indoor_temp = indoor.create_dataset('temperature', data=indoor_temp, compression='gzip')
        ds_indoor_temp.attrs['units'] = 'Celsius'
        ds_indoor_temp.attrs['ahdf5-smd-dataset_description'] = (
            "Indoor air temperature inside the instrument shelter in degrees Celsius. "
            "Maintained within 20-24°C by HVAC system to ensure optimal equipment performance. "
            "Measured with a digital thermistor sensor with ±0.5°C accuracy."
        )

        # Indoor humidity
        indoor_humidity = 45 + 10 * np.cos(np.linspace(0, 2*np.pi, n_samples)) + np.random.randn(n_samples) * 3
        ds_indoor_humidity = indoor.create_dataset('relative_humidity', data=indoor_humidity, compression='gzip')
        ds_indoor_humidity.attrs['units'] = 'percent'
        ds_indoor_humidity.attrs['ahdf5-smd-dataset_description'] = (
            "Indoor relative humidity percentage inside the instrument shelter. "
            "Target range is 30-60% RH to prevent condensation on electronics while "
            "avoiding electrostatic discharge issues. Measured with solid-state humidity sensor."
        )

        # === SOIL SENSORS GROUP ===
        soil = f.create_group('soil_sensors')
        soil.attrs['ahdf5-smd-group_description'] = (
            "Subsurface soil measurements from buried sensors at multiple depths. "
            "Sensors are installed in undisturbed soil profile to monitor temperature "
            "and moisture conditions relevant to agriculture and hydrology research."
        )

        # Soil temperature at 10cm depth
        soil_temp_10cm = 15 + 5 * np.sin(np.linspace(0, 4*np.pi, n_samples) - np.pi/4) + np.random.randn(n_samples) * 1
        ds_soil_temp = soil.create_dataset('temperature_10cm', data=soil_temp_10cm, compression='gzip')
        ds_soil_temp.attrs['units'] = 'Celsius'
        ds_soil_temp.attrs['ahdf5-smd-dataset_description'] = (
            "Soil temperature at 10 centimeters depth measured in degrees Celsius. "
            "Sensor is a thermistor encapsulated in waterproof housing, installed horizontally "
            "at the specified depth. Temperature lags and dampens compared to air temperature "
            "due to thermal inertia of soil. Important for plant root zone monitoring."
        )

        # Soil moisture
        soil_moisture = 0.25 + 0.1 * np.cos(np.linspace(0, 3*np.pi, n_samples)) + np.random.randn(n_samples) * 0.03
        ds_soil_moisture = soil.create_dataset('volumetric_water_content', data=soil_moisture, compression='gzip')
        ds_soil_moisture.attrs['units'] = 'm³/m³'
        ds_soil_moisture.attrs['ahdf5-smd-dataset_description'] = (
            "Volumetric soil water content measured as cubic meters of water per cubic meter of soil. "
            "Measured using time-domain reflectometry (TDR) sensor at 10cm depth. "
            "Values typically range from 0.05 (dry) to 0.45 (saturated) depending on soil type. "
            "Site soil is loamy sand with field capacity around 0.35 m³/m³."
        )

        # === DERIVED PRODUCTS GROUP ===
        derived = f.create_group('derived_products')
        derived.attrs['ahdf5-smd-group_description'] = (
            "Computed quantities derived from raw sensor measurements using standard "
            "meteorological algorithms. These products provide additional insight into "
            "atmospheric conditions and energy balance."
        )

        # Dewpoint temperature
        # Simplified dewpoint calculation for demo
        dewpoint = temp_data - ((100 - humidity_data) / 5)
        ds_dewpoint = derived.create_dataset('dewpoint_temperature', data=dewpoint, compression='gzip')
        ds_dewpoint.attrs['units'] = 'Celsius'
        ds_dewpoint.attrs['ahdf5-smd-dataset_description'] = (
            "Dewpoint temperature in degrees Celsius, calculated from air temperature and "
            "relative humidity using the Magnus-Tetens approximation. Represents the temperature "
            "to which air must be cooled for saturation to occur. Useful for assessing "
            "condensation risk and human thermal comfort."
        )

        # Heat index (apparent temperature)
        heat_index = temp_data + 0.5 * (humidity_data / 100) * (temp_data - 14)
        ds_heat_index = derived.create_dataset('heat_index', data=heat_index, compression='gzip')
        ds_heat_index.attrs['units'] = 'Celsius'
        ds_heat_index.attrs['ahdf5-smd-dataset_description'] = (
            "Heat index (apparent temperature) in degrees Celsius, combining air temperature "
            "and relative humidity to represent perceived temperature. Calculated using "
            "simplified Steadman equation. Values above 32°C indicate increasing heat stress. "
            "Most relevant during summer conditions with high humidity."
        )

        # === QUALITY CONTROL GROUP ===
        qc = f.create_group('quality_control')
        qc.attrs['ahdf5-smd-group_description'] = (
            "Quality control flags and diagnostic information for all measurements. "
            "Each flag indicates the result of automated quality checks including "
            "range validation, rate-of-change tests, and cross-sensor consistency checks."
        )

        # QC flags (0 = good, 1 = suspect, 2 = bad)
        qc_flags = np.random.choice([0, 0, 0, 0, 0, 0, 0, 0, 1, 2], size=n_samples)
        ds_qc = qc.create_dataset('temperature_qc_flags', data=qc_flags, compression='gzip')
        ds_qc.attrs['ahdf5-smd-dataset_description'] = (
            "Quality control flags for outdoor temperature measurements. "
            "Flag values: 0 = data passed all quality checks, 1 = data is suspect and should "
            "be used with caution, 2 = data failed quality checks and should not be used. "
            "Flags are assigned by automated algorithms checking for sensor malfunction, "
            "out-of-range values, and unrealistic rate of change."
        )

    print(f"Successfully created {output_path}")
    print(f"File contains {n_samples} timesteps across multiple sensor groups")
    print("\nStructure:")
    print("  /outdoor_sensors/ - main weather measurements")
    print("  /indoor_sensors/ - instrument shelter conditions")
    print("  /soil_sensors/ - subsurface measurements")
    print("  /derived_products/ - computed quantities")
    print("  /quality_control/ - QC flags")
    print("\nAll objects have semantic metadata (ahdf5-smd-*) attributes")


if __name__ == '__main__':
    generate_smd_file('example_with_smd.h5')
