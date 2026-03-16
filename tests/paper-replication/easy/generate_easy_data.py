"""Generate the HDF5 data file referenced by the easy test paper."""
import h5py
import numpy as np

np.random.seed(42)

with h5py.File('sensor_readings_2024.h5', 'w') as f:
    f.attrs['title'] = 'Indoor Temperature Sensor Readings, Building 7, Jan 2024'
    f.attrs['creator'] = 'J. Smith, Facilities Division'
    f.attrs['date_created'] = '2024-02-01'

    temps = 21.0 + 2.0 * np.sin(np.linspace(0, 2 * np.pi * 31, 744)) + np.random.normal(0, 0.3, 744)
    timestamps = np.arange(744)  # hourly readings for 31 days

    dset = f.create_dataset('temperature', data=temps)
    dset.attrs['units'] = 'degrees_C'
    dset.attrs['long_name'] = 'Indoor air temperature, Room 101'
    dset.attrs['sampling'] = 'hourly'

    f.create_dataset('hour', data=timestamps)

    print(f"Mean: {temps.mean():.4f}")
    print(f"Std:  {temps.std():.4f}")
    print(f"Min:  {temps.min():.4f}")
    print(f"Max:  {temps.max():.4f}")
    print(f"Hours above 23C: {(temps > 23.0).sum()}")
