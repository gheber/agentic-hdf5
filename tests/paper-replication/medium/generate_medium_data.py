"""Generate the HDF5 data file referenced by the medium test paper."""
import h5py
import numpy as np

np.random.seed(7)

N_STATIONS = 12
N_DAYS = 365

with h5py.File('river_discharge_2023.h5', 'w') as f:
    f.attrs['Conventions'] = 'CF-1.8'
    f.attrs['title'] = 'Daily River Discharge at 12 Monitoring Stations, Elkhorn Basin, 2023'
    f.attrs['institution'] = 'State Water Resources Institute'
    f.attrs['source'] = 'Automated stream gauges, calibrated monthly'
    f.attrs['date_created'] = '2024-01-15'
    f.attrs['license'] = 'CC-BY-4.0'
    f.attrs['id'] = '10.5281/zenodo.00000001'  # fake DOI

    # Station metadata
    station_names = [f'ELK-{i:03d}' for i in range(1, N_STATIONS + 1)]
    upstream_area_km2 = np.array([12.4, 28.1, 45.7, 63.2, 89.0, 112.5,
                                   145.3, 178.9, 210.6, 256.0, 301.4, 350.8])

    g = f.create_group('stations')
    g.create_dataset('name', data=station_names)
    area_ds = g.create_dataset('upstream_area', data=upstream_area_km2)
    area_ds.attrs['units'] = 'km^2'

    # Daily discharge: base flow proportional to upstream area + seasonal + noise
    day = np.arange(N_DAYS)
    seasonal = 1.0 + 0.6 * np.sin(2 * np.pi * (day - 80) / 365)  # peak ~late March

    discharge = np.zeros((N_STATIONS, N_DAYS))
    for i in range(N_STATIONS):
        base = 0.015 * upstream_area_km2[i]  # m^3/s per km^2
        discharge[i] = base * seasonal + np.random.exponential(0.1 * base, N_DAYS)

    dset = f.create_dataset('discharge', data=discharge, chunks=(1, N_DAYS))
    dset.attrs['units'] = 'm^3/s'
    dset.attrs['long_name'] = 'Daily mean river discharge'

    f.create_dataset('day_of_year', data=day + 1)

    # Print key statistics for paper
    annual_means = discharge.mean(axis=1)
    print("Annual mean discharge per station (m^3/s):")
    for i in range(N_STATIONS):
        print(f"  {station_names[i]}: {annual_means[i]:.3f}")

    r = np.corrcoef(upstream_area_km2, annual_means)[0, 1]
    print(f"\nPearson r (area vs mean discharge): {r:.4f}")

    slope, intercept = np.polyfit(upstream_area_km2, annual_means, 1)
    print(f"Linear fit: Q = {slope:.6f} * A + {intercept:.4f}")
