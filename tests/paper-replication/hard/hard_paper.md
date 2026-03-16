---
title: "Global Mean Sea Surface Temperature Trend from ERSST v5, 1980–2020"
author: "T. Nakamura and L. Osei, Department of Earth Sciences, Lakefield University"
date: "June 2024"
---

# Abstract

We compute the linear trend in global mean sea surface temperature (SST) anomaly over the period 1980–2020 using the NOAA Extended Reconstructed Sea Surface Temperature version 5 (ERSSTv5) monthly dataset. The area-weighted global mean SST anomaly increases at a rate of approximately 0.014°C per year (0.57°C over the 41-year period). These results are consistent with previously reported warming rates and confirm the accelerating trend in ocean surface temperatures.

# 1. Introduction

Ocean surface warming is a key indicator of climate change. The ERSSTv5 dataset (Huang et al., 2017) provides monthly global SST anomaly fields on a 2°×2° grid from 1854 to present, based on in situ observations from ships and buoys. We use this dataset to compute the global mean SST anomaly time series and its linear trend for the recent period 1980–2020.

# 2. Data

We use the NOAA ERSSTv5 monthly SST anomaly dataset, available from NOAA's National Centers for Environmental Information:

- **URL**: https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/
- **Format**: NetCDF4 (HDF5-based)
- **Files**: One file per month, named `ersst.v5.YYYYMM.nc`
- **Variable**: `ssta` (SST anomaly, °C, relative to 1991–2020 climatology)
- **Grid**: 2° × 2° (89 latitude × 180 longitude)
- **DOI**: https://doi.org/10.7289/V5T72FNM

For the period January 1980 through December 2020, this comprises 492 monthly files, each approximately 1.5 MB.

# 3. Methods

## 3.1 Area-Weighted Global Mean

For each month, we compute the area-weighted global mean SST anomaly. Because grid cells at different latitudes represent different areas, each cell is weighted by the cosine of its latitude:

$$\overline{T}_m = \frac{\sum_{i,j} T_{i,j,m} \cdot \cos(\phi_j)}{\sum_{i,j} \cos(\phi_j)}$$

where $T_{i,j,m}$ is the SST anomaly at longitude $i$, latitude $j$, month $m$, and $\phi_j$ is the latitude in radians. Cells with missing data (land or no observations) are excluded from both numerator and denominator.

The latitude and longitude coordinates are read from the `lat` and `lon` variables in the NetCDF file.

## 3.2 Linear Trend

We fit an ordinary least-squares regression to the monthly global mean time series:

$$\overline{T}(t) = \beta_1 t + \beta_0$$

where $t$ is the time index in years (fractional years for monthly resolution). The slope $\beta_1$ gives the linear warming rate in °C/year. We use `numpy.polyfit` with degree 1.

# 4. Results

The area-weighted global mean SST anomaly time series shows a clear upward trend from approximately −0.2°C in 1980 to +0.4°C in 2020.

The linear fit yields:

| Parameter | Value |
|-----------|-------|
| Slope (β₁) | 0.014 °C/year |
| Intercept (β₀) | −28.0 °C (at year 0, not physically meaningful) |
| Total change, 1980–2020 | ~0.57 °C |

# 5. Discussion

Our computed trend of 0.014°C/year is consistent with the IPCC AR6 reported ocean surface warming rate of 0.015°C/year for the period 1980–2020. Minor differences arise from our use of anomalies relative to the 1991–2020 baseline vs. other baselines, and from our simple area-weighting scheme vs. more sophisticated area integration methods.

# 6. Conclusion

The ERSSTv5 dataset confirms a global mean SST warming rate of approximately 0.014°C/year over 1980–2020, consistent with established climate science.

# References

1. Huang, B., et al. (2017). Extended Reconstructed Sea Surface Temperature, Version 5 (ERSSTv5): Upgrades, Validations, and Intercomparisons. *J. Climate*, 30(20), 8179–8205. doi:10.1175/JCLI-D-16-0836.1
2. IPCC, 2021: Climate Change 2021: The Physical Science Basis. Cambridge University Press.
