---
title: "Summary Statistics of Indoor Temperature Readings in Building 7, January 2024"
author: "J. Smith, Facilities Division"
date: "February 2024"
---

# Abstract

We report basic descriptive statistics for indoor air temperature in Room 101 of Building 7 during January 2024, measured by a single automated sensor at hourly intervals. The mean temperature was approximately 21.0°C with a standard deviation of 1.4°C. A total of 55 hourly readings exceeded the 23°C comfort threshold.

# Data

Hourly temperature readings (744 measurements, one per hour for 31 days) are stored in a single HDF5 file, `sensor_readings_2024.h5`, available as a supplementary file alongside this report. The file contains two datasets: `temperature` (float64, length 744, units: degrees Celsius) and `hour` (integer index, 0-743).

# Methods

We computed the following over the full `temperature` dataset:

1. Arithmetic mean
2. Standard deviation (population, i.e., dividing by N)
3. Minimum and maximum values
4. Count of readings exceeding 23.0°C

All computations used NumPy (version ≥1.20).

Note: standard deviation uses the default NumPy convention (`ddof=0`).

# Results

| Statistic | Value |
|-----------|-------|
| Mean | 20.9957°C |
| Std. Dev. | 1.4393°C |
| Minimum | 18.3677°C |
| Maximum | 23.7191°C |
| Hours > 23°C | 55 |

# Conclusion

Room 101 maintained temperatures near the 21°C setpoint. The 55 hours above 23°C (7.4% of the month) suggest occasional overheating, primarily during afternoon periods.
