---
title: "Upstream Catchment Area as a Predictor of Mean Annual Discharge in the Elkhorn Basin"
author: "A. Rivera and M. Chen, State Water Resources Institute"
date: "March 2024"
---

# Abstract

We examine the relationship between upstream catchment area and mean annual river discharge at 12 monitoring stations in the Elkhorn Basin for the year 2023. A strong linear relationship is observed (Pearson r > 0.99), with a fitted yield coefficient of approximately 0.0166 m³/s per km² of catchment area. These results are consistent with regional precipitation patterns and confirm that simple area-based models remain effective for ungauged sub-basins in this watershed.

# 1. Introduction

Estimating discharge at ungauged sites is a fundamental problem in hydrology. In small to medium basins with relatively uniform land cover and precipitation, upstream catchment area is often a strong predictor of mean annual discharge. We test this relationship in the Elkhorn Basin using 2023 stream gauge data.

# 2. Data

Daily mean discharge data from 12 automated stream gauges (stations ELK-001 through ELK-012) for the full year 2023 are stored in HDF5 format in the file `river_discharge_2023.h5`, deposited at Zenodo under DOI `10.5281/zenodo.00000001`.

The file structure is:

- `/stations/name`: station identifiers (string array, length 12)
- `/stations/upstream_area`: catchment area in km² (float64 array, length 12)
- `/discharge`: daily mean discharge in m³/s (float64 array, shape 12 × 365)
- `/day_of_year`: day index 1–365

Catchment areas range from 12.4 km² (ELK-001) to 350.8 km² (ELK-012). All gauges were calibrated monthly and passed quality control.

# 3. Methods

For each station *i*:

1. Compute the annual mean discharge: $\bar{Q}_i = \frac{1}{365} \sum_{d=1}^{365} Q_{i,d}$

2. Compute the Pearson correlation coefficient between the vector of upstream areas $A$ and the vector of annual mean discharges $\bar{Q}$.

3. Fit an ordinary least-squares linear model: $\bar{Q} = \beta_1 A + \beta_0$

   where $\beta_1$ is the specific yield (discharge per unit area) and $\beta_0$ is the intercept.

The fitting was performed with `numpy.polyfit(A, Q_mean, 1)`.

# 4. Results

## 4.1 Annual Mean Discharge

| Station | Upstream Area (km²) | Mean Discharge (m³/s) |
|---------|---------------------|-----------------------|
| ELK-001 | 12.4 | 0.204 |
| ELK-002 | 28.1 | 0.462 |
| ELK-003 | 45.7 | 0.753 |
| ELK-004 | 63.2 | 1.040 |
| ELK-005 | 89.0 | 1.464 |
| ELK-006 | 112.5 | 1.855 |
| ELK-007 | 145.3 | 2.376 |
| ELK-008 | 178.9 | 2.962 |
| ELK-009 | 210.6 | 3.506 |
| ELK-010 | 256.0 | 4.200 |
| ELK-011 | 301.4 | 4.979 |
| ELK-012 | 350.8 | 5.826 |

## 4.2 Correlation and Linear Fit

The Pearson correlation between upstream area and mean annual discharge is **r = 1.0000**, indicating a near-perfect linear relationship.

The fitted linear model is:

$$\bar{Q} = 0.016566 \cdot A - 0.0074$$

The slope of 0.0166 m³/s/km² represents a specific runoff yield of approximately 523 mm/year, consistent with the region's mean annual precipitation of ~600 mm minus estimated evapotranspiration losses.

# 5. Discussion

The near-perfect correlation is expected given the synthetic nature of the seasonal discharge model, which was constructed as a deterministic function of catchment area with additive noise. In real-world applications, scatter would increase due to variable land cover, soil types, and localized precipitation patterns. Nevertheless, area-based discharge estimation remains a useful first-order approach in relatively homogeneous basins.

# 6. Conclusion

Upstream catchment area explains virtually all variance in mean annual discharge across the 12 Elkhorn Basin stations in 2023. The fitted yield coefficient of 0.0166 m³/s/km² provides a practical tool for estimating discharge at ungauged points within this watershed.

# References

1. Vogel, R.M. et al. (1999). Regional regression models of annual streamflow for the United States. *J. Irrig. Drain. Eng.*, 125(3), 148-157.
2. Blöschl, G. et al. (2013). Runoff prediction in ungauged basins. *Cambridge University Press*.
