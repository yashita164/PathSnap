# PathSnap GPS Trajectory EDA

This report analyzes GPS `.track` observations only. Road-network files and route labels were not read or used.
Raw observations were preserved; invalid intervals and outliers were reported rather than removed.

## 1. Dataset overview

| Split | Trip directories | Available `.track` files | Missing | Empty | GPS observations | Min/trip | Max/trip | Mean/trip | Median/trip | Std/trip | Q1 | Q3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 80 | 80 | 0 | 0 | 208,417 | 361 | 19,313 | 2,605.21 | 1,529.00 | 3,312.35 | 945.75 | 2,792.75 |
| student_test | 10 | 10 | 0 | 0 | 19,160 | 262 | 3,958 | 1,916.00 | 1,420.00 | 1,335.43 | 917.75 | 2,872.00 |

Observation-count statistics use every trip with a known row count; missing and empty tracks have count zero, and malformed files are flagged.

- **train minimum/maximum trips:** 00000026 (361 points) / 00000054 (19,313 points).
- **student_test minimum/maximum trips:** 00000052 (262 points) / 00000056 (3,958 points).
Across the dataset, the shortest observed trip is `00000052` (student_test, 262 points) and the longest is `00000054` (train, 19,313 points).

## 2. GPS observation statistics

- **train:** 208,417 fixes; 361–19,313 fixes per trip (mean 2,605.21, median 1,529.00).
- **student_test:** 19,160 fixes; 262–3,958 fixes per trip (mean 1,916.00, median 1,420.00).
- Trip-count outliers (1.5×IQR within split): `00000002` (train, long, 7,243 points), `00000021` (train, long, 15,783 points), `00000033` (train, long, 5,844 points), `00000042` (train, long, 7,739 points), `00000054` (train, long, 19,313 points), `00000084` (train, long, 16,031 points), `00000092` (train, long, 7,892 points), `00000098` (train, long, 5,584 points).

## 3. Timestamp findings

Dataset timestamps are analyzed as numeric seconds. Interval statistics include every finite consecutive difference, including zero and negative intervals; no gaps were removed.
- **train:** 208,337 finite intervals; range 1.00–153,064.00 s; mean 2.18 s; median 1.00 s; one-second intervals (within 1 μs) 201,053; zero 0; negative 0; non-monotonic trips 0.
- **student_test:** 19,150 finite intervals; range 1.00–210.00 s; mean 1.06 s; median 1.00 s; one-second intervals (within 1 μs) 18,886; zero 0; negative 0; non-monotonic trips 0.

- Intervals greater than 5 seconds: **1,260**.
- Intervals greater than 10 seconds: **617**.
- Intervals greater than 30 seconds: **135**.
- Intervals greater than 60 seconds: **70**.

Trips with the largest individual timestamp gaps:
- `00000090` (train): 153,064 s maximum gap.
- `00000078` (train): 52,810 s maximum gap.
- `00000069` (train): 2,658 s maximum gap.
- `00000021` (train): 2,161 s maximum gap.
- `00000011` (train): 909 s maximum gap.

## 4. Speed findings

Distance is WGS84 geodesic distance in metres (`src.geo.calculate_distance`). Speed is distance divided by positive elapsed seconds and is reported in m/s and km/h. Pairs with invalid coordinates/timestamps or non-positive elapsed time remain in the interval table with unavailable speed.

The unusually-high-speed flag threshold is **200.00 km/h** (maximum of a 200 km/h floor and pooled Q3 + 3×IQR = 197.33 km/h). This is a review threshold, not an automatic outlier filter.

- **train:** 208,337 valid speed calculations; mean 13.71 m/s; median 11.08 m/s; maximum 650.14 km/h; invalid speed calculations 0; flagged high-speed transitions 924.
- **student_test:** 19,150 valid speed calculations; mean 15.42 m/s; median 14.15 m/s; maximum 262.75 km/h; invalid speed calculations 0; flagged high-speed transitions 4.

Trips with the most transitions above the review threshold:
- `00000032` (train): 615 transitions; maximum flagged speed 300.34 km/h.
- `00000001` (train): 150 transitions; maximum flagged speed 260.80 km/h.
- `00000013` (train): 105 transitions; maximum flagged speed 510.75 km/h.
- `00000024` (train): 11 transitions; maximum flagged speed 236.80 km/h.
- `00000043` (train): 7 transitions; maximum flagged speed 474.31 km/h.
- `00000092` (train): 4 transitions; maximum flagged speed 306.96 km/h.
- `00000028` (train): 4 transitions; maximum flagged speed 243.87 km/h.
- `00000055` (train): 4 transitions; maximum flagged speed 236.02 km/h.
- `00000079` (train): 3 transitions; maximum flagged speed 650.14 km/h.
- `00000033` (train): 3 transitions; maximum flagged speed 511.65 km/h.

## 5. Coordinate findings

Across valid coordinate pairs, longitude spans -100.42° to 137.24° and latitude spans -34.64° to 61.70°. Mean coordinate: (15.46°, 41.47°); median coordinate: (12.44°, 49.56°).
Missing longitude values: 0; missing latitude values: 0; invalid coordinate observations: 0.
The overall bounding envelope is approximately 19,853.50 km × 10,678.14 km (211,998,400.46 km²). These are geodesic dimensions of a bounding rectangle, not route distances; the dataset is geographically dispersed.

## 6. GPS trajectory maps

Maps use only `.track` coordinates, ordered by timestamp (stable observation order for ties). Invalid-coordinate rows are omitted from map rendering only and remain in the analysis tables.
- short: `00000052` (student_test, 262 fixes, 262 mapped points) — `reports/maps/trip_00000052.html`.
- medium: `00000083` (train, 1,507 fixes, 1,507 mapped points) — `reports/maps/trip_00000083.html`.
- long: `00000054` (train, 19,313 fixes, 19,313 mapped points) — `reports/maps/trip_00000054.html`.
- student-test medium: `00000076` (student_test, 1,486 fixes, 1,486 mapped points) — `reports/maps/trip_00000076.html`.

## 7. Data-quality findings

| Issue type | Count |
|---|---:|
| timestamp_gaps_over_10_seconds | 617 |
| timestamp_gaps_over_30_seconds | 135 |
| timestamp_gaps_over_5_seconds | 1,260 |
| timestamp_gaps_over_60_seconds | 70 |
| unusually_high_speed | 928 |

The trip-level details and descriptions are available in the notebook's `quality_issues` table.

## 8. Unusual trips and investigation priorities

Calculated findings:
- `00000090` (train) has the largest observed finite interval (153,064 s).
- The highest valid point-to-point speed is 650.14 km/h on `00000079` (train, transition 90).
- No missing/empty/malformed files, invalid coordinates, invalid timestamp values, or unusable speed transitions were detected.

## 9. Phase 3 considerations — recommendations and assumptions

These are recommendations, not calculated dataset facts:
- Preserve observation order and raw timestamps; carry interval validity and gap indicators into downstream processing rather than bridging invalid or very long gaps silently.
- Consider deriving speed and heading only for positive, finite time intervals and valid coordinates; retain explicit missingness flags for unavailable values.
- Review flagged high-speed transitions and long-gap trips before using them to calibrate any GPS uncertainty or search-radius assumptions.
- Candidate generation and feature engineering should use the existing geographic utilities and a suitable local metric projection; no candidates, labels, or model features were generated in this EDA.

## Limitations

- Timestamp values were treated as elapsed seconds based on the observed numeric `.track` format; no timezone or wall-clock interpretation was inferred.
- Geodesic bounding-box dimensions summarize geographic coverage only and do not measure route length.
- Folium maps use OpenStreetMap tiles, which require an internet connection when viewed; their HTML structure and saved paths were verified locally.
