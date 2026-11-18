"""GPS-only exploratory analysis helpers for PathSnap."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.geo import calculate_distance
from src.io import load_track


SPLITS = ("train", "student_test")
GAP_THRESHOLDS_SECONDS = (5, 10, 30, 60)
HIGH_SPEED_FLOOR_KMH = 200.0
TRACK_COLUMNS = ("longitude", "latitude", "timestamp")


def _append_issue(
    issues: list[dict[str, Any]],
    split: str,
    trip_id: str,
    issue_type: str,
    count: int,
    description: str,
) -> None:
    if count > 0:
        issues.append(
            {
                "dataset_split": split,
                "trip_id": trip_id,
                "issue_type": issue_type,
                "count": int(count),
                "description": description,
            }
        )


def _read_malformed_track(path: Path) -> pd.DataFrame:
    """Keep one row per nonblank raw observation when the standard loader rejects it."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as track_file:
        for line_number, line in enumerate(track_file, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            fields = raw_line.split()
            values = fields[: len(TRACK_COLUMNS)]
            values.extend([None] * (len(TRACK_COLUMNS) - len(values)))
            rows.append(
                {
                    "longitude": values[0],
                    "latitude": values[1],
                    "timestamp": values[2],
                    "parse_error": len(fields) != len(TRACK_COLUMNS),
                    "raw_line": raw_line,
                    "source_line": line_number,
                }
            )
    return pd.DataFrame(rows)


def _as_observations(
    path: Path, split: str, trip_id: str
) -> tuple[pd.DataFrame, str, str | None]:
    try:
        observations = load_track(path)
        observations = observations.assign(
            parse_error=False,
            raw_line=None,
            source_line=np.arange(1, len(observations) + 1),
        )
        return observations, "valid", None
    except (OSError, UnicodeError, ValueError) as error:
        try:
            observations = _read_malformed_track(path)
        except (OSError, UnicodeError) as read_error:
            return pd.DataFrame(columns=TRACK_COLUMNS), "unreadable", str(read_error)
        if observations.empty:
            return observations, "empty", str(error)
        observations["longitude"] = pd.to_numeric(observations["longitude"], errors="coerce")
        observations["latitude"] = pd.to_numeric(observations["latitude"], errors="coerce")
        observations["timestamp"] = pd.to_numeric(observations["timestamp"], errors="coerce")
        return observations, "malformed", str(error)


def _coordinate_validity(observations: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    longitude = observations["longitude"].to_numpy(dtype=np.float64, na_value=np.nan)
    latitude = observations["latitude"].to_numpy(dtype=np.float64, na_value=np.nan)
    finite_longitude = np.isfinite(longitude)
    finite_latitude = np.isfinite(latitude)
    valid_longitude = finite_longitude & (longitude >= -180) & (longitude <= 180)
    valid_latitude = finite_latitude & (latitude >= -90) & (latitude <= 90)
    valid_coordinates = valid_longitude & valid_latitude
    invalid_coordinates = (
        (observations["longitude"].notna().to_numpy() & ~valid_longitude)
        | (observations["latitude"].notna().to_numpy() & ~valid_latitude)
    )
    return valid_coordinates, invalid_coordinates


def _numeric_series_values(observations: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(observations[column], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )


def _summary_statistics(values: np.ndarray) -> dict[str, float | int]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return {
            "count": 0,
            "minimum": np.nan,
            "maximum": np.nan,
            "mean": np.nan,
            "median": np.nan,
        }
    return {
        "count": int(finite_values.size),
        "minimum": float(np.min(finite_values)),
        "maximum": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "median": float(np.median(finite_values)),
    }


def _coordinate_summary(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("all", observations)]
    groups.extend(
        (split, observations.loc[observations["dataset_split"] == split])
        for split in SPLITS
    )

    for group_name, group in groups:
        valid_coordinates, invalid_coordinates = _coordinate_validity(group)
        longitude = _numeric_series_values(group, "longitude")
        latitude = _numeric_series_values(group, "latitude")
        valid_longitude = longitude[valid_coordinates]
        valid_latitude = latitude[valid_coordinates]
        summary_rows.append(
            {
                "dataset_split": group_name,
                "valid_coordinate_count": int(valid_coordinates.sum()),
                "missing_longitude_values": int(group["longitude"].isna().sum()),
                "missing_latitude_values": int(group["latitude"].isna().sum()),
                "invalid_coordinate_observations": int(invalid_coordinates.sum()),
                "longitude_min": float(np.min(valid_longitude)) if valid_longitude.size else np.nan,
                "longitude_max": float(np.max(valid_longitude)) if valid_longitude.size else np.nan,
                "longitude_mean": float(np.mean(valid_longitude)) if valid_longitude.size else np.nan,
                "longitude_median": float(np.median(valid_longitude)) if valid_longitude.size else np.nan,
                "latitude_min": float(np.min(valid_latitude)) if valid_latitude.size else np.nan,
                "latitude_max": float(np.max(valid_latitude)) if valid_latitude.size else np.nan,
                "latitude_mean": float(np.mean(valid_latitude)) if valid_latitude.size else np.nan,
                "latitude_median": float(np.median(valid_latitude)) if valid_latitude.size else np.nan,
            }
        )
        if valid_coordinates.any():
            west = float(np.min(valid_longitude))
            east = float(np.max(valid_longitude))
            south = float(np.min(valid_latitude))
            north = float(np.max(valid_latitude))
            center_longitude = float(np.mean(valid_longitude))
            center_latitude = float(np.mean(valid_latitude))
            longitude_sample_count = max(1, int(np.ceil(east - west)))
            longitude_samples = np.linspace(
                west, east, longitude_sample_count + 1, dtype=np.float64
            )
            width_km = float(
                np.sum(
                    calculate_distance(
                        longitude_samples[:-1],
                        np.full(longitude_sample_count, center_latitude),
                        longitude_samples[1:],
                        np.full(longitude_sample_count, center_latitude),
                    )
                )
            ) / 1000.0
            height_km = float(
                calculate_distance(center_longitude, south, center_longitude, north)
            ) / 1000.0
            coverage_rows.append(
                {
                    "dataset_split": group_name,
                    "west_longitude": west,
                    "east_longitude": east,
                    "south_latitude": south,
                    "north_latitude": north,
                    "approximate_width_km": width_km,
                    "approximate_height_km": height_km,
                    "approximate_bbox_area_km2": width_km * height_km,
                }
            )
        else:
            coverage_rows.append({"dataset_split": group_name})
    return pd.DataFrame(summary_rows), pd.DataFrame(coverage_rows)


def _count_summary(inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_rows: list[dict[str, Any]] = []
    outlier_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        group = inventory.loc[inventory["dataset_split"] == split]
        counts = group["gps_point_count"].dropna().to_numpy(dtype=np.float64)
        known_trips = group.loc[group["gps_point_count"].notna()]
        available_files = int(group["track_file_available"].sum())
        stats = _summary_statistics(counts)
        minimum_trip_ids = (
            known_trips.loc[
                known_trips["gps_point_count"] == stats["minimum"], "trip_id"
            ].astype(str).tolist()
            if counts.size
            else []
        )
        maximum_trip_ids = (
            known_trips.loc[
                known_trips["gps_point_count"] == stats["maximum"], "trip_id"
            ].astype(str).tolist()
            if counts.size
            else []
        )
        split_rows.append(
            {
                "dataset_split": split,
                "trip_directory_count": int(len(group)),
                "available_track_files": available_files,
                "missing_track_files": int((group["track_status"] == "missing").sum()),
                "empty_track_files": int((group["track_status"] == "empty").sum()),
                "malformed_track_files": int(
                    group["track_status"].isin(("malformed", "unreadable")).sum()
                ),
                "total_gps_observations": int(np.nansum(counts)),
                "trips_with_known_point_count": stats["count"],
                "minimum_observations_per_trip": stats["minimum"],
                "maximum_observations_per_trip": stats["maximum"],
                "mean_observations_per_trip": stats["mean"],
                "median_observations_per_trip": stats["median"],
                "standard_deviation_observations_per_trip": (
                    float(np.std(counts, ddof=1)) if counts.size > 1 else np.nan
                ),
                "first_quartile_observations_per_trip": (
                    float(np.quantile(counts, 0.25)) if counts.size else np.nan
                ),
                "third_quartile_observations_per_trip": (
                    float(np.quantile(counts, 0.75)) if counts.size else np.nan
                ),
                "minimum_observation_trip_ids": ", ".join(minimum_trip_ids),
                "maximum_observation_trip_ids": ", ".join(maximum_trip_ids),
            }
        )
        if counts.size == 0:
            continue
        first_quartile, third_quartile = np.quantile(counts, [0.25, 0.75])
        interquartile_range = third_quartile - first_quartile
        lower_fence = first_quartile - 1.5 * interquartile_range
        upper_fence = third_quartile + 1.5 * interquartile_range
        outliers = group.loc[
            group["gps_point_count"].notna()
            & (
                (group["gps_point_count"] < lower_fence)
                | (group["gps_point_count"] > upper_fence)
            )
        ]
        for row in outliers.itertuples(index=False):
            outlier_rows.append(
                {
                    "trip_id": row.trip_id,
                    "dataset_split": split,
                    "gps_point_count": int(row.gps_point_count),
                    "outlier_direction": (
                        "short" if row.gps_point_count < lower_fence else "long"
                    ),
                    "lower_fence": float(lower_fence),
                    "upper_fence": float(upper_fence),
                }
            )
    return pd.DataFrame(split_rows), pd.DataFrame(outlier_rows)


def analyze_gps_dataset(data_root: str | Path) -> dict[str, Any]:
    """Analyze only ``.track`` files in the training and student-test splits.

    Original files are read but never modified. Invalid or malformed observations
    remain represented in the returned observation table when their raw rows can
    be recovered; calculations that require valid values are marked unavailable.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"GPS dataset directory was not found: {root}")

    inventory_rows: list[dict[str, Any]] = []
    observation_frames: list[pd.DataFrame] = []
    interval_frames: list[pd.DataFrame] = []
    trip_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for split in SPLITS:
        split_root = root / split
        if not split_root.is_dir():
            _append_issue(
                issues,
                split,
                split,
                "missing_dataset_split",
                1,
                f"Expected dataset split directory does not exist: {split_root}",
            )
            continue

        trip_directories = sorted(
            (path for path in split_root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
        for trip_directory in trip_directories:
            trip_id = trip_directory.name
            track_path = trip_directory / f"{trip_id}.track"
            row: dict[str, Any] = {
                "trip_id": trip_id,
                "dataset_split": split,
                "gps_point_count": np.nan,
                "track_status": "missing",
                "track_file_available": track_path.is_file(),
                "track_path": str(track_path),
            }
            if not track_path.is_file():
                row["gps_point_count"] = 0
                _append_issue(
                    issues,
                    split,
                    trip_id,
                    "missing_track_file",
                    1,
                    "Expected .track file is missing.",
                )
                inventory_rows.append(row)
                trip_rows.append(
                    {
                        "trip_id": trip_id,
                        "dataset_split": split,
                        "gps_point_count": 0,
                        "track_status": "missing",
                    }
                )
                continue

            if track_path.stat().st_size == 0:
                observations = pd.DataFrame(columns=TRACK_COLUMNS)
                status, error_message = "empty", "File has zero bytes."
            else:
                observations, status, error_message = _as_observations(
                    track_path, split, trip_id
                )
            if status == "empty":
                row["gps_point_count"] = 0
                _append_issue(
                    issues,
                    split,
                    trip_id,
                    "empty_track_file",
                    1,
                    "Track file contains no nonblank observation rows.",
                )
            elif status == "unreadable":
                _append_issue(
                    issues,
                    split,
                    trip_id,
                    "unreadable_track_file",
                    1,
                    error_message or "Track file could not be read.",
                )
            else:
                row["gps_point_count"] = len(observations)

            row["track_status"] = status
            if status == "malformed":
                _append_issue(
                    issues,
                    split,
                    trip_id,
                    "malformed_track_file",
                    1,
                    error_message or "Standard track validation failed.",
                )
                malformed_rows = int(observations["parse_error"].sum())
                _append_issue(
                    issues,
                    split,
                    trip_id,
                    "malformed_track_rows",
                    malformed_rows,
                    "Observation rows do not contain exactly three whitespace-separated fields.",
                )

            if not observations.empty:
                observations = observations.copy()
                for column in TRACK_COLUMNS:
                    observations[column] = pd.to_numeric(
                        observations[column], errors="coerce"
                    )
                observations.insert(0, "dataset_split", split)
                observations.insert(1, "trip_id", trip_id)
                observations.insert(
                    2, "observation_index", np.arange(len(observations), dtype=np.int64)
                )
                observation_frames.append(observations)
            inventory_rows.append(row)

            trip_rows.append(
                _analyze_trip_observations(
                    observations,
                    split,
                    trip_id,
                    status,
                    issues,
                    interval_frames,
                )
            )

    inventory = pd.DataFrame(
        inventory_rows,
        columns=(
            "trip_id",
            "dataset_split",
            "gps_point_count",
            "track_status",
            "track_file_available",
            "track_path",
        ),
    )
    observations = (
        pd.concat(observation_frames, ignore_index=True)
        if observation_frames
        else pd.DataFrame(
            columns=(
                "dataset_split",
                "trip_id",
                "observation_index",
                *TRACK_COLUMNS,
                "parse_error",
                "raw_line",
                "source_line",
            )
        )
    )
    intervals = (
        pd.concat(interval_frames, ignore_index=True)
        if interval_frames
        else pd.DataFrame()
    )
    trip_summary = pd.DataFrame(trip_rows)
    if intervals.empty:
        intervals = pd.DataFrame(
            columns=(
                "dataset_split",
                "trip_id",
                "transition_index",
                "from_observation_index",
                "to_observation_index",
                "start_timestamp",
                "end_timestamp",
                "elapsed_seconds",
                "distance_m",
                "speed_mps",
                "speed_kmh",
                "speed_invalid_reason",
            )
        )

    valid_speeds = intervals["speed_mps"].dropna().to_numpy(dtype=np.float64)
    if valid_speeds.size:
        third_quartile = float(np.quantile(valid_speeds, 0.75))
        interquartile_range = float(
            np.quantile(valid_speeds, 0.75) - np.quantile(valid_speeds, 0.25)
        )
        robust_speed_threshold_mps = third_quartile + 3.0 * interquartile_range
    else:
        robust_speed_threshold_mps = 0.0
    high_speed_threshold_mps = max(
        HIGH_SPEED_FLOOR_KMH / 3.6, robust_speed_threshold_mps
    )
    intervals["is_unusually_high_speed"] = (
        intervals["speed_mps"] > high_speed_threshold_mps
    )

    high_speed_counts = (
        intervals.loc[intervals["is_unusually_high_speed"]]
        .groupby(["dataset_split", "trip_id"])
        .size()
        .to_dict()
    )
    maximum_speeds = intervals.groupby(["dataset_split", "trip_id"])[
        "speed_kmh"
    ].max().to_dict()
    for row in trip_rows:
        key = (row["dataset_split"], row["trip_id"])
        row["unusually_high_speed_interval_count"] = int(high_speed_counts.get(key, 0))
        row["maximum_speed_kmh"] = maximum_speeds.get(key, np.nan)
        _append_issue(
            issues,
            row["dataset_split"],
            row["trip_id"],
            "unusually_high_speed",
            row["unusually_high_speed_interval_count"],
            f"Speed exceeds the flagged threshold of {high_speed_threshold_mps * 3.6:.2f} km/h.",
        )

    trip_summary = pd.DataFrame(trip_rows)
    split_summary, trip_count_outliers = _count_summary(inventory)
    coordinate_summary, geographic_coverage = _coordinate_summary(observations)
    quality_issues = pd.DataFrame(
        issues,
        columns=("dataset_split", "trip_id", "issue_type", "count", "description"),
    )
    timestamp_columns = [
        "dataset_split",
        "trip_id",
        "gps_point_count",
        "timestamp_value_count",
        "timestamp_invalid_value_count",
        "timestamp_monotonic",
        "duration_seconds",
        "minimum_interval_seconds",
        "maximum_interval_seconds",
        "mean_interval_seconds",
        "median_interval_seconds",
        "one_second_interval_count",
        "zero_interval_count",
        "negative_interval_count",
        *(f"gap_over_{threshold}_seconds_count" for threshold in GAP_THRESHOLDS_SECONDS),
    ]
    speed_columns = [
        "dataset_split",
        "trip_id",
        "gps_point_count",
        "valid_speed_interval_count",
        "invalid_speed_calculation_count",
        "mean_speed_mps",
        "median_speed_mps",
        "maximum_speed_mps",
        "mean_speed_kmh",
        "median_speed_kmh",
        "maximum_speed_kmh",
        "unusually_high_speed_interval_count",
    ]
    return {
        "inventory": inventory,
        "observations": observations,
        "intervals": intervals,
        "trip_summary": trip_summary,
        "split_summary": split_summary,
        "trip_count_outliers": trip_count_outliers,
        "timestamp_summary": trip_summary.reindex(columns=timestamp_columns),
        "speed_summary": trip_summary.reindex(columns=speed_columns),
        "coordinate_summary": coordinate_summary,
        "geographic_coverage": geographic_coverage,
        "quality_issues": quality_issues,
        "high_speed_trips": _high_speed_trip_summary(intervals),
        "high_speed_transitions": intervals.loc[
            intervals["is_unusually_high_speed"]
        ].sort_values("speed_kmh", ascending=False),
        "high_speed_threshold_mps": high_speed_threshold_mps,
        "robust_speed_threshold_mps": robust_speed_threshold_mps,
    }


def _analyze_trip_observations(
    observations: pd.DataFrame,
    split: str,
    trip_id: str,
    status: str,
    issues: list[dict[str, Any]],
    interval_frames: list[pd.DataFrame],
) -> dict[str, Any]:
    point_count = len(observations)
    longitude = _numeric_series_values(observations, "longitude")
    latitude = _numeric_series_values(observations, "latitude")
    timestamps = _numeric_series_values(observations, "timestamp")
    valid_coordinates, invalid_coordinates = _coordinate_validity(observations)
    finite_timestamps = np.isfinite(timestamps)
    timestamp_differences = np.full(max(point_count - 1, 0), np.nan, dtype=np.float64)
    valid_timestamp_pairs = finite_timestamps[:-1] & finite_timestamps[1:]
    timestamp_differences[valid_timestamp_pairs] = (
        timestamps[1:][valid_timestamp_pairs] - timestamps[:-1][valid_timestamp_pairs]
    )
    finite_intervals = timestamp_differences[np.isfinite(timestamp_differences)]
    zero_interval_count = int((finite_intervals == 0).sum())
    negative_interval_count = int((finite_intervals < 0).sum())
    invalid_timestamp_value_count = int(
        (observations["timestamp"].notna().to_numpy() & ~finite_timestamps).sum()
    )
    missing_value_count = int(observations[list(TRACK_COLUMNS)].isna().sum().sum())
    duplicate_observation_count = int(
        observations.loc[valid_coordinates & finite_timestamps].duplicated(
            list(TRACK_COLUMNS)
        ).sum()
    )
    duplicate_timestamp_count = int(
        pd.Series(timestamps[finite_timestamps]).duplicated().sum()
    )

    _append_issue(
        issues,
        split,
        trip_id,
        "missing_values",
        missing_value_count,
        "Missing or non-numeric longitude, latitude, or timestamp fields.",
    )
    _append_issue(
        issues,
        split,
        trip_id,
        "invalid_coordinates",
        int(invalid_coordinates.sum()),
        "Coordinate observations contain non-finite values or values outside longitude/latitude bounds.",
    )
    _append_issue(
        issues,
        split,
        trip_id,
        "duplicate_gps_observations",
        duplicate_observation_count,
        "Repeated complete longitude, latitude, and timestamp observations.",
    )
    _append_issue(
        issues,
        split,
        trip_id,
        "duplicate_timestamps",
        duplicate_timestamp_count,
        "Timestamp values repeated within the trip.",
    )
    _append_issue(
        issues,
        split,
        trip_id,
        "invalid_timestamp_values",
        invalid_timestamp_value_count,
        "Non-finite timestamp values.",
    )
    _append_issue(
        issues,
        split,
        trip_id,
        "zero_time_intervals",
        zero_interval_count,
        "Consecutive observations have identical timestamps.",
    )
    _append_issue(
        issues,
        split,
        trip_id,
        "negative_time_intervals",
        negative_interval_count,
        "A consecutive timestamp is earlier than the preceding timestamp.",
    )
    _append_issue(
        issues,
        split,
        trip_id,
        "non_monotonic_timestamps",
        negative_interval_count,
        "Count of consecutive timestamp intervals that decrease.",
    )
    for threshold in GAP_THRESHOLDS_SECONDS:
        gap_count = int((finite_intervals > threshold).sum())
        _append_issue(
            issues,
            split,
            trip_id,
            f"timestamp_gaps_over_{threshold}_seconds",
            gap_count,
            f"Consecutive timestamp intervals greater than {threshold} seconds.",
        )

    duration = (
        float(timestamps[-1] - timestamps[0])
        if point_count and finite_timestamps[0] and finite_timestamps[-1]
        else np.nan
    )
    interval_stats = _summary_statistics(finite_intervals)
    time_deltas = timestamp_differences
    transition_count = max(point_count - 1, 0)
    valid_coordinate_pairs = valid_coordinates[:-1] & valid_coordinates[1:]
    distances = np.full(transition_count, np.nan, dtype=np.float64)
    if valid_coordinate_pairs.any():
        distances[valid_coordinate_pairs] = calculate_distance(
            longitude[:-1][valid_coordinate_pairs],
            latitude[:-1][valid_coordinate_pairs],
            longitude[1:][valid_coordinate_pairs],
            latitude[1:][valid_coordinate_pairs],
        )
    speed_valid = (
        valid_coordinate_pairs
        & np.isfinite(time_deltas)
        & (time_deltas > 0)
        & np.isfinite(distances)
    )
    speeds_mps = np.full(transition_count, np.nan, dtype=np.float64)
    speeds_mps[speed_valid] = distances[speed_valid] / time_deltas[speed_valid]
    invalid_speed_count = int((~speed_valid).sum())

    if transition_count:
        reason = np.full(transition_count, "", dtype=object)
        reason[~valid_coordinate_pairs] = "invalid endpoint coordinates"
        invalid_times = ~np.isfinite(time_deltas)
        reason[invalid_times] = np.where(
            reason[invalid_times] == "",
            "missing or invalid timestamp",
            reason[invalid_times] + "; missing or invalid timestamp",
        )
        nonpositive_times = np.isfinite(time_deltas) & (time_deltas <= 0)
        reason[nonpositive_times] = np.where(
            reason[nonpositive_times] == "",
            "non-positive elapsed time",
            reason[nonpositive_times] + "; non-positive elapsed time",
        )
        interval_frame = pd.DataFrame(
            {
                "dataset_split": split,
                "trip_id": trip_id,
                "transition_index": np.arange(1, point_count, dtype=np.int64),
                "from_observation_index": np.arange(point_count - 1, dtype=np.int64),
                "to_observation_index": np.arange(1, point_count, dtype=np.int64),
                "start_timestamp": timestamps[:-1],
                "end_timestamp": timestamps[1:],
                "elapsed_seconds": time_deltas,
                "distance_m": distances,
                "speed_mps": speeds_mps,
                "speed_kmh": speeds_mps * 3.6,
                "speed_invalid_reason": reason,
            }
        )
        interval_frames.append(interval_frame)

    _append_issue(
        issues,
        split,
        trip_id,
        "invalid_speed_calculations",
        invalid_speed_count,
        "Transitions without finite endpoint coordinates and a positive elapsed time cannot yield speed.",
    )

    speed_stats = _summary_statistics(speeds_mps)
    row: dict[str, Any] = {
        "trip_id": trip_id,
        "dataset_split": split,
        "gps_point_count": point_count,
        "track_status": status,
        "missing_value_count": missing_value_count,
        "invalid_coordinate_count": int(invalid_coordinates.sum()),
        "duplicate_gps_observation_count": duplicate_observation_count,
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "timestamp_value_count": int(finite_timestamps.sum()),
        "timestamp_invalid_value_count": invalid_timestamp_value_count,
        "timestamp_monotonic": bool(
            finite_timestamps.all() and negative_interval_count == 0
        ),
        "duration_seconds": duration,
        "interval_count": int(finite_intervals.size),
        "minimum_interval_seconds": interval_stats["minimum"],
        "maximum_interval_seconds": interval_stats["maximum"],
        "mean_interval_seconds": interval_stats["mean"],
        "median_interval_seconds": interval_stats["median"],
        "one_second_interval_count": int(
            np.isclose(finite_intervals, 1.0, rtol=0.0, atol=1e-6).sum()
        ),
        "zero_interval_count": zero_interval_count,
        "negative_interval_count": negative_interval_count,
        "valid_speed_interval_count": speed_stats["count"],
        "invalid_speed_calculation_count": invalid_speed_count,
        "mean_speed_mps": speed_stats["mean"],
        "median_speed_mps": speed_stats["median"],
        "maximum_speed_mps": speed_stats["maximum"],
        "mean_speed_kmh": (
            float(speed_stats["mean"]) * 3.6
            if np.isfinite(speed_stats["mean"])
            else np.nan
        ),
        "median_speed_kmh": (
            float(speed_stats["median"]) * 3.6
            if np.isfinite(speed_stats["median"])
            else np.nan
        ),
        "maximum_speed_kmh": (
            float(speed_stats["maximum"]) * 3.6
            if np.isfinite(speed_stats["maximum"])
            else np.nan
        ),
    }
    for threshold in GAP_THRESHOLDS_SECONDS:
        row[f"gap_over_{threshold}_seconds_count"] = int(
            (finite_intervals > threshold).sum()
        )
    return row


def _high_speed_trip_summary(intervals: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "dataset_split",
        "trip_id",
        "unusually_high_speed_interval_count",
        "maximum_flagged_speed_kmh",
    )
    if intervals.empty:
        return pd.DataFrame(columns=columns)
    flagged = intervals.loc[intervals["is_unusually_high_speed"]]
    if flagged.empty:
        return pd.DataFrame(columns=columns)
    return (
        flagged.groupby(["dataset_split", "trip_id"], as_index=False)
        .agg(
            unusually_high_speed_interval_count=("speed_mps", "size"),
            maximum_flagged_speed_kmh=("speed_kmh", "max"),
        )
        .sort_values(
            ["unusually_high_speed_interval_count", "maximum_flagged_speed_kmh"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def create_representative_maps(
    analysis: dict[str, Any], output_directory: str | Path
) -> pd.DataFrame:
    """Save short, medium, and long GPS-only Folium maps in timestamp order."""
    import folium

    inventory = analysis["inventory"]
    eligible = inventory.loc[
        inventory["gps_point_count"].fillna(0).gt(0)
        & inventory["track_status"].isin(("valid", "malformed"))
    ].sort_values(["gps_point_count", "dataset_split", "trip_id"])
    if eligible.empty:
        return pd.DataFrame(
            columns=(
                "selection",
                "dataset_split",
                "trip_id",
                "gps_point_count",
                "map_path",
                "mapped_point_count",
            )
        )

    selected: list[tuple[str, pd.Series]] = [
        ("short", eligible.iloc[0]),
        ("medium", eligible.iloc[len(eligible) // 2]),
        ("long", eligible.iloc[-1]),
    ]
    student_test = eligible.loc[eligible["dataset_split"] == "student_test"]
    if not student_test.empty:
        student_median = student_test.iloc[len(student_test) // 2]
        if (student_median["dataset_split"], student_median["trip_id"]) not in {
            (row["dataset_split"], row["trip_id"]) for _, row in selected
        }:
            selected.append(("student-test medium", student_median))

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    observations = analysis["observations"]
    map_rows: list[dict[str, Any]] = []
    colors = ("blue", "green", "red", "purple", "orange")
    for index, (selection, trip) in enumerate(selected):
        trip_id = str(trip["trip_id"])
        split = str(trip["dataset_split"])
        trip_observations = observations.loc[
            (observations["dataset_split"] == split)
            & (observations["trip_id"] == trip_id)
        ].copy()
        valid_coordinates, _ = _coordinate_validity(trip_observations)
        trip_observations = trip_observations.loc[valid_coordinates].sort_values(
            ["timestamp", "observation_index"],
            kind="mergesort",
            na_position="last",
        )
        if trip_observations.empty:
            continue

        locations = list(
            zip(
                trip_observations["latitude"].astype(float),
                trip_observations["longitude"].astype(float),
            )
        )
        center = [
            float(trip_observations["latitude"].mean()),
            float(trip_observations["longitude"].mean()),
        ]
        gps_map = folium.Map(location=center, zoom_start=5, control_scale=True)
        color = colors[index % len(colors)]
        folium.PolyLine(
            locations=locations,
            color=color,
            weight=4,
            opacity=0.85,
            tooltip=f"Trip {trip_id} ({split})",
        ).add_to(gps_map)
        start = locations[0]
        end = locations[-1]
        folium.Marker(
            start,
            tooltip=f"Trip {trip_id} start",
            popup=f"Start: {trip_id}",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(gps_map)
        folium.Marker(
            end,
            tooltip=f"Trip {trip_id} end",
            popup=f"End: {trip_id}",
            icon=folium.Icon(color="red", icon="stop"),
        ).add_to(gps_map)
        gps_map.fit_bounds(
            [
                [
                    float(trip_observations["latitude"].min()),
                    float(trip_observations["longitude"].min()),
                ],
                [
                    float(trip_observations["latitude"].max()),
                    float(trip_observations["longitude"].max()),
                ],
            ]
        )
        map_path = destination / f"trip_{trip_id}.html"
        gps_map.save(map_path)
        map_rows.append(
            {
                "selection": selection,
                "dataset_split": split,
                "trip_id": trip_id,
                "gps_point_count": int(trip["gps_point_count"]),
                "map_path": str(map_path),
                "mapped_point_count": len(locations),
            }
        )
    return pd.DataFrame(map_rows)


def _format_number(value: Any, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "not available"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):,.{decimals}f}"


def build_summary_markdown(
    analysis: dict[str, Any], map_records: pd.DataFrame
) -> str:
    """Build the final report with observed results separated from guidance."""
    split_summary = analysis["split_summary"].set_index("dataset_split")
    train = split_summary.loc["train"]
    student = split_summary.loc["student_test"]
    trip_summary = analysis["trip_summary"]
    valid_trips = trip_summary.loc[trip_summary["gps_point_count"].notna()]
    all_intervals = analysis["intervals"]
    valid_intervals = all_intervals.loc[np.isfinite(all_intervals["elapsed_seconds"])]
    valid_speeds = all_intervals["speed_mps"].dropna()
    quality = analysis["quality_issues"]
    coordinate_summary = analysis["coordinate_summary"].set_index("dataset_split")
    coverage = analysis["geographic_coverage"].set_index("dataset_split")
    all_coordinates = coordinate_summary.loc["all"]
    all_coverage = coverage.loc["all"]

    minimum_trip = (
        valid_trips.sort_values("gps_point_count").iloc[0]
        if not valid_trips.empty
        else None
    )
    maximum_trip = (
        valid_trips.sort_values("gps_point_count").iloc[-1]
        if not valid_trips.empty
        else None
    )
    worst_gaps = (
        analysis["timestamp_summary"]
        .sort_values("maximum_interval_seconds", ascending=False)
        .head(5)
    )
    worst_speeds = analysis["high_speed_trips"].head(10)

    lines = [
        "# PathSnap GPS Trajectory EDA",
        "",
        "This report analyzes GPS `.track` observations only. Road-network files and route labels were not read or used.",
        "Raw observations were preserved; invalid intervals and outliers were reported rather than removed.",
        "",
        "## 1. Dataset overview",
        "",
        "| Split | Trip directories | Available `.track` files | Missing | Empty | GPS observations | Min/trip | Max/trip | Mean/trip | Median/trip | Std/trip | Q1 | Q3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["split_summary"].itertuples(index=False):
        lines.append(
            f"| {row.dataset_split} | {row.trip_directory_count} | {row.available_track_files} | "
            f"{row.missing_track_files} | {row.empty_track_files} | {row.total_gps_observations:,} | "
            f"{_format_number(row.minimum_observations_per_trip, 0)} | "
            f"{_format_number(row.maximum_observations_per_trip, 0)} | "
            f"{_format_number(row.mean_observations_per_trip)} | "
            f"{_format_number(row.median_observations_per_trip)} | "
            f"{_format_number(row.standard_deviation_observations_per_trip)} | "
            f"{_format_number(row.first_quartile_observations_per_trip)} | "
            f"{_format_number(row.third_quartile_observations_per_trip)} |"
        )
    lines.extend(
        [
        "",
        "Observation-count statistics use every trip with a known row count; missing and empty tracks have count zero, and malformed files are flagged.",
        "",
        ]
    )
    for row in analysis["split_summary"].itertuples(index=False):
        lines.append(
        f"- **{row.dataset_split} minimum/maximum trips:** "
        f"{row.minimum_observation_trip_ids or 'not available'} "
        f"({_format_number(row.minimum_observations_per_trip, 0)} points) / "
        f"{row.maximum_observation_trip_ids or 'not available'} "
        f"({_format_number(row.maximum_observations_per_trip, 0)} points)."
        )
    if minimum_trip is not None and maximum_trip is not None:
        lines.append(
            f"Across the dataset, the shortest observed trip is `{minimum_trip['trip_id']}` "
            f"({minimum_trip['dataset_split']}, {int(minimum_trip['gps_point_count']):,} points) "
            f"and the longest is `{maximum_trip['trip_id']}` "
            f"({maximum_trip['dataset_split']}, {int(maximum_trip['gps_point_count']):,} points)."
        )
    lines.extend(["", "## 2. GPS observation statistics", ""])
    for row in analysis["split_summary"].itertuples(index=False):
        lines.append(
            f"- **{row.dataset_split}:** {row.total_gps_observations:,} fixes; "
            f"{_format_number(row.minimum_observations_per_trip, 0)}–"
            f"{_format_number(row.maximum_observations_per_trip, 0)} fixes per trip "
            f"(mean {_format_number(row.mean_observations_per_trip)}, "
            f"median {_format_number(row.median_observations_per_trip)})."
        )
    count_outliers = analysis["trip_count_outliers"]
    if count_outliers.empty:
        lines.append("- No trip-count outliers were flagged using the 1.5×IQR rule within each split.")
    else:
        listed = ", ".join(
            f"`{row.trip_id}` ({row.dataset_split}, {row.outlier_direction}, "
            f"{row.gps_point_count:,} points)"
            for row in count_outliers.itertuples(index=False)
        )
        lines.append(f"- Trip-count outliers (1.5×IQR within split): {listed}.")

    lines.extend(
        [
            "",
            "## 3. Timestamp findings",
            "",
            "Dataset timestamps are analyzed as numeric seconds. Interval statistics include every finite consecutive difference, including zero and negative intervals; no gaps were removed.",
        ]
    )
    for split in SPLITS:
        group = valid_intervals.loc[valid_intervals["dataset_split"] == split, "elapsed_seconds"]
        if group.empty:
            continue
        diffs = group.to_numpy(dtype=np.float64)
        trip_group = valid_trips.loc[valid_trips["dataset_split"] == split]
        lines.append(
            f"- **{split}:** {len(diffs):,} finite intervals; range "
            f"{_format_number(np.min(diffs))}–{_format_number(np.max(diffs))} s; "
            f"mean {_format_number(np.mean(diffs))} s; median {_format_number(np.median(diffs))} s; "
            f"one-second intervals (within 1 μs) "
            f"{int(np.isclose(diffs, 1.0, rtol=0.0, atol=1e-6).sum()):,}; "
            f"zero {int((diffs == 0).sum())}; negative {int((diffs < 0).sum())}; "
            f"non-monotonic trips {int((~trip_group['timestamp_monotonic']).sum())}."
        )
    lines.append("")
    for threshold in GAP_THRESHOLDS_SECONDS:
        column = f"gap_over_{threshold}_seconds_count"
        total = int(analysis["timestamp_summary"][column].sum())
        lines.append(f"- Intervals greater than {threshold} seconds: **{total:,}**.")
    lines.append("")
    if worst_gaps["maximum_interval_seconds"].notna().any():
        lines.append("Trips with the largest individual timestamp gaps:")
        for row in worst_gaps.itertuples(index=False):
            lines.append(
                f"- `{row.trip_id}` ({row.dataset_split}): "
                f"{_format_number(row.maximum_interval_seconds, 0)} s maximum gap."
            )
    else:
        lines.append("No finite timestamp intervals were available.")

    lines.extend(
        [
            "",
            "## 4. Speed findings",
            "",
            "Distance is WGS84 geodesic distance in metres (`src.geo.calculate_distance`). Speed is distance divided by positive elapsed seconds and is reported in m/s and km/h. Pairs with invalid coordinates/timestamps or non-positive elapsed time remain in the interval table with unavailable speed.",
            "",
            f"The unusually-high-speed flag threshold is **{_format_number(analysis['high_speed_threshold_mps'] * 3.6)} km/h** "
            f"(maximum of a 200 km/h floor and pooled Q3 + 3×IQR = "
            f"{_format_number(analysis['robust_speed_threshold_mps'] * 3.6)} km/h). "
            "This is a review threshold, not an automatic outlier filter.",
            "",
        ]
    )
    for split in SPLITS:
        speeds = all_intervals.loc[
            (all_intervals["dataset_split"] == split) & all_intervals["speed_mps"].notna(),
            "speed_mps",
        ].to_numpy(dtype=np.float64)
        trip_group = valid_trips.loc[valid_trips["dataset_split"] == split]
        lines.append(
            f"- **{split}:** {len(speeds):,} valid speed calculations; "
            f"mean {_format_number(np.mean(speeds) if len(speeds) else np.nan)} m/s; "
            f"median {_format_number(np.median(speeds) if len(speeds) else np.nan)} m/s; "
            f"maximum {_format_number(np.max(speeds) * 3.6 if len(speeds) else np.nan)} km/h; "
            f"invalid speed calculations "
            f"{int(trip_group['invalid_speed_calculation_count'].sum()):,}; "
            f"flagged high-speed transitions "
            f"{int(trip_group['unusually_high_speed_interval_count'].sum()):,}."
        )
    lines.append("")
    if worst_speeds.empty:
        lines.append("No trips exceeded the high-speed review threshold.")
    else:
        lines.append("Trips with the most transitions above the review threshold:")
        for row in worst_speeds.itertuples(index=False):
            lines.append(
                f"- `{row.trip_id}` ({row.dataset_split}): "
                f"{row.unusually_high_speed_interval_count:,} transitions; "
                f"maximum flagged speed {_format_number(row.maximum_flagged_speed_kmh)} km/h."
            )

    lines.extend(
        [
            "",
            "## 5. Coordinate findings",
            "",
            f"Across valid coordinate pairs, longitude spans "
            f"{_format_number(all_coordinates['longitude_min'])}° to "
            f"{_format_number(all_coordinates['longitude_max'])}° and latitude spans "
            f"{_format_number(all_coordinates['latitude_min'])}° to "
            f"{_format_number(all_coordinates['latitude_max'])}°. "
            f"Mean coordinate: ({_format_number(all_coordinates['longitude_mean'])}°, "
            f"{_format_number(all_coordinates['latitude_mean'])}°); median coordinate: "
            f"({_format_number(all_coordinates['longitude_median'])}°, "
            f"{_format_number(all_coordinates['latitude_median'])}°).",
            f"Missing longitude values: {int(all_coordinates['missing_longitude_values']):,}; "
            f"missing latitude values: {int(all_coordinates['missing_latitude_values']):,}; "
            f"invalid coordinate observations: "
            f"{int(all_coordinates['invalid_coordinate_observations']):,}.",
            f"The overall bounding envelope is approximately "
            f"{_format_number(all_coverage['approximate_width_km'])} km × "
            f"{_format_number(all_coverage['approximate_height_km'])} km "
            f"({_format_number(all_coverage['approximate_bbox_area_km2'])} km²). "
            "These are geodesic dimensions of a bounding rectangle, not route distances; the dataset is geographically dispersed.",
            "",
            "## 6. GPS trajectory maps",
            "",
            "Maps use only `.track` coordinates, ordered by timestamp (stable observation order for ties). Invalid-coordinate rows are omitted from map rendering only and remain in the analysis tables.",
        ]
    )
    if map_records.empty:
        lines.append("No trips had valid coordinates available for mapping.")
    else:
        for row in map_records.itertuples(index=False):
            lines.append(
                f"- {row.selection}: `{row.trip_id}` ({row.dataset_split}, "
                    f"{row.gps_point_count:,} fixes, {row.mapped_point_count:,} mapped points) — "
                    f"`reports/maps/{Path(row.map_path).name}`."
            )

    lines.extend(["", "## 7. Data-quality findings", ""])
    if quality.empty:
        lines.append("No data-quality issues were detected.")
    else:
        issue_totals = (
            quality.groupby("issue_type", as_index=False)["count"].sum().sort_values(
                "issue_type"
            )
        )
        lines.append("| Issue type | Count |")
        lines.append("|---|---:|")
        for row in issue_totals.itertuples(index=False):
            lines.append(f"| {row.issue_type} | {row.count:,} |")
        lines.append("")
        lines.append("The trip-level details and descriptions are available in the notebook's `quality_issues` table.")
    lines.extend(
        [
            "",
            "## 8. Unusual trips and investigation priorities",
            "",
            "Calculated findings:",
        ]
    )
    if not worst_gaps.empty:
        largest_gap = worst_gaps.iloc[0]
        lines.append(
            f"- `{largest_gap['trip_id']}` ({largest_gap['dataset_split']}) has the largest observed finite interval "
            f"({_format_number(largest_gap['maximum_interval_seconds'], 0)} s)."
        )
    if not analysis["high_speed_transitions"].empty:
        peak = analysis["high_speed_transitions"].iloc[0]
        lines.append(
            f"- The highest valid point-to-point speed is {_format_number(peak['speed_kmh'])} km/h "
            f"on `{peak['trip_id']}` ({peak['dataset_split']}, transition "
            f"{int(peak['transition_index'])})."
        )
    if quality.empty:
        lines.append("- No missing tracks, empty tracks, malformed rows, invalid coordinates, or invalid timestamps were detected.")
    else:
        severe = quality.loc[
            quality["issue_type"].isin(
                (
                    "missing_track_file",
                    "empty_track_file",
                    "malformed_track_file",
                    "unreadable_track_file",
                    "invalid_coordinates",
                    "invalid_timestamp_values",
                    "zero_time_intervals",
                    "negative_time_intervals",
                    "invalid_speed_calculations",
                )
            )
        ]
        if severe.empty:
            lines.append("- No missing/empty/malformed files, invalid coordinates, invalid timestamp values, or unusable speed transitions were detected.")
        else:
            lines.append(
                f"- {len(severe):,} trip-level file/coordinate/timestamp/speed issue records require review; "
                "see the issue table for counts and IDs."
            )

    lines.extend(
        [
            "",
            "## 9. Phase 3 considerations — recommendations and assumptions",
            "",
            "These are recommendations, not calculated dataset facts:",
            "- Preserve observation order and raw timestamps; carry interval validity and gap indicators into downstream processing rather than bridging invalid or very long gaps silently.",
            "- Consider deriving speed and heading only for positive, finite time intervals and valid coordinates; retain explicit missingness flags for unavailable values.",
            "- Review flagged high-speed transitions and long-gap trips before using them to calibrate any GPS uncertainty or search-radius assumptions.",
            "- Candidate generation and feature engineering should use the existing geographic utilities and a suitable local metric projection; no candidates, labels, or model features were generated in this EDA.",
            "",
            "## Limitations",
            "",
            "- Timestamp values were treated as elapsed seconds based on the observed numeric `.track` format; no timezone or wall-clock interpretation was inferred.",
            "- Geodesic bounding-box dimensions summarize geographic coverage only and do not measure route length.",
            "- Folium maps use OpenStreetMap tiles, which require an internet connection when viewed; their HTML structure and saved paths were verified locally.",
            "",
        ]
    )
    return "\n".join(lines)


def write_summary_report(
    analysis: dict[str, Any],
    map_records: pd.DataFrame,
    report_path: str | Path,
) -> Path:
    """Write the final GPS EDA report to the requested Markdown path."""
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        build_summary_markdown(analysis, map_records),
        encoding="utf-8",
    )
    return destination
