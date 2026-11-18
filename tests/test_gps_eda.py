from pathlib import Path

from src.gps_eda import analyze_gps_dataset


def _write_trip(root: Path, trip_id: str, contents: str) -> None:
    trip_directory = root / "train" / trip_id
    trip_directory.mkdir(parents=True)
    (trip_directory / f"{trip_id}.track").write_text(contents, encoding="utf-8")


def test_invalid_coordinates_and_time_intervals_are_retained_and_reported(
    tmp_path: Path,
) -> None:
    (tmp_path / "student_test").mkdir()
    _write_trip(
        tmp_path,
        "00000001",
        "0 0 0\n0.001 0 1\n181 0 1\n0.002 0 0\n",
    )

    analysis = analyze_gps_dataset(tmp_path)
    trip = analysis["trip_summary"].iloc[0]
    intervals = analysis["intervals"]
    issues = analysis["quality_issues"].set_index("issue_type")["count"]

    assert len(analysis["observations"]) == 4
    assert len(intervals) == 3
    assert trip["zero_interval_count"] == 1
    assert trip["negative_interval_count"] == 1
    assert trip["invalid_speed_calculation_count"] == 2
    assert issues["invalid_coordinates"] == 1
    assert issues["zero_time_intervals"] == 1
    assert issues["negative_time_intervals"] == 1


def test_malformed_track_rows_are_counted_without_dropping_observations(
    tmp_path: Path,
) -> None:
    (tmp_path / "student_test").mkdir()
    _write_trip(tmp_path, "00000002", "0 0 0\n0 0\n0 0 2 extra\n")

    analysis = analyze_gps_dataset(tmp_path)
    inventory_row = analysis["inventory"].iloc[0]
    issues = analysis["quality_issues"].set_index("issue_type")["count"]

    assert inventory_row["track_status"] == "malformed"
    assert inventory_row["gps_point_count"] == 3
    assert len(analysis["observations"]) == 3
    assert issues["malformed_track_rows"] == 2
    assert issues["missing_values"] == 1
