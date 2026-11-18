import numpy as np
import pytest

from src.geo import (
    angular_difference,
    calculate_bearing,
    calculate_distance,
    calculate_gps_heading,
    point_to_segment_distance,
    project_coordinates,
)


def test_point_on_segment_has_zero_distance() -> None:
    assert point_to_segment_distance((5.0, 0.0), (0.0, 0.0), (10.0, 0.0)) == pytest.approx(0.0)


def test_point_perpendicular_to_segment_uses_perpendicular_distance() -> None:
    assert point_to_segment_distance((5.0, 3.0), (0.0, 0.0), (10.0, 0.0)) == pytest.approx(3.0)


def test_point_beyond_segment_endpoint_uses_endpoint_distance() -> None:
    assert point_to_segment_distance((13.0, 4.0), (0.0, 0.0), (10.0, 0.0)) == pytest.approx(5.0)


def test_zero_length_segment_is_treated_as_a_point() -> None:
    assert point_to_segment_distance((5.0, 7.0), (2.0, 3.0), (2.0, 3.0)) == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("end_longitude", "end_latitude", "expected_bearing"),
    [(0.0, 1.0, 0.0), (1.0, 0.0, 90.0), (0.0, -1.0, 180.0), (-1.0, 0.0, 270.0)],
)
def test_bearing_is_clockwise_from_north(
    end_longitude: float, end_latitude: float, expected_bearing: float
) -> None:
    assert calculate_bearing(0.0, 0.0, end_longitude, end_latitude) == pytest.approx(
        expected_bearing, abs=1e-6
    )


def test_angular_difference_wraps_at_zero_degrees() -> None:
    assert angular_difference(359.0, 1.0) == pytest.approx(2.0)
    np.testing.assert_allclose(angular_difference([0.0, 180.0], [359.0, 0.0]), [1.0, 180.0])


def test_projection_returns_local_metric_coordinates() -> None:
    x_coordinates, y_coordinates = project_coordinates([0.0, 0.001], [0.0, 0.0])

    assert x_coordinates.shape == (2,)
    assert y_coordinates.shape == (2,)
    assert 100.0 < x_coordinates[1] - x_coordinates[0] < 120.0
    assert abs(y_coordinates[1] - y_coordinates[0]) < 0.1


def test_invalid_geographic_coordinates_are_rejected() -> None:
    with pytest.raises(ValueError, match="Longitude"):
        project_coordinates([181.0], [0.0])


def test_gps_heading_is_missing_for_low_speed_movement() -> None:
    headings = calculate_gps_heading(
        [0.0, 0.000001, 0.001],
        [0.0, 0.0, 0.0],
        timestamps=[0.0, 1.0, 2.0],
    )

    assert np.isnan(headings[0])
    assert np.isnan(headings[1])
    assert headings[2] == pytest.approx(90.0, abs=1e-6)


def test_calculate_distance_uses_geodesic_metres() -> None:
    distance = calculate_distance(0.0, 0.0, 0.001, 0.0)

    assert 100.0 < distance < 120.0
