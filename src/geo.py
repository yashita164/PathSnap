"""Geographic calculations for GPS observations and road geometry.

Geographic coordinate arguments use ``(longitude, latitude)`` in degrees.
Projected coordinates and distance results use metres. Bearings and headings
are degrees clockwise from north in the range [0, 360).
"""

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyproj import CRS, Geod, Transformer


FloatArray: TypeAlias = NDArray[np.float64]
_GEOD = Geod(ellps="WGS84")


def _longitude_latitude(
    longitude: ArrayLike, latitude: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Validate and broadcast longitude/latitude values in degrees."""
    try:
        longitude_array, latitude_array = np.broadcast_arrays(
            np.asarray(longitude, dtype=np.float64),
            np.asarray(latitude, dtype=np.float64),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Longitude and latitude must be broadcast-compatible numbers") from error

    if not np.isfinite(longitude_array).all() or not np.isfinite(latitude_array).all():
        raise ValueError("Longitude and latitude must be finite")
    if ((longitude_array < -180) | (longitude_array > 180)).any():
        raise ValueError("Longitude must be in the range [-180, 180] degrees")
    if ((latitude_array < -90) | (latitude_array > 90)).any():
        raise ValueError("Latitude must be in the range [-90, 90] degrees")
    return longitude_array, latitude_array


def _scalar_if_scalar(value: FloatArray) -> float | FloatArray:
    """Return a Python float for scalar results and an array otherwise."""
    if value.ndim == 0:
        return float(value)
    return value


def project_coordinates(
    longitude: ArrayLike, latitude: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Project a trip's GPS coordinates into a local azimuthal-equidistant CRS.

    The CRS is centered on the circular-mean longitude and mean latitude of the
    supplied trip coordinates. Input coordinates are longitude/latitude degrees;
    returned ``(x, y)`` arrays are metres, with x eastward and y northward.
    """
    longitude_array, latitude_array = _longitude_latitude(longitude, latitude)
    if longitude_array.size == 0:
        raise ValueError("At least one coordinate is required to choose a local projection")

    longitude_radians = np.deg2rad(longitude_array)
    center_longitude = np.rad2deg(
        np.arctan2(np.sin(longitude_radians).mean(), np.cos(longitude_radians).mean())
    )
    center_latitude = float(latitude_array.mean())
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_latitude} +lon_0={center_longitude} "
        "+datum=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    x_coordinates, y_coordinates = transformer.transform(longitude_array, latitude_array)
    return np.asarray(x_coordinates, dtype=np.float64), np.asarray(y_coordinates, dtype=np.float64)


def _projected_points(value: ArrayLike, name: str) -> FloatArray:
    """Validate arrays of projected x/y coordinates in metres."""
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric x/y coordinates") from error
    if points.ndim == 0 or points.shape[-1] != 2:
        raise ValueError(f"{name} must have a final dimension of size 2 (x, y)")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain finite coordinates")
    return points


def point_to_segment_distance(
    point: ArrayLike, segment_start: ArrayLike, segment_end: ArrayLike
) -> float | FloatArray:
    """Return shortest point-to-segment distance in projected metres.

    Inputs are ``(x, y)`` coordinates in a metric CRS. Arrays may be broadcast
    together; a zero-length segment is treated as a point.
    """
    point_array = _projected_points(point, "point")
    start_array = _projected_points(segment_start, "segment_start")
    end_array = _projected_points(segment_end, "segment_end")
    try:
        point_array, start_array, end_array = np.broadcast_arrays(
            point_array, start_array, end_array
        )
    except ValueError as error:
        raise ValueError("Point and segment coordinates must be broadcast-compatible") from error

    segment_vector = end_array - start_array
    point_vector = point_array - start_array
    segment_length_squared = np.sum(segment_vector * segment_vector, axis=-1)
    projection = np.divide(
        np.sum(point_vector * segment_vector, axis=-1),
        segment_length_squared,
        out=np.zeros_like(segment_length_squared, dtype=np.float64),
        where=segment_length_squared != 0,
    )
    projection = np.clip(projection, 0.0, 1.0)
    closest_point = start_array + projection[..., np.newaxis] * segment_vector
    distances = np.linalg.norm(point_array - closest_point, axis=-1)
    return _scalar_if_scalar(np.asarray(distances, dtype=np.float64))


def calculate_bearing(
    longitude_1: ArrayLike,
    latitude_1: ArrayLike,
    longitude_2: ArrayLike,
    latitude_2: ArrayLike,
) -> float | FloatArray:
    """Calculate initial geodesic bearing clockwise from north, in degrees.

    Coincident coordinates have no defined direction and produce ``NaN``.
    Coordinate arrays are broadcast together.
    """
    lon_1, lat_1 = _longitude_latitude(longitude_1, latitude_1)
    lon_2, lat_2 = _longitude_latitude(longitude_2, latitude_2)
    try:
        lon_1, lat_1, lon_2, lat_2 = np.broadcast_arrays(lon_1, lat_1, lon_2, lat_2)
    except ValueError as error:
        raise ValueError("Coordinate pairs must be broadcast-compatible") from error
    azimuth, _, distance = _GEOD.inv(lon_1, lat_1, lon_2, lat_2)
    bearing = np.mod(np.asarray(azimuth, dtype=np.float64), 360.0)
    bearing = np.where(np.asarray(distance) == 0, np.nan, bearing)
    return _scalar_if_scalar(np.asarray(bearing, dtype=np.float64))


def calculate_gps_heading(
    longitude: ArrayLike,
    latitude: ArrayLike,
    timestamps: ArrayLike | None = None,
    min_speed_mps: float = 0.5,
) -> FloatArray:
    """Return per-observation headings from consecutive GPS fixes.

    The first heading is ``NaN`` because it has no preceding fix. When timestamps
    (seconds) are provided, headings below ``min_speed_mps`` are also ``NaN``;
    non-increasing timestamp pairs are treated as unavailable. Without timestamps,
    zero-length movements have missing headings but speed cannot be inferred.
    """
    try:
        raw_longitude = np.asarray(longitude, dtype=np.float64)
        raw_latitude = np.asarray(latitude, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("GPS longitude and latitude must contain numeric values") from error
    if raw_longitude.ndim != 1 or raw_latitude.ndim != 1:
        raise ValueError("GPS longitude and latitude inputs must be one-dimensional")
    if raw_longitude.shape != raw_latitude.shape:
        raise ValueError("GPS longitude and latitude must have the same number of observations")
    longitude_array, latitude_array = _longitude_latitude(raw_longitude, raw_latitude)
    if not np.isfinite(min_speed_mps) or min_speed_mps < 0:
        raise ValueError("min_speed_mps must be a finite, non-negative number")

    observation_count = longitude_array.size
    headings = np.full(observation_count, np.nan, dtype=np.float64)
    timestamp_array: FloatArray | None = None
    if timestamps is not None:
        try:
            timestamp_array = np.asarray(timestamps, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("timestamps must contain numeric seconds") from error
        if timestamp_array.ndim != 1 or timestamp_array.size != observation_count:
            raise ValueError("timestamps must be one-dimensional with one value per GPS observation")
        if not np.isfinite(timestamp_array).all():
            raise ValueError("timestamps must be finite")
    if observation_count < 2:
        return headings

    bearings = np.asarray(
        calculate_bearing(
            longitude_array[:-1],
            latitude_array[:-1],
            longitude_array[1:],
            latitude_array[1:],
        ),
        dtype=np.float64,
    )
    if timestamp_array is None:
        headings[1:] = bearings
        return headings

    distances = np.asarray(
        calculate_distance(
            longitude_array[:-1],
            latitude_array[:-1],
            longitude_array[1:],
            latitude_array[1:],
        ),
        dtype=np.float64,
    )
    elapsed_seconds = np.diff(timestamp_array)
    speeds = np.divide(
        distances,
        elapsed_seconds,
        out=np.full_like(distances, np.nan),
        where=elapsed_seconds > 0,
    )
    headings[1:] = np.where(speeds >= min_speed_mps, bearings, np.nan)
    return headings


def angular_difference(angle_1: ArrayLike, angle_2: ArrayLike) -> float | FloatArray:
    """Return the smallest absolute angular separation in degrees, in [0, 180]."""
    try:
        first, second = np.broadcast_arrays(
            np.asarray(angle_1, dtype=np.float64),
            np.asarray(angle_2, dtype=np.float64),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Angles must be broadcast-compatible numbers") from error
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Angles must be finite")
    difference = np.abs(np.mod(first - second + 180.0, 360.0) - 180.0)
    return _scalar_if_scalar(np.asarray(difference, dtype=np.float64))


def calculate_distance(
    longitude_1: ArrayLike,
    latitude_1: ArrayLike,
    longitude_2: ArrayLike,
    latitude_2: ArrayLike,
) -> float | FloatArray:
    """Calculate WGS84 geodesic distance between coordinates, in metres."""
    lon_1, lat_1 = _longitude_latitude(longitude_1, latitude_1)
    lon_2, lat_2 = _longitude_latitude(longitude_2, latitude_2)
    try:
        lon_1, lat_1, lon_2, lat_2 = np.broadcast_arrays(lon_1, lat_1, lon_2, lat_2)
    except ValueError as error:
        raise ValueError("Coordinate pairs must be broadcast-compatible") from error
    _, _, distance = _GEOD.inv(lon_1, lat_1, lon_2, lat_2)
    return _scalar_if_scalar(np.asarray(distance, dtype=np.float64))
