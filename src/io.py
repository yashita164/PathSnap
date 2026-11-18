"""Load PathSnap trip files while preserving their row-index identifiers."""

from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd


class TripData(TypedDict):
    """Tables and route IDs belonging to one trip."""

    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: np.ndarray


def _read_table(path: str | Path, columns: tuple[str, ...]) -> pd.DataFrame:
    """Read a headerless, whitespace-delimited table and validate its values."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Required dataset file was not found: {file_path}")
    if file_path.stat().st_size == 0:
        raise ValueError(f"Dataset file is empty: {file_path}")

    try:
        frame = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            names=list(columns),
            dtype="string",
        )
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Dataset file contains no rows: {file_path}") from error
    except pd.errors.ParserError as error:
        raise ValueError(
            f"Could not parse {file_path}; expected {len(columns)} whitespace-separated "
            f"columns per row"
        ) from error

    if frame.empty:
        raise ValueError(f"Dataset file contains no rows: {file_path}")
    if frame.isna().any().any():
        raise ValueError(
            f"Dataset file has missing values or does not contain exactly "
            f"{len(columns)} columns per row: {file_path}"
        )

    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Column '{column}' in {file_path} must contain numeric values"
            ) from error
        if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
            raise ValueError(
                f"Column '{column}' in {file_path} must contain finite values"
            )
        frame[column] = values

    return frame


def _validate_ids(frame: pd.DataFrame, columns: tuple[str, ...], path: str | Path) -> None:
    """Require zero-based IDs to be non-negative integers that fit in int64."""
    int64_max = np.iinfo(np.int64).max
    for column in columns:
        integer_ids = []
        for value in frame[column]:
            if isinstance(value, (int, np.integer)):
                integer_id = int(value)
            else:
                numeric_value = float(value)
                if not numeric_value.is_integer():
                    raise ValueError(f"ID column '{column}' in {path} must contain integers")
                integer_id = int(numeric_value)
            if integer_id < 0:
                raise ValueError(f"ID column '{column}' in {path} cannot contain negative IDs")
            if integer_id > int64_max:
                raise ValueError(f"ID column '{column}' in {path} exceeds the int64 range")
            integer_ids.append(integer_id)
        frame[column] = np.asarray(integer_ids, dtype=np.int64)


def load_track(path: str | Path) -> pd.DataFrame:
    """Load GPS observations with longitude/latitude in degrees and time as numeric."""
    return _read_table(path, ("longitude", "latitude", "timestamp"))


def load_nodes(path: str | Path) -> pd.DataFrame:
    """Load road nodes in source order with longitude/latitude in degrees."""
    return _read_table(path, ("longitude", "latitude"))


def load_arcs(path: str | Path) -> pd.DataFrame:
    """Load directed arcs in source order; endpoint IDs index rows in the nodes file."""
    frame = _read_table(path, ("from_node", "to_node"))
    _validate_ids(frame, ("from_node", "to_node"), path)
    return frame


def load_route(path: str | Path) -> np.ndarray:
    """Load route arc IDs in source order as a one-dimensional int64 array."""
    frame = _read_table(path, ("arc_id",))
    _validate_ids(frame, ("arc_id",), path)
    return frame["arc_id"].to_numpy(dtype=np.int64)


def load_trip(trip_directory: str | Path) -> TripData:
    """Load a trip's ``.track``, ``.nodes``, ``.arcs``, and ``.route`` files.

    Each filename is formed from the directory name and its extension. Files are
    read without sorting, so their row positions remain the dataset's IDs.
    """
    directory = Path(trip_directory)
    trip_id = directory.name
    return {
        "track": load_track(directory / f"{trip_id}.track"),
        "nodes": load_nodes(directory / f"{trip_id}.nodes"),
        "arcs": load_arcs(directory / f"{trip_id}.arcs"),
        "route": load_route(directory / f"{trip_id}.route"),
    }
