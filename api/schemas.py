"""Pydantic request/response models.

By design none of these models has a field for reference routes, labels or any
ground truth; ``extra="forbid"`` on the request also rejects unknown fields.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    model_available: bool
    default_decoder: str
    available_decoders: list[str]
    dev_split: str


class TripListResponse(BaseModel):
    split: str
    count: int
    trips: list[str]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decoder: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    search_radius_m: float | None = Field(default=None, gt=0, le=500)
    max_candidates: int | None = Field(default=None, ge=1, le=32)


class ObservationMatch(BaseModel):
    gps_observation_index: int
    candidate_count: int
    arc_id: int | None
    probability: float | None


class LineStringGeometry(BaseModel):
    type: Literal["LineString"] = "LineString"
    coordinates: list[list[float]]  # [[lon, lat], [lon, lat]]


class ArcProperties(BaseModel):
    arc_id: int
    sequence_index: int


class ArcFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: LineStringGeometry
    properties: ArcProperties


class GeometryCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[ArcFeature]


class TimingsMs(BaseModel):
    load: float
    candidates: float
    features: float
    classify: float
    decode: float
    total: float


class PredictMetrics(BaseModel):
    n_gps_observations: int
    n_observations_with_candidates: int
    n_candidates: int
    n_matched_observations: int
    match_coverage: float
    n_predicted_arcs: int
    mean_match_probability: float | None
    min_match_probability: float | None
    search_radius_m: float
    max_candidates: int
    timings_ms: TimingsMs


class PredictResponse(BaseModel):
    trip_id: str
    status: Literal["ok", "partial", "no_match", "no_candidates"]
    model: str | None
    decoder: str
    predicted_arc_ids: list[int]
    geometry: GeometryCollection
    matches: list[ObservationMatch]
    metrics: PredictMetrics


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    detail: ErrorDetail
