"""FastAPI application.

Run from the repository root:
    uvicorn api.main:app --reload
"""

from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse

from api import __version__
from api.config import Settings, get_settings
from api.decoding import available_decoders
from api.errors import ApiError, ConfigurationError
from api.model_loader import (
    ProbabilityClassifier,
    get_classifier_loader,
    model_is_available,
)
from api.hmm_decoder import register_hmm_decoder
from api.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    TripListResponse,
)
from api.services import list_trip_ids, predict_trip, resolve_options

_ERROR_RESPONSES = {
    404: {"model": ErrorResponse, "description": "Unknown trip"},
    422: {"model": ErrorResponse, "description": "Invalid trip ID, trip data or decoder"},
    503: {"model": ErrorResponse, "description": "Classifier unavailable"},
}
_FRONTEND_FILE = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def create_app() -> FastAPI:
    register_hmm_decoder()
    app = FastAPI(
        title="PathSnap API",
        version=__version__,
        description="GPS map matching: candidate classification and route reconstruction.",
    )

    try:
        origins = get_settings().cors_origins
    except ConfigurationError:
        origins = ()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(origins),
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.get("/", include_in_schema=False)
    def frontend() -> FileResponse:
        return FileResponse(_FRONTEND_FILE, media_type="text/html")

    @app.exception_handler(ApiError)
    async def handle_api_error(_: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={"detail": {"code": error.code, "message": error.message}},
        )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
        return HealthResponse(
            status="ok",
            service="pathsnap-api",
            version=__version__,
            model_available=model_is_available(settings),
            default_decoder=settings.decoder,
            available_decoders=available_decoders(),
            dev_split=settings.dev_split,
        )

    @app.get("/trips", response_model=TripListResponse, tags=["trips"])
    def trips(settings: Settings = Depends(get_settings)) -> TripListResponse:
        trip_ids = list_trip_ids(settings)
        return TripListResponse(
            split=settings.dev_split, count=len(trip_ids), trips=trip_ids
        )

    @app.post(
        "/predict/{trip_id}",
        response_model=PredictResponse,
        responses=_ERROR_RESPONSES,
        tags=["predict"],
    )
    def predict(
        trip_id: str,
        request: PredictRequest | None = None,
        settings: Settings = Depends(get_settings),
        load_classifier=Depends(get_classifier_loader),
    ) -> PredictResponse:
        options = resolve_options(request, settings)
        return predict_trip(trip_id, settings, load_classifier, options)

    return app


app = create_app()
