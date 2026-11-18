"""API error types. Each maps to one HTTP status and a stable machine-readable code.

Messages are written for API clients: they never contain server file paths.
"""


class ApiError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidTripIdError(ApiError):
    status_code = 422
    code = "invalid_trip_id"


class TripNotFoundError(ApiError):
    status_code = 404
    code = "trip_not_found"


class TripDataIncompleteError(ApiError):
    status_code = 422
    code = "trip_data_incomplete"


class TripDataInvalidError(ApiError):
    status_code = 422
    code = "trip_data_invalid"


class CandidateGenerationError(ApiError):
    status_code = 422
    code = "candidate_generation_failed"


class FeatureExtractionError(ApiError):
    status_code = 422
    code = "feature_extraction_failed"


class ModelUnavailableError(ApiError):
    status_code = 503
    code = "model_unavailable"


class ClassifierError(ApiError):
    status_code = 500
    code = "classifier_error"


class UnknownDecoderError(ApiError):
    status_code = 422
    code = "unknown_decoder"


class DecoderError(ApiError):
    status_code = 500
    code = "decoder_error"


class ConfigurationError(ApiError):
    status_code = 500
    code = "configuration_error"
