from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.hmm import decode_route
from src.models.decision_tree import predict_proba_model as decision_tree_proba
from src.models.knn import predict_proba_model as knn_proba
from src.models.logistic_regression import predict_proba_model as logistic_proba
from src.models.mlp import predict_proba_model as mlp_proba
from src.models.random_forest import predict_proba_model as random_forest_proba


class _ProbabilityPipeline:
    def __init__(
        self,
        probability_matrix: np.ndarray,
        classes: tuple[int, int],
        step_name: str | None,
    ) -> None:
        self._probability_matrix = probability_matrix
        self.classes_ = np.asarray(classes)
        self.named_steps = (
            {step_name: SimpleNamespace(classes_=self.classes_)}
            if step_name is not None
            else {}
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert len(X) == len(self._probability_matrix)
        return self._probability_matrix


def _network() -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame(
        {
            "longitude": np.arange(9, dtype=np.float64) * 0.0001,
            "latitude": np.zeros(9),
        }
    )
    arcs = pd.DataFrame(
        {
            "from_node": np.arange(8, dtype=np.int64),
            "to_node": np.arange(1, 9, dtype=np.int64),
        }
    )
    return nodes, arcs


@pytest.mark.parametrize(
    ("name", "extract_probability", "step_name", "class_order"),
    [
        ("Logistic Regression", logistic_proba, "logistic_regression", (1, 0)),
        ("Decision Tree", decision_tree_proba, "decision_tree", (1, 0)),
        ("Random Forest", random_forest_proba, "random_forest", (1, 0)),
        ("MLP", mlp_proba, "mlp", (1, 0)),
        ("KNN", knn_proba, None, (0, 1)),
    ],
)
def test_classifier_probability_interfaces_decode_without_training(
    name: str,
    extract_probability,
    step_name: str | None,
    class_order: tuple[int, int],
) -> None:
    del name
    candidate_positive_probabilities = np.array(
        [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1],
        dtype=np.float64,
    )
    if class_order == (1, 0):
        matrix = np.column_stack(
            (candidate_positive_probabilities, 1.0 - candidate_positive_probabilities)
        )
    else:
        matrix = np.column_stack(
            (1.0 - candidate_positive_probabilities, candidate_positive_probabilities)
        )
    model = _ProbabilityPipeline(matrix, class_order, step_name)
    feature_rows = np.arange(len(matrix), dtype=np.float64).reshape(-1, 1)

    probabilities = extract_probability(model, feature_rows)

    assert probabilities.shape == (len(matrix),)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    np.testing.assert_array_equal(probabilities, candidate_positive_probabilities)

    nodes, arcs = _network()
    track = pd.DataFrame(
        {
            "longitude": np.arange(4, dtype=np.float64) * 0.0001 + 0.00005,
            "latitude": np.zeros(4),
            "timestamp": np.arange(4, dtype=np.float64) + 1.0,
        }
    )
    candidates = pd.DataFrame(
        {
            "gps_observation_index": np.repeat(np.arange(4, dtype=np.int64), 2),
            "candidate_arc_id": np.column_stack(
                (np.arange(4, dtype=np.int64), np.arange(4, 8, dtype=np.int64))
            ).ravel(),
        }
    )

    result = decode_route(track, nodes, arcs, candidates, probabilities)

    assert result.predictions["classifier_only_arc_id"].tolist() == [0, 1, 2, 3]
    assert result.predictions["hmm_arc_id"].between(0, len(arcs) - 1).all()
    for row in result.predictions.itertuples(index=False):
        valid_candidates = candidates.loc[
            candidates["gps_observation_index"] == row.gps_observation_index,
            "candidate_arc_id",
        ].tolist()
        assert row.classifier_only_arc_id in valid_candidates
        assert row.hmm_arc_id in valid_candidates
