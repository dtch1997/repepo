import torch
from repepo.steering.concept_metrics import (
    VarianceOfNormSimilarityMetric,
    EuclideanSimilarityMetric,
    CosineSimilarityMetric,
    compute_difference_vectors,
)

from syrupy import SnapshotAssertion


def test_compute_difference_vectors():
    pos_acts = [
        torch.tensor([[1, 2, 3]], dtype=torch.float32),
        torch.tensor([[4, 5, 6]], dtype=torch.float32),
    ]
    neg_acts = [
        torch.tensor([[7, 8, 9]], dtype=torch.float32),
        torch.tensor([[10, 11, 12]], dtype=torch.float32),
    ]
    diff_vecs = compute_difference_vectors(pos_acts, neg_acts)
    target_diff_vecs = [
        torch.tensor([[-6, -6, -6]], dtype=torch.float32),
        torch.tensor([[-6, -6, -6]], dtype=torch.float32),
    ]

    for diff_vec, target_diff_vec in zip(diff_vecs, target_diff_vecs):
        assert torch.allclose(diff_vec, target_diff_vec, atol=1e-5)


def test_cosine_similarity_metric(snapshot: SnapshotAssertion):
    metric = CosineSimilarityMetric()
    diff_vecs = [
        torch.tensor([[1, 2, 3]], dtype=torch.float32),
        torch.tensor([[4, 5, 6]], dtype=torch.float32),
    ]
    assert metric(diff_vecs) == snapshot


def test_euclidean_similarity_metric(snapshot: SnapshotAssertion):
    metric = EuclideanSimilarityMetric()
    diff_vecs = [
        torch.tensor([[1, 2, 3]], dtype=torch.float32),
        torch.tensor([[4, 5, 6]], dtype=torch.float32),
    ]
    assert metric(diff_vecs) == snapshot


def test_variance_of_norm_similarity_metric(snapshot: SnapshotAssertion):
    metric = VarianceOfNormSimilarityMetric()
    diff_vecs = [
        torch.tensor([[1, 2, 3]], dtype=torch.float32),
        torch.tensor([[4, 5, 6]], dtype=torch.float32),
    ]
    assert metric(diff_vecs) == snapshot
