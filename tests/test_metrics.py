import numpy as np
import pytest

from secure_audio.metrics import dev_threshold, eer, summary


def test_perfect_eer() -> None:
    # Both genuine scores exceed all spoof scores: FAR=FRR=0 is attainable.
    assert eer(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])) == 0


def test_known_half_eer() -> None:
    # Ascending class order 0,1,0,1: at the midpoint one of each class errs.
    assert eer(np.array([0, 1, 0, 1]), np.array([.1, .2, .3, .4])) == .5


def test_inverted_eer() -> None:
    assert eer(np.array([1, 1, 0, 0]), np.array([.1, .2, .8, .9])) == 1


def test_reference_algorithm_equivalence() -> None:
    # Independent transcription of the published DET accumulation convention.
    rng = np.random.default_rng(9)
    labels = np.array([1]*17 + [0]*23)
    scores = rng.normal(size=40)
    order = np.argsort(scores, kind="mergesort")
    true_positive_counts = np.cumsum(labels[order])
    false_negative_counts = 23 - (np.arange(1, 41) - true_positive_counts)
    frr = np.concatenate(([0], true_positive_counts / 17))
    far = np.concatenate(([1], false_negative_counts / 23))
    index = np.argmin(abs(frr - far))
    assert eer(labels, scores) == (frr[index]+far[index])/2


def test_constant_scores_tie_diagnostic() -> None:
    result = summary(np.array([0, 0, 1, 1]), np.ones(4), 1)
    assert result["eer_interpolated_tie_aware"] == .5
    assert result["tied_observations"] == 3


def test_dev_threshold_and_confusion() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([.1, .2, .8, .9])
    threshold = dev_threshold(labels, scores)
    result = summary(labels, scores, threshold)
    assert result["accuracy"] == 1
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]


@pytest.mark.parametrize("labels,scores", [([1,1], [0,1]), ([0,1], [0,float('nan')]), ([0,1], [0]), ([0,2], [0,1])])
def test_invalid_metrics(labels: list, scores: list) -> None:
    with pytest.raises(ValueError):
        eer(np.asarray(labels), np.asarray(scores))
