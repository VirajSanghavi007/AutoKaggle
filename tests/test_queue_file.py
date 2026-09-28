import pytest

from queue_file import _parse_kaggle_url, QueueError


def test_parses_dataset_url():
    result = _parse_kaggle_url("https://www.kaggle.com/datasets/someowner/some-dataset")
    assert result == {"target_slug": "someowner/some-dataset"}


def test_rejects_competition_url():
    with pytest.raises(QueueError):
        _parse_kaggle_url("https://www.kaggle.com/competitions/titanic")
    with pytest.raises(QueueError):
        _parse_kaggle_url("https://www.kaggle.com/c/titanic")


def test_rejects_empty_url():
    with pytest.raises(QueueError):
        _parse_kaggle_url("")


def test_rejects_unrecognized_url():
    with pytest.raises(QueueError):
        _parse_kaggle_url("https://example.com/not-kaggle")
