import pytest

from asr_ro.parallelism import resolve_worker_count


def test_resolve_worker_count_supports_zero_when_allowed() -> None:
    assert resolve_worker_count(0, allow_zero=True) == 0


def test_resolve_worker_count_supports_auto(monkeypatch) -> None:
    monkeypatch.setattr("asr_ro.parallelism.os.cpu_count", lambda: 6)
    assert resolve_worker_count(-1, allow_zero=True) == 6


def test_resolve_worker_count_rejects_negative_values_other_than_auto() -> None:
    with pytest.raises(ValueError):
        resolve_worker_count(-2, allow_zero=True)