from asr_ro.metrics import char_error_rate, summarize_metrics, word_error_rate


def test_word_error_rate_is_zero_for_identical_text() -> None:
    assert word_error_rate(["Ana are mere"], ["Ana are mere"]) == 0.0


def test_char_error_rate_detects_single_difference() -> None:
    assert char_error_rate(["mere"], ["pere"]) == 0.25


def test_summarize_metrics_returns_expected_keys() -> None:
    metrics = summarize_metrics(["Salut lume"], ["salut, lume"])
    assert set(metrics) == {"wer", "cer"}
    assert metrics["wer"] == 0.0
