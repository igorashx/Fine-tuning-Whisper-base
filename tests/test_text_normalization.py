from asr_ro.text_normalization import normalize_for_metric, normalize_transcript


def test_normalize_transcript_keeps_diacritics_and_spaces() -> None:
    source = "  Dnă   președintă — mă  opun…  "
    assert normalize_transcript(source) == 'Dnă președintă - mă opun...'


def test_normalize_for_metric_removes_punctuation_and_lowercases() -> None:
    source = "Mă bucur, că am putut clarifica neînțelegerea!"
    assert normalize_for_metric(source) == "mă bucur că am putut clarifica neînțelegerea"
