from __future__ import annotations

from typing import Iterable

from asr_ro.text_normalization import normalize_for_metric


def _levenshtein_distance(source: list[str], target: list[str]) -> int:
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous_row = list(range(len(target) + 1))
    for source_index, source_item in enumerate(source, start=1):
        current_row = [source_index]
        for target_index, target_item in enumerate(target, start=1):
            insertions = previous_row[target_index] + 1
            deletions = current_row[target_index - 1] + 1
            substitutions = previous_row[target_index - 1] + (source_item != target_item)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _safe_division(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator or 1)


def word_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    total_distance = 0
    total_words = 0
    for reference, hypothesis in zip(references, hypotheses):
        reference_tokens = normalize_for_metric(reference).split()
        hypothesis_tokens = normalize_for_metric(hypothesis).split()
        total_distance += _levenshtein_distance(reference_tokens, hypothesis_tokens)
        total_words += len(reference_tokens)
    return _safe_division(total_distance, total_words)


def char_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    total_distance = 0
    total_characters = 0
    for reference, hypothesis in zip(references, hypotheses):
        reference_text = list(normalize_for_metric(reference).replace(" ", ""))
        hypothesis_text = list(normalize_for_metric(hypothesis).replace(" ", ""))
        total_distance += _levenshtein_distance(reference_text, hypothesis_text)
        total_characters += len(reference_text)
    return _safe_division(total_distance, total_characters)


def summarize_metrics(references: Iterable[str], hypotheses: Iterable[str]) -> dict[str, float]:
    reference_list = list(references)
    hypothesis_list = list(hypotheses)
    return {
        "wer": word_error_rate(reference_list, hypothesis_list),
        "cer": char_error_rate(reference_list, hypothesis_list),
    }
