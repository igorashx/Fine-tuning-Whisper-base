from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from asr_ro.audio_loading import load_audio_sample


ManifestRow = dict[str, str]


def read_manifest_rows(csv_path: Path, limit: int | None = None) -> list[ManifestRow]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[ManifestRow] = []
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


class WhisperTrainingDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[ManifestRow], processor: Any) -> None:
        self.rows = rows
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        audio = load_audio_sample(row["audio_path"])
        input_features = self.processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]
        labels = self.processor.tokenizer(row["text"]).input_ids
        return {
            "input_features": input_features,
            "labels": labels,
        }
