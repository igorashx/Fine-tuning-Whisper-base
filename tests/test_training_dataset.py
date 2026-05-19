import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from asr_ro.training_dataset import WhisperTrainingDataset, read_manifest_rows


class FakeFeatureExtractor:
    def __call__(self, array, sampling_rate):
        return SimpleNamespace(input_features=[[float(len(array)), float(sampling_rate)]])


class FakeTokenizer:
    def __call__(self, text: str):
        return SimpleNamespace(input_ids=[len(text), 7])


class FakeProcessor:
    def __init__(self) -> None:
        self.feature_extractor = FakeFeatureExtractor()
        self.tokenizer = FakeTokenizer()


def write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "text"])
        writer.writeheader()
        writer.writerow({"audio_path": "audio_16k/train/sample.wav", "text": "Salut"})
        writer.writerow({"audio_path": "audio_16k/train/sample_2.wav", "text": "Bună"})


def test_read_manifest_rows_respects_limit(tmp_path: Path) -> None:
    manifest_path = tmp_path / "output" / "manifests" / "train.csv"
    write_manifest(manifest_path)

    rows = read_manifest_rows(manifest_path, limit=1)

    assert len(rows) == 1
    assert rows[0]["audio_path"].endswith("audio_16k\\train\\sample.wav") or rows[0]["audio_path"].endswith("audio_16k/train/sample.wav")


def test_whisper_training_dataset_loads_features_lazily(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "output" / "manifests" / "train.csv"
    write_manifest(manifest_path)
    rows = read_manifest_rows(manifest_path)

    def fake_load_audio_sample(audio_path: str):
        return {
            "array": np.ones(16_000, dtype=np.float32),
            "sampling_rate": 16_000,
            "path": audio_path,
        }

    monkeypatch.setattr("asr_ro.training_dataset.load_audio_sample", fake_load_audio_sample)

    dataset = WhisperTrainingDataset(rows, FakeProcessor())
    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["input_features"] == [16000.0, 16000.0]
    assert sample["labels"] == [5, 7]
