from types import SimpleNamespace

import numpy as np
import torch

from asr_ro.evaluate_model import transcribe_dataset


class FakeFeatureExtractor:
    def __call__(self, array, sampling_rate):
        return SimpleNamespace(input_features=[[float(len(array)), float(sampling_rate)]])

    def pad(self, input_features, return_tensors="pt"):
        padded = torch.tensor([feature["input_features"] for feature in input_features], dtype=torch.float32)
        return SimpleNamespace(input_features=padded)


class FakeProcessor:
    def __init__(self) -> None:
        self.feature_extractor = FakeFeatureExtractor()

    def get_decoder_prompt_ids(self, language: str, task: str):
        return [[1, 2]]

    def batch_decode(self, generated_ids, skip_special_tokens=True):
        return [f"pred-{index}" for index in range(len(generated_ids))]


class FakeModel:
    def __init__(self) -> None:
        self.generation_config = SimpleNamespace(language=None, task=None, forced_decoder_ids=None)

    def to(self, device: str):
        return self

    def eval(self) -> None:
        return None

    def generate(self, input_features, max_new_tokens=128):
        batch_size = input_features.shape[0]
        return torch.ones((batch_size, 2), dtype=torch.int64)


def test_transcribe_dataset_uses_dataloader_workers_zero(monkeypatch) -> None:
    rows = [
        {"audio_path": "sample-1.wav", "text": "Salut"},
        {"audio_path": "sample-2.wav", "text": "Bună"},
    ]

    def fake_load_audio_sample(audio_path: str):
        return {
            "array": np.ones(16_000, dtype=np.float32),
            "sampling_rate": 16_000,
            "path": audio_path,
        }

    monkeypatch.setattr("asr_ro.evaluate_model.load_audio_sample", fake_load_audio_sample)
    monkeypatch.setattr("asr_ro.evaluate_model.WhisperProcessor.from_pretrained", lambda *args, **kwargs: FakeProcessor())
    monkeypatch.setattr("asr_ro.evaluate_model.WhisperForConditionalGeneration.from_pretrained", lambda *args, **kwargs: FakeModel())

    predictions = transcribe_dataset("fake-model", rows, device="cpu", batch_size=2, num_workers=0)

    assert predictions == ["pred-0", "pred-1"]