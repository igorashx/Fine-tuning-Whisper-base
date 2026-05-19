from pathlib import Path

import numpy as np
import soundfile as sf

from asr_ro.audio_loading import TARGET_SAMPLING_RATE, load_audio_sample


def test_load_audio_sample_reads_mono_wav(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    waveform = np.linspace(-0.25, 0.25, TARGET_SAMPLING_RATE, dtype=np.float32)
    sf.write(audio_path, waveform, TARGET_SAMPLING_RATE)

    sample = load_audio_sample(audio_path)

    assert sample["sampling_rate"] == TARGET_SAMPLING_RATE
    assert sample["path"].endswith("sample.wav")
    assert len(sample["array"]) == TARGET_SAMPLING_RATE
