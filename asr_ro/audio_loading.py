from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

TARGET_SAMPLING_RATE = 16_000


def load_audio_sample(audio_path: str | Path) -> dict[str, Any]:
    path = Path(audio_path)
    waveform, sampling_rate = sf.read(path, dtype="float32", always_2d=False)

    if isinstance(waveform, np.ndarray) and waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    if sampling_rate != TARGET_SAMPLING_RATE:
        raise ValueError(
            f"Fișierul {path} are sampling rate {sampling_rate}, dar pipeline-ul așteaptă {TARGET_SAMPLING_RATE}. Rulează mai întâi `asr_ro.data_prep` cu normalizare audio activă."
        )

    return {
        "array": np.asarray(waveform, dtype=np.float32),
        "sampling_rate": sampling_rate,
        "path": str(path),
    }
