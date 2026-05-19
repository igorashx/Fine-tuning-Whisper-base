from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from asr_ro.text_normalization import normalize_transcript

OFFICIAL_SPLITS = ("train", "dev", "test")


@dataclass
class ManifestRow:
    audio_path: str
    text: str


def read_clip_durations(dataset_root: Path) -> dict[str, int]:
    durations_path = dataset_root / "ro" / "clip_durations.tsv"
    durations: dict[str, int] = {}
    with durations_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["clip"]:
                durations[row["clip"]] = int(row["duration[ms]"])
    return durations


def load_split_rows(dataset_root: Path, split: str) -> list[dict[str, str]]:
    split_path = dataset_root / "ro" / f"{split}.tsv"
    with split_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def convert_audio_to_wav(source_path: Path, target_path: Path, ffmpeg_path: str = "ffmpeg") -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_ffmpeg_path = resolve_ffmpeg_path(ffmpeg_path)
    command = [
        resolved_ffmpeg_path,
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(target_path),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "Nu am găsit `ffmpeg`. Instalează ffmpeg în PATH sau folosește fallback-ul din `imageio-ffmpeg`. În Colab, ffmpeg este disponibil implicit."
        ) from error


def resolve_ffmpeg_path(ffmpeg_path: str) -> str:
    if ffmpeg_path != "ffmpeg":
        return ffmpeg_path
    discovered = shutil.which(ffmpeg_path)
    if discovered:
        return discovered
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except Exception as error:
        raise RuntimeError(
            "Nu am găsit `ffmpeg` în PATH și nici fallback-ul `imageio-ffmpeg` nu este disponibil."
        ) from error


def prepare_split_manifest(
    dataset_root: Path,
    output_root: Path,
    split: str,
    normalize_audio: bool = True,
    ffmpeg_path: str = "ffmpeg",
    min_duration_ms: int = 500,
    max_duration_ms: int = 30_000,
    limit: int | None = None,
) -> dict[str, object]:
    durations = read_clip_durations(dataset_root)
    rows = load_split_rows(dataset_root, split)
    manifest_rows: list[ManifestRow] = []
    normalized_audio_root = output_root / "audio_16k" / split
    source_audio_root = output_root / "source_audio" / split
    skipped_missing = 0
    skipped_duration = 0

    for row in rows:
        clip_name = row["path"]
        source_path = dataset_root / "ro" / "clips" / clip_name
        if not source_path.exists():
            skipped_missing += 1
            continue

        duration_ms = durations.get(clip_name, 0)
        if duration_ms and not (min_duration_ms <= duration_ms <= max_duration_ms):
            skipped_duration += 1
            continue

        if normalize_audio:
            target_path = normalized_audio_root / f"{Path(clip_name).stem}.wav"
            if not target_path.exists():
                convert_audio_to_wav(source_path, target_path, ffmpeg_path=ffmpeg_path)
            audio_path = target_path
        else:
            target_path = source_audio_root / clip_name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if not target_path.exists():
                shutil.copy2(source_path, target_path)
            audio_path = target_path

        manifest_rows.append(
            ManifestRow(
                audio_path=audio_path.relative_to(output_root).as_posix(),
                text=normalize_transcript(row["sentence"]),
            )
        )
        if limit is not None and len(manifest_rows) >= limit:
            break

    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{split}.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["audio_path", "text"],
        )
        writer.writeheader()
        for manifest_row in manifest_rows:
            writer.writerow(asdict(manifest_row))

    summary = {
        "split": split,
        "examples": len(manifest_rows),
        "skipped_missing": skipped_missing,
        "skipped_duration": skipped_duration,
        "manifest_path": str(manifest_path.resolve()),
        "audio_root": str(output_root.resolve()),
        "normalize_audio": normalize_audio,
    }
    return summary


def prepare_all_manifests(
    dataset_root: Path,
    output_root: Path,
    normalize_audio: bool = True,
    ffmpeg_path: str = "ffmpeg",
    min_duration_ms: int = 500,
    max_duration_ms: int = 30_000,
    limit_per_split: int | None = None,
) -> list[dict[str, object]]:
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for split in OFFICIAL_SPLITS:
        summaries.append(
            prepare_split_manifest(
                dataset_root=dataset_root,
                output_root=output_root,
                split=split,
                normalize_audio=normalize_audio,
                ffmpeg_path=ffmpeg_path,
                min_duration_ms=min_duration_ms,
                max_duration_ms=max_duration_ms,
                limit=limit_per_split,
            )
        )
    summary_path = output_root / "manifests" / "summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    return summaries


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pregătește manifestele Common Voice ro pentru Whisper.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--ffmpeg-path", default="ffmpeg")
    parser.add_argument("--skip-audio-normalization", action="store_true")
    parser.add_argument("--min-duration-ms", type=int, default=500)
    parser.add_argument("--max-duration-ms", type=int, default=30_000)
    parser.add_argument("--limit-per-split", type=int)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    summaries = prepare_all_manifests(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        normalize_audio=not args.skip_audio_normalization,
        ffmpeg_path=args.ffmpeg_path,
        min_duration_ms=args.min_duration_ms,
        max_duration_ms=args.max_duration_ms,
        limit_per_split=args.limit_per_split,
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
