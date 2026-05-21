from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from asr_ro.text_normalization import normalize_transcript

OFFICIAL_SPLITS = ("train", "dev", "test")

LOGGER = logging.getLogger(__name__)


@dataclass
class ManifestRow:
    audio_path: str
    text: str


@dataclass
class PreparationTask:
    source_path: str
    target_path: str
    relative_audio_path: str
    text: str
    normalize_audio: bool
    ffmpeg_path: str


@dataclass
class PreparationResult:
    audio_path: str | None
    text: str | None
    status: str
    error: str | None = None


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


def prepare_audio_task(task: PreparationTask) -> PreparationResult:
    source_path = Path(task.source_path)
    target_path = Path(task.target_path)
    try:
        if task.normalize_audio:
            if not target_path.exists():
                convert_audio_to_wav(source_path, target_path, ffmpeg_path=task.ffmpeg_path)
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if not target_path.exists():
                shutil.copy2(source_path, target_path)
        return PreparationResult(
            audio_path=task.relative_audio_path,
            text=task.text,
            status="ok",
        )
    except Exception as error:
        return PreparationResult(
            audio_path=None,
            text=None,
            status="failed",
            error=f"{source_path}: {error}",
        )


def build_preparation_tasks(
    dataset_root: Path,
    output_root: Path,
    split: str,
    normalize_audio: bool,
    ffmpeg_path: str,
    min_duration_ms: int,
    max_duration_ms: int,
    limit: int | None,
) -> tuple[list[PreparationTask], int, int]:
    durations = read_clip_durations(dataset_root)
    rows = load_split_rows(dataset_root, split)
    normalized_audio_root = output_root / "audio_16k" / split
    source_audio_root = output_root / "source_audio" / split
    skipped_missing = 0
    skipped_duration = 0
    tasks: list[PreparationTask] = []

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
        else:
            target_path = source_audio_root / clip_name

        tasks.append(
            PreparationTask(
                source_path=str(source_path),
                target_path=str(target_path),
                relative_audio_path=target_path.relative_to(output_root).as_posix(),
                text=normalize_transcript(row["sentence"]),
                normalize_audio=normalize_audio,
                ffmpeg_path=ffmpeg_path,
            )
        )

        if limit is not None and len(tasks) >= limit:
            break

    return tasks, skipped_missing, skipped_duration


def iter_preparation_results(
    tasks: list[PreparationTask],
    split: str,
    num_workers: int,
    chunksize: int,
):
    if num_workers < 1:
        raise ValueError("`num_workers` trebuie să fie cel puțin 1.")
    if chunksize < 1:
        raise ValueError("`chunksize` trebuie să fie cel puțin 1.")

    if num_workers == 1:
        for task in tqdm(tasks, total=len(tasks), desc=f"Pregătire {split}", unit="exemplu", leave=True):
            yield prepare_audio_task(task)
        return

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        mapped_results = executor.map(prepare_audio_task, tasks, chunksize=chunksize)
        for result in tqdm(mapped_results, total=len(tasks), desc=f"Pregătire {split}", unit="exemplu", leave=True):
            yield result


def prepare_split_manifest(
    dataset_root: Path,
    output_root: Path,
    split: str,
    normalize_audio: bool = True,
    ffmpeg_path: str = "ffmpeg",
    min_duration_ms: int = 500,
    max_duration_ms: int = 30_000,
    limit: int | None = None,
    num_workers: int = 1,
    chunksize: int = 1,
) -> dict[str, object]:
    LOGGER.info("Pregătesc split-ul `%s` din `%s`.", split, dataset_root)
    tasks, skipped_missing, skipped_duration = build_preparation_tasks(
        dataset_root=dataset_root,
        output_root=output_root,
        split=split,
        normalize_audio=normalize_audio,
        ffmpeg_path=ffmpeg_path,
        min_duration_ms=min_duration_ms,
        max_duration_ms=max_duration_ms,
        limit=limit,
    )
    manifest_rows: list[ManifestRow] = []
    skipped_failed = 0

    LOGGER.info(
        "Split `%s`: %s task-uri eligibile, `%s` workeri, `chunksize=%s`.",
        split,
        len(tasks),
        num_workers,
        chunksize,
    )
    for result in iter_preparation_results(tasks, split=split, num_workers=num_workers, chunksize=chunksize):
        if result.status == "ok" and result.audio_path and result.text is not None:
            manifest_rows.append(ManifestRow(audio_path=result.audio_path, text=result.text))
            continue
        skipped_failed += 1
        if result.error:
            LOGGER.warning("Exemplu omis în `%s`: %s", split, result.error)

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
        "skipped_failed": skipped_failed,
        "manifest_path": str(manifest_path.resolve()),
        "audio_root": str(output_root.resolve()),
        "normalize_audio": normalize_audio,
        "num_workers": num_workers,
    }
    LOGGER.info(
        "Split `%s` gata: %s exemple, %s lipsă, %s filtrate după durată, %s eșuate.",
        split,
        len(manifest_rows),
        skipped_missing,
        skipped_duration,
        skipped_failed,
    )
    return summary


def prepare_all_manifests(
    dataset_root: Path,
    output_root: Path,
    normalize_audio: bool = True,
    ffmpeg_path: str = "ffmpeg",
    min_duration_ms: int = 500,
    max_duration_ms: int = 30_000,
    limit_per_split: int | None = None,
    num_workers: int = 1,
    chunksize: int = 1,
) -> list[dict[str, object]]:
    output_root.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Pornesc generarea manifestelor în `%s`.", output_root)
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
                num_workers=num_workers,
                chunksize=chunksize,
            )
        )
    summary_path = output_root / "manifests" / "summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Toate manifestele au fost generate. Rezumatul este în `%s`.", summary_path)
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
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=1)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
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
        num_workers=args.num_workers,
        chunksize=args.chunksize,
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
