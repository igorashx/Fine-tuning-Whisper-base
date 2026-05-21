import csv
from pathlib import Path

import pytest

from asr_ro.data_prep import prepare_split_manifest, resolve_num_workers


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_prepare_split_manifest_exports_audio_path_and_text(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    clips_dir = dataset_root / "ro" / "clips"
    clips_dir.mkdir(parents=True)

    clip_path = clips_dir / "sample.mp3"
    clip_path.write_bytes(b"fake mp3")

    write_tsv(
        dataset_root / "ro" / "clip_durations.tsv",
        [{"clip": "sample.mp3", "duration[ms]": "1200"}],
        ["clip", "duration[ms]"],
    )
    write_tsv(
        dataset_root / "ro" / "train.tsv",
        [
            {
                "client_id": "speaker-1",
                "path": "sample.mp3",
                "sentence_id": "sent-1",
                "sentence": "  Salut,   România!  ",
                "sentence_domain": "",
                "up_votes": "2",
                "down_votes": "0",
                "age": "",
                "gender": "",
                "accents": "",
                "variant": "",
                "locale": "ro",
                "segment": "",
            }
        ],
        [
            "client_id",
            "path",
            "sentence_id",
            "sentence",
            "sentence_domain",
            "up_votes",
            "down_votes",
            "age",
            "gender",
            "accents",
            "variant",
            "locale",
            "segment",
        ],
    )

    output_root = tmp_path / "artifacts"
    summary = prepare_split_manifest(
        dataset_root=dataset_root,
        output_root=output_root,
        split="train",
        normalize_audio=False,
    )

    manifest_path = output_root / "manifests" / "train.csv"
    assert summary["examples"] == 1
    assert manifest_path.exists()

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == ["audio_path", "text"]
    assert rows[0]["audio_path"] == "source_audio/train/sample.mp3"
    assert (output_root / rows[0]["audio_path"]).exists()
    assert rows[0]["text"] == "Salut, România!"


def test_prepare_split_manifest_parallel_copy_mode_preserves_order(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    clips_dir = dataset_root / "ro" / "clips"
    clips_dir.mkdir(parents=True)

    for clip_name in ("sample_1.mp3", "sample_2.mp3"):
        (clips_dir / clip_name).write_bytes(b"fake mp3")

    write_tsv(
        dataset_root / "ro" / "clip_durations.tsv",
        [
            {"clip": "sample_1.mp3", "duration[ms]": "1200"},
            {"clip": "sample_2.mp3", "duration[ms]": "1300"},
        ],
        ["clip", "duration[ms]"],
    )
    write_tsv(
        dataset_root / "ro" / "train.tsv",
        [
            {
                "client_id": "speaker-1",
                "path": "sample_2.mp3",
                "sentence_id": "sent-2",
                "sentence": "Salut doi",
                "sentence_domain": "",
                "up_votes": "2",
                "down_votes": "0",
                "age": "",
                "gender": "",
                "accents": "",
                "variant": "",
                "locale": "ro",
                "segment": "",
            },
            {
                "client_id": "speaker-2",
                "path": "sample_1.mp3",
                "sentence_id": "sent-1",
                "sentence": "Salut unu",
                "sentence_domain": "",
                "up_votes": "2",
                "down_votes": "0",
                "age": "",
                "gender": "",
                "accents": "",
                "variant": "",
                "locale": "ro",
                "segment": "",
            },
        ],
        [
            "client_id",
            "path",
            "sentence_id",
            "sentence",
            "sentence_domain",
            "up_votes",
            "down_votes",
            "age",
            "gender",
            "accents",
            "variant",
            "locale",
            "segment",
        ],
    )

    output_root = tmp_path / "artifacts"
    summary = prepare_split_manifest(
        dataset_root=dataset_root,
        output_root=output_root,
        split="train",
        normalize_audio=False,
        num_workers=2,
        chunksize=1,
    )

    manifest_path = output_root / "manifests" / "train.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["examples"] == 2
    assert summary["skipped_failed"] == 0
    assert [row["audio_path"] for row in rows] == [
        "source_audio/train/sample_2.mp3",
        "source_audio/train/sample_1.mp3",
    ]
    assert [row["text"] for row in rows] == ["Salut doi", "Salut unu"]


def test_prepare_split_manifest_counts_failed_parallel_items(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    clips_dir = dataset_root / "ro" / "clips"
    clips_dir.mkdir(parents=True)

    clip_path = clips_dir / "sample.mp3"
    clip_path.write_bytes(b"fake mp3")

    write_tsv(
        dataset_root / "ro" / "clip_durations.tsv",
        [{"clip": "sample.mp3", "duration[ms]": "1200"}],
        ["clip", "duration[ms]"],
    )
    write_tsv(
        dataset_root / "ro" / "train.tsv",
        [
            {
                "client_id": "speaker-1",
                "path": "sample.mp3",
                "sentence_id": "sent-1",
                "sentence": "Salut eroare",
                "sentence_domain": "",
                "up_votes": "2",
                "down_votes": "0",
                "age": "",
                "gender": "",
                "accents": "",
                "variant": "",
                "locale": "ro",
                "segment": "",
            }
        ],
        [
            "client_id",
            "path",
            "sentence_id",
            "sentence",
            "sentence_domain",
            "up_votes",
            "down_votes",
            "age",
            "gender",
            "accents",
            "variant",
            "locale",
            "segment",
        ],
    )

    output_root = tmp_path / "artifacts"
    summary = prepare_split_manifest(
        dataset_root=dataset_root,
        output_root=output_root,
        split="train",
        normalize_audio=True,
        ffmpeg_path="missing-ffmpeg-binary",
        num_workers=2,
        chunksize=1,
    )

    manifest_path = output_root / "manifests" / "train.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["examples"] == 0
    assert summary["skipped_failed"] == 1
    assert rows == []


def test_resolve_num_workers_auto_uses_cpu_count(monkeypatch) -> None:
    monkeypatch.setattr("asr_ro.data_prep.os.cpu_count", lambda: 8)

    assert resolve_num_workers(-1) == 8


def test_resolve_num_workers_rejects_zero() -> None:
    with pytest.raises(ValueError):
        resolve_num_workers(0)


def test_prepare_split_manifest_auto_workers_reports_resolved_value(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    clips_dir = dataset_root / "ro" / "clips"
    clips_dir.mkdir(parents=True)

    clip_path = clips_dir / "sample.mp3"
    clip_path.write_bytes(b"fake mp3")

    write_tsv(
        dataset_root / "ro" / "clip_durations.tsv",
        [{"clip": "sample.mp3", "duration[ms]": "1200"}],
        ["clip", "duration[ms]"],
    )
    write_tsv(
        dataset_root / "ro" / "train.tsv",
        [
            {
                "client_id": "speaker-1",
                "path": "sample.mp3",
                "sentence_id": "sent-1",
                "sentence": "Salut auto",
                "sentence_domain": "",
                "up_votes": "2",
                "down_votes": "0",
                "age": "",
                "gender": "",
                "accents": "",
                "variant": "",
                "locale": "ro",
                "segment": "",
            }
        ],
        [
            "client_id",
            "path",
            "sentence_id",
            "sentence",
            "sentence_domain",
            "up_votes",
            "down_votes",
            "age",
            "gender",
            "accents",
            "variant",
            "locale",
            "segment",
        ],
    )

    output_root = tmp_path / "artifacts"
    summary = prepare_split_manifest(
        dataset_root=dataset_root,
        output_root=output_root,
        split="train",
        normalize_audio=False,
        num_workers=-1,
    )

    assert summary["examples"] == 1
    assert summary["num_workers"] >= 1
