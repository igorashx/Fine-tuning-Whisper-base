import csv
from pathlib import Path

from asr_ro.data_prep import prepare_split_manifest


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
