from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path
from typing import Any
from urllib import request

DEFAULT_DATASET_ID = "cmn2e8rmi01l6mm07vxurptse"
DEFAULT_DATASET_SLUG = "common-voice-scripted-speech-25-0-romani-701de4ae"
DEFAULT_API_BASE_URL = "https://mozilladatacollective.com/api/datasets"
DEFAULT_API_KEY_ENV = "MOZILLA_DATA_COLLECTIVE_API_KEY"


def read_api_key(api_key: str | None = None, api_key_env: str = DEFAULT_API_KEY_ENV) -> str:
    if api_key:
        return api_key
    env_value = os.environ.get(api_key_env)
    if env_value:
        return env_value
    raise RuntimeError(
        f"Cheia API lipsește. Setează variabila de mediu `{api_key_env}` sau furnizează cheia explicit prin apel intern."
    )


def request_download_url(
    dataset_id: str = DEFAULT_DATASET_ID,
    *,
    api_key: str | None = None,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    api_base_url: str = DEFAULT_API_BASE_URL,
) -> str:
    resolved_api_key = read_api_key(api_key=api_key, api_key_env=api_key_env)
    endpoint = f"{api_base_url}/{dataset_id}/download"
    req = request.Request(
        endpoint,
        data=b"",
        method="POST",
        headers={
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        },
    )
    with request.urlopen(req) as response:
        payload = json.loads(response.read().decode("utf-8"))
    download_url = payload.get("downloadUrl")
    if not download_url:
        raise RuntimeError("Răspunsul API nu conține `downloadUrl`.")
    return download_url


def download_file(download_url: str, destination: Path, chunk_size: int = 1024 * 1024) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with request.urlopen(download_url) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
    return destination


def _assert_safe_member_path(target_dir: Path, member_name: str) -> Path:
    resolved_target = target_dir.resolve()
    candidate = (target_dir / member_name).resolve()
    if resolved_target not in candidate.parents and candidate != resolved_target:
        raise RuntimeError(f"Arhiva conține o cale invalidă: {member_name}")
    return candidate


def extract_archive(archive_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            _assert_safe_member_path(extract_dir, member.name)
        try:
            archive.extractall(extract_dir, filter="data")
        except TypeError:
            archive.extractall(extract_dir)
    return extract_dir


def find_dataset_root(search_root: Path) -> Path:
    search_root = search_root.resolve()
    if (search_root / "ro" / "train.tsv").exists() and (search_root / "ro" / "clips").exists():
        return search_root
    for candidate in search_root.rglob("train.tsv"):
        parent = candidate.parent.parent
        if candidate.parent.name == "ro" and (parent / "ro" / "clips").exists():
            return parent
    raise RuntimeError(f"Nu am găsit rădăcina datasetului în {search_root}")


def write_metadata(metadata_path: Path, payload: dict[str, Any]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def obtain_dataset(
    *,
    download_root: Path,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_slug: str = DEFAULT_DATASET_SLUG,
    api_key: str | None = None,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    force_download: bool = False,
    force_extract: bool = False,
) -> Path:
    dataset_cache_root = download_root / dataset_slug
    archive_path = dataset_cache_root / "downloads" / f"{dataset_slug}.tar.gz"
    extract_dir = dataset_cache_root / "extracted"
    metadata_path = dataset_cache_root / "metadata.json"

    if extract_dir.exists() and not force_extract:
        try:
            dataset_root = find_dataset_root(extract_dir)
            write_metadata(
                metadata_path,
                {
                    "dataset_id": dataset_id,
                    "dataset_slug": dataset_slug,
                    "archive_path": str(archive_path),
                    "extract_dir": str(extract_dir),
                    "dataset_root": str(dataset_root),
                    "source": "cache",
                },
            )
            return dataset_root
        except RuntimeError:
            pass

    if force_download or not archive_path.exists():
        download_url = request_download_url(dataset_id=dataset_id, api_key=api_key, api_key_env=api_key_env)
        download_file(download_url, archive_path)

    if force_extract and extract_dir.exists():
        for item in extract_dir.iterdir():
            if item.is_dir():
                import shutil

                shutil.rmtree(item)
            else:
                item.unlink()

    extract_archive(archive_path, extract_dir)
    dataset_root = find_dataset_root(extract_dir)
    write_metadata(
        metadata_path,
        {
            "dataset_id": dataset_id,
            "dataset_slug": dataset_slug,
            "archive_path": str(archive_path),
            "extract_dir": str(extract_dir),
            "dataset_root": str(dataset_root),
            "source": "api",
        },
    )
    return dataset_root


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Descarcă și extrage datasetul Common Voice ro prin API.")
    parser.add_argument("--download-root", type=Path, default=Path("artifacts/datasets"))
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    dataset_root = obtain_dataset(
        download_root=args.download_root,
        dataset_id=args.dataset_id,
        dataset_slug=args.dataset_slug,
        api_key_env=args.api_key_env,
        force_download=args.force_download,
        force_extract=args.force_extract,
    )
    print(str(dataset_root))


if __name__ == "__main__":
    main()
