import io
import json
import tarfile
from pathlib import Path

from asr_ro.dataset_api import extract_archive, find_dataset_root, read_api_key, request_download_url


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_read_api_key_prefers_env(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_ENV", "token-123")

    assert read_api_key(api_key=None, api_key_env="TEST_API_ENV") == "token-123"


def test_request_download_url_parses_payload(monkeypatch) -> None:
    payload = json.dumps({"downloadUrl": "https://example.com/archive.tar.gz"}).encode("utf-8")

    monkeypatch.setattr("asr_ro.dataset_api.request.urlopen", lambda req: FakeResponse(payload))

    url = request_download_url(dataset_id="dataset-id", api_key="token")

    assert url == "https://example.com/archive.tar.gz"


def test_extract_archive_and_find_dataset_root(tmp_path: Path) -> None:
    archive_path = tmp_path / "dataset.tar.gz"
    extract_dir = tmp_path / "extracted"

    with tarfile.open(archive_path, "w:gz") as archive:
        train_data = b"client_id\tpath\tsentence_id\tsentence\n"
        train_info = tarfile.TarInfo(name="bundle/ro/train.tsv")
        train_info.size = len(train_data)
        archive.addfile(train_info, io.BytesIO(train_data))

        clips_info = tarfile.TarInfo(name="bundle/ro/clips")
        clips_info.type = tarfile.DIRTYPE
        archive.addfile(clips_info)

    extract_archive(archive_path, extract_dir)
    dataset_root = find_dataset_root(extract_dir)

    assert dataset_root.name == "bundle"
    assert (dataset_root / "ro" / "train.tsv").exists()