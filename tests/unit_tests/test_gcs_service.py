import json
from datetime import datetime
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.app.services.gcs_service import (
    _MULTIPART_THRESHOLD,
    download_blob_to_file,
    download_json_from_gcs,
    find_prediction_in_bucket,
    get_latest_blob,
    upload_file_to_gcs,
    upload_json_to_gcs,
)


@pytest.fixture
def mock_storage_client():
    with patch("src.app.services.gcs_service.get_client") as mock_get_client:
        client = MagicMock()
        mock_get_client.return_value = client
        yield client


def test_upload_file_to_gcs_small_file(mock_storage_client):
    # Setup
    bucket_name = "test-bucket"
    local_path = "small_file.txt"
    blob_name = "dest_blob"
    file_size = _MULTIPART_THRESHOLD - 1024

    mock_blob = MagicMock()
    mock_storage_client.bucket.return_value.blob.return_value = mock_blob

    with (
        patch("pathlib.Path.stat") as mock_stat,
        patch("builtins.open", mock_open(read_data=b"data")),
    ):
        mock_stat.return_value.st_size = file_size

        # Execute
        upload_file_to_gcs(bucket_name, local_path, blob_name)

        # Verify: upload_from_file should be used for small files
        mock_blob.upload_from_file.assert_called_once()
        mock_blob.upload_from_filename.assert_not_called()


def test_upload_file_to_gcs_large_file(mock_storage_client):
    # Setup
    bucket_name = "test-bucket"
    local_path = "large_file.bin"
    blob_name = "dest_blob"
    file_size = _MULTIPART_THRESHOLD + 1024

    mock_blob = MagicMock()
    mock_storage_client.bucket.return_value.blob.return_value = mock_blob

    with patch("pathlib.Path.stat") as mock_stat:
        mock_stat.return_value.st_size = file_size

        # Execute
        upload_file_to_gcs(bucket_name, local_path, blob_name)

        # Verify: upload_from_filename (resumable) should be used for large files
        mock_blob.upload_from_filename.assert_called_once_with(local_path, retry=pytest.any)
        mock_blob.upload_from_file.assert_not_called()


def test_upload_file_to_gcs_retry_and_success(mock_storage_client):
    # Setup
    mock_blob = MagicMock()
    # Fail once, then succeed
    mock_blob.upload_from_filename.side_effect = [Exception("Transient Error"), None]
    mock_storage_client.bucket.return_value.blob.return_value = mock_blob

    with (
        patch("pathlib.Path.stat") as mock_stat,
        patch("time.sleep") as mock_sleep,
    ):  # Don't actually wait in tests
        mock_stat.return_value.st_size = _MULTIPART_THRESHOLD + 1024

        # Execute
        upload_file_to_gcs("b", "p", "n", max_attempts=2)

        # Verify
        assert mock_blob.upload_from_filename.call_count == 2
        mock_sleep.assert_called_once()


def test_upload_file_to_gcs_terminal_failure(mock_storage_client):
    # Setup
    mock_blob = MagicMock()
    mock_blob.upload_from_filename.side_effect = Exception("Persistent Error")
    mock_storage_client.bucket.return_value.blob.return_value = mock_blob

    with patch("pathlib.Path.stat") as mock_stat, patch("time.sleep"):
        mock_stat.return_value.st_size = _MULTIPART_THRESHOLD + 1024

        # Verify that RuntimeError is raised after max_attempts
        with pytest.raises(RuntimeError, match="Upload GCS abandonné"):
            upload_file_to_gcs("b", "p", "n", max_attempts=2)


def test_upload_json_to_gcs(mock_storage_client):
    mock_blob = MagicMock()
    mock_storage_client.bucket.return_value.blob.return_value = mock_blob
    data = {"key": "value"}

    upload_json_to_gcs("bucket", "blob.json", data)

    mock_blob.upload_from_string.assert_called_once()
    args, kwargs = mock_blob.upload_from_string.call_args
    assert json.loads(args[0]) == data
    assert kwargs["content_type"] == "application/json"


def test_download_json_from_gcs_success(mock_storage_client):
    mock_blob = MagicMock()
    mock_blob.exists.return_value = True
    mock_blob.download_as_text.return_value = '{"status": "ok"}'
    mock_storage_client.bucket.return_value.blob.return_value = mock_blob

    result = download_json_from_gcs("bucket", "blob.json")

    assert result == {"status": "ok"}


def test_download_json_from_gcs_not_found(mock_storage_client):
    mock_blob = MagicMock()
    mock_blob.exists.return_value = False
    mock_storage_client.bucket.return_value.blob.return_value = mock_blob

    result = download_json_from_gcs("bucket", "missing.json")

    assert result is None


def test_find_prediction_in_bucket_found(mock_storage_client):
    # Setup list of blobs
    mock_blob_1 = MagicMock()
    mock_blob_1.name = "predictions/123_result.json"
    mock_blob_1.download_as_text.return_value = '{"id": "123"}'

    mock_blob_2 = MagicMock()
    mock_blob_2.name = "predictions/456_result.json"

    mock_storage_client.list_blobs.return_value = [mock_blob_1, mock_blob_2]

    # Execute
    result = find_prediction_in_bucket("bucket", "123")

    # Verify
    assert result == {"id": "123"}
    mock_storage_client.list_blobs.assert_called_with("bucket", prefix="predictions/")


def test_get_latest_blob(mock_storage_client):

    b1 = MagicMock()
    b1.name = "old"
    b1.updated = datetime(2023, 1, 1)

    b2 = MagicMock()
    b2.name = "new"
    b2.updated = datetime(2023, 1, 2)

    mock_storage_client.list_blobs.return_value = [b1, b2]

    result = get_latest_blob("bucket", "prefix/")

    assert result == "new"


def test_get_latest_blob_empty(mock_storage_client):
    mock_storage_client.list_blobs.return_value = []
    result = get_latest_blob("bucket", "prefix/")
    assert result is None


def test_download_blob_to_file(mock_storage_client):
    mock_blob = MagicMock()
    mock_storage_client.bucket.return_value.blob.return_value = mock_blob

    download_blob_to_file("bucket", "remote.path", "local.path")

    mock_blob.download_to_filename.assert_called_once_with("local.path", retry=pytest.any)
