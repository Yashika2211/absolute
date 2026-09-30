import zipfile
from pathlib import Path

from streamline.ingest.download import has_kaggle_credentials, unzip_if_needed


def test_no_credentials(tmp_path: Path) -> None:
    assert not has_kaggle_credentials(home=tmp_path, env={})


def test_credentials_from_env(tmp_path: Path) -> None:
    assert has_kaggle_credentials(home=tmp_path, env={"KAGGLE_API_TOKEN": "x"})
    assert has_kaggle_credentials(home=tmp_path, env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"})
    assert not has_kaggle_credentials(home=tmp_path, env={"KAGGLE_USERNAME": "u"})


def test_credentials_from_files(tmp_path: Path) -> None:
    (tmp_path / ".kaggle").mkdir()
    (tmp_path / ".kaggle" / "kaggle.json").write_text("{}")
    assert has_kaggle_credentials(home=tmp_path, env={})


def test_unzip_if_needed(tmp_path: Path) -> None:
    with zipfile.ZipFile(tmp_path / "events.csv.zip", "w") as zf:
        zf.writestr("events.csv", "timestamp,visitorid,event,itemid,transactionid\n")
    path = unzip_if_needed(tmp_path)
    assert path.read_text().startswith("timestamp")
    assert not (tmp_path / "events.csv.zip").exists()
