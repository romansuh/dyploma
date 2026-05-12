"""Download the LogHub ZooKeeper datasets into `data/raw/`.

Fetches:
  1. Three 2k-sample files (raw .log + structured CSV + templates CSV) from the
     logpai/loghub GitHub repository (raw.githubusercontent.com).
  2. The full Zookeeper.tar.gz from Zenodo record 8196385, verifies its MD5
     checksum, and extracts `Zookeeper.log` into `data/raw/`.

The script is idempotent: any file already present and non-empty is skipped.
On a checksum mismatch the corrupt tarball is deleted and the run aborts.
"""

from __future__ import annotations

import hashlib
import sys
import tarfile
import time
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"

LOGHUB_BASE_URL = "https://raw.githubusercontent.com/logpai/loghub/master/Zookeeper"
LOGHUB_2K_FILES: tuple[str, ...] = (
    "Zookeeper_2k.log",
    "Zookeeper_2k.log_structured.csv",
    "Zookeeper_2k.log_templates.csv",
)

ZENODO_TAR_URL = "https://zenodo.org/records/8196385/files/Zookeeper.tar.gz?download=1"
ZENODO_TAR_NAME = "Zookeeper.tar.gz"
EXPECTED_MD5 = "11458b4f9fa0911c238f5c1189d4f2d4"
FULL_LOG_NAME = "Zookeeper.log"

CHUNK_SIZE = 8192
MAX_RETRIES = 3


def download_with_retry(url: str, dest: Path, max_retries: int = MAX_RETRIES) -> None:
    """Stream `url` to `dest` with a tqdm bar, retrying on transient failures."""
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("wb") as f, tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            return
        except (requests.RequestException, OSError) as e:
            last_error = e
            # Exponential backoff: 2s, 4s, 8s on attempts 0, 1, 2.
            wait = 2 ** (attempt + 1)
            print(
                f"  [warn] attempt {attempt + 1}/{max_retries} for {url} failed: "
                f"{e!r}; retrying in {wait}s",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError(f"failed to download {url} after {max_retries} attempts") from last_error


def verify_md5(path: Path, expected: str) -> None:
    """Compute streaming MD5 of `path` and raise if it does not match `expected`."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"MD5 mismatch for {path.name}: expected {expected}, got {actual}; "
            "the corrupt file has been removed"
        )


def extract_zookeeper_log(tar_path: Path, dest_dir: Path) -> Path:
    """Extract only `Zookeeper.log` from the archive; reject path-traversal entries."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / FULL_LOG_NAME
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(FULL_LOG_NAME) and member.isfile():
                if Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                    raise RuntimeError(f"refusing to extract suspicious entry: {member.name}")
                src = tar.extractfile(member)
                if src is None:
                    raise RuntimeError(f"cannot read {member.name} from {tar_path}")
                with src, target.open("wb") as out:
                    while True:
                        chunk = src.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        out.write(chunk)
                return target
    raise RuntimeError(f"{FULL_LOG_NAME} not found inside {tar_path}")


def is_present(path: Path) -> bool:
    """Return True if `path` exists and is non-empty (idempotency check)."""
    return path.exists() and path.stat().st_size > 0


def fetch_2k_files() -> None:
    """Download the three 2k-sample files unless already present."""
    for name in LOGHUB_2K_FILES:
        dest = DATA_RAW / name
        if is_present(dest):
            print(f"[skip] {name} already present ({dest.stat().st_size} bytes)")
            continue
        url = f"{LOGHUB_BASE_URL}/{name}"
        print(f"[get ] {url}")
        download_with_retry(url, dest)


def fetch_full_log() -> None:
    """Download the full archive, verify checksum, and extract Zookeeper.log."""
    full_log = DATA_RAW / FULL_LOG_NAME
    if is_present(full_log):
        print(f"[skip] {FULL_LOG_NAME} already present ({full_log.stat().st_size} bytes)")
        return
    tar_path = DATA_RAW / ZENODO_TAR_NAME
    if not is_present(tar_path):
        print(f"[get ] {ZENODO_TAR_URL}")
        download_with_retry(ZENODO_TAR_URL, tar_path)
    print(f"[md5 ] verifying {tar_path.name}")
    verify_md5(tar_path, EXPECTED_MD5)
    print(f"[tar ] extracting {FULL_LOG_NAME} from {tar_path.name}")
    extract_zookeeper_log(tar_path, DATA_RAW)


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    fetch_2k_files()
    fetch_full_log()
    print("\nFinal contents of data/raw:")
    for p in sorted(DATA_RAW.iterdir()):
        if p.is_file():
            print(f"  {p.name:<40} {p.stat().st_size:>12} bytes")


if __name__ == "__main__":
    main()
