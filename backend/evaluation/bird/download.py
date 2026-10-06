"""Downloads and extracts the BIRD dev set (~330MB) into ./data.

Idempotent — safe to run again; it skips whatever step already has its
output on disk.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path(__file__).resolve().parent / "data"
DEV_ZIP_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dev_json = DATA_DIR / "dev_20240627" / "dev.json"
    if not dev_json.exists():
        zip_path = DATA_DIR / "dev.zip"
        if not zip_path.exists():
            print(f"downloading {DEV_ZIP_URL} ...")
            urlretrieve(DEV_ZIP_URL, zip_path)
        print("extracting dev.zip ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
    else:
        print(f"already present: {dev_json}")

    databases_dir = DATA_DIR / "dev_20240627" / "dev_databases"
    if not databases_dir.exists() or not any(databases_dir.iterdir()):
        nested_zip = DATA_DIR / "dev_20240627" / "dev_databases.zip"
        print("extracting dev_databases.zip ...")
        with zipfile.ZipFile(nested_zip) as zf:
            zf.extractall(DATA_DIR / "dev_20240627")
    else:
        print(f"already present: {databases_dir}")

    macosx = DATA_DIR / "dev_20240627" / "__MACOSX"
    if macosx.exists():
        shutil.rmtree(macosx)

    print("done.")


if __name__ == "__main__":
    main()
