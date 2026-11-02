"""Fetch the raw inputs that are not stored in the repository and verify them.

usage (from the repository root):  python dataset/download_data.py

- UCI Sales Transactions Weekly (CC BY 4.0) is stored in the repository: dataset/raw/UCI_SalesTransactionsWeekly/source.zip
- M5 Forecasting - Accuracy: downloaded from its Zenodo copy (https://doi.org/10.5281/zenodo.12636070).
- Rossmann Store Sales: downloaded with the Kaggle API (`pip install kaggle`, an API token in ~/.kaggle/kaggle.json,
  and the competition rules accepted at https://www.kaggle.com/competitions/rossmann-store-sales/rules).
"""
import hashlib, shutil, subprocess, sys, urllib.request, zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"
SHA256 = {
    "UCI_SalesTransactionsWeekly/source.zip": "dededa20f199af8f2bb96184bb1696befda902849a700bde283fe763afc90ba9",
    "m5/m5-forecasting-accuracy.zip": "0349ba38a2efd30d0f5acc6394c1110e140e1a990c650d7b5ca44c5b25dd12f5",
    "rossmann/train.csv": "f6e4597c142d7d909a13d53b68a8e85c00b9a4c7b5ff40adbb37d6829cc1f4cc",
    "rossmann/store.csv": "f56bd124a2849489e6bbb5c000f5fc9640204355e316475c918ae4d089afb344",
}
M5_URL = "https://zenodo.org/records/12636070/files/m5-forecasting-accuracy.zip?download=1"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(rel):
    path = RAW / rel
    if not path.exists():
        print(f"missing   {rel}")
        return False
    ok = sha256(path) == SHA256[rel]
    print(f"{'ok' if ok else 'DIFFERS':9} {rel}")
    return ok


def get_m5():
    target = RAW / "m5" / "m5-forecasting-accuracy.zip"
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print("downloading M5 from Zenodo (48 MB) ...")
    with urllib.request.urlopen(M5_URL) as r, open(target, "wb") as f:
        shutil.copyfileobj(r, f)


def get_rossmann():
    folder = RAW / "rossmann"
    if (folder / "train.csv").exists() and (folder / "store.csv").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    print("downloading Rossmann Store Sales with the Kaggle API ...")
    try:
        subprocess.run(["kaggle", "competitions", "download", "-c", "rossmann-store-sales", "-p", str(folder)], check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"Kaggle download failed ({e}). Download train.csv and store.csv manually from "
              "https://www.kaggle.com/competitions/rossmann-store-sales/data into dataset/raw/rossmann/.")
        return
    with zipfile.ZipFile(folder / "rossmann-store-sales.zip") as z:
        for name in ("train.csv", "store.csv"):
            z.extract(name, folder)


if __name__ == "__main__":
    get_m5()
    get_rossmann()
    results = [check(rel) for rel in SHA256]
    if not all(results):
        print("Some files are missing or differ from the versions used for the stored results "
              "(a different but equivalent copy of a public file can differ in line endings).")
        sys.exit(1)
    print("all raw inputs are in place")
