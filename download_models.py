#!/usr/bin/env python3
"""Download model weights for thermal-drone-detection."""

import os
import urllib.request
import subprocess

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FILES = {
    "hituav_thermal_yolov8n.pt": "https://raw.githubusercontent.com/baturalpoz06/thermal-object-detection-yolov8/main/hituav_thermal_yolov8n.pt",
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt",
}

GDRIVE = {
    "thermal_sentinel_yolo11s.pt": "1qMIDt8Vol9cJlj0FJl5xwS5ctx_ZTUbo",
}


def download(url, dest):
    print(f"  Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  -> {dest}")


def gdown_file(file_id, dest):
    print(f"  gdown {file_id}")
    subprocess.check_call(["gdown", file_id, "-O", dest])
    print(f"  -> {dest}")


def main():
    for name, url in FILES.items():
        dest = os.path.join(MODELS_DIR, name)
        if os.path.exists(dest):
            print(f"[skip] {name} already exists")
            continue
        try:
            download(url, dest)
        except Exception as e:
            print(f"[error] {name}: {e}")

    for name, fid in GDRIVE.items():
        dest = os.path.join(MODELS_DIR, name)
        if os.path.exists(dest):
            print(f"[skip] {name} already exists")
            continue
        try:
            subprocess.check_call(["pip", "install", "-q", "gdown"])
            gdown_file(fid, dest)
        except Exception as e:
            print(f"[error] {name}: {e}")

    print("\nDone. Models are in models/")


if __name__ == "__main__":
    main()
