from pathlib import Path
import os

ROOT = Path(__file__).parent.parent.resolve()
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
RAW_VIDEOS = DATA / "raw_videos"

direktorijumi = [DATA, PROCESSED, RAW_VIDEOS]

for direktorijum in direktorijumi:
    direktorijum.mkdir(parents=True, exist_ok=False)