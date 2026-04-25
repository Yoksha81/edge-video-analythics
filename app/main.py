from pathlib import Path
import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = ROOT / "data" / "raw_videos"
OUT_PATH = ROOT / "data" / "processed" / "analiza_videa.csv"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi"}

def analiza_svih_videa(video_folder):
    rezultati = []
    for video_path in video_folder.iterdir():

        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        row = analiza_jednog_videa(video_path)
        if row is not None:
            rezultati.append(row)

    return rezultati

def analiza_jednog_videa(video_path):

    filename = video_path.name
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Video {filename} nece da se otvori. :(")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        print(f"FPS je 0 za video: {filename}")
        cap.release()
        return None

    ukupno_frejmova = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    trajanje_videa = ukupno_frejmova / fps
    sirina = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    visina = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    row = {
        "filename": filename,
        "fps": fps,
        "ukupno_frejmova": ukupno_frejmova,
        "trajanje_videa": trajanje_videa,
        "sirina": sirina,
        "visina": visina
    }
    cap.release()
    return row

def main():
    rezultati = analiza_svih_videa(VIDEO_PATH)
    df = pd.DataFrame(rezultati)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print("Gascina sve radi kako treba")

if __name__ == "__main__":
    main()
