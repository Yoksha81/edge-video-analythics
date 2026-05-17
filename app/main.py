from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from scipy import ndimage
from timeit import default_timer as timer
from skimage import io,color
import matplotlib
matplotlib.use("TkAgg")   # bitno: pre import pyplot
import matplotlib.pyplot as plt
from roipoly import RoiPoly


ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = ROOT / "data" / "raw_videos"
OUT_PATH = ROOT / "data" / "processed" / "analiza_videa.csv"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi"}

def roi_maska(video_path):
    filename = video_path.name
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print(f"Video {filename} nece da se otvori.")
        return None

    ret, frame = capture.read()

    if not ret:
        print(f"Ne moze da se ucita frejm")
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(15,15))
    plt.imshow(gray)
    plt.axis('off')

    my_roi = RoiPoly(color='r')
    maska = my_roi.get_mask(gray)

    plt.figure()
    plt.imshow(maska)
    plt.show()

    return maska


def analiza_jednog_videa(video_path, maska):

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

    step = round(fps)
    L = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    k=0
    prag = 0.3
    brightness_values = []
    laplasian_variance = []
    motion_between_frames = []
    frame_motion = []
    active_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        sivi_frejm = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) * maska

        if k % step == 0:
            laplasian_variance.append(np.var(ndimage.convolve(sivi_frejm.astype('float'),L,mode='mirror')))

        brightness_values.append(sivi_frejm.mean())

        if k == 0:
            prethodni_frejm = sivi_frejm
            k=1
            continue

        trenutni_frejm = sivi_frejm

        pokret = np.mean(abs(trenutni_frejm.astype('float') - prethodni_frejm.astype('float')))
        motion_between_frames.append(pokret)
        frame_motion.append(pokret/255)

        active_frame += frame_motion[-1] > prag

        prethodni_frejm = sivi_frejm

        k+=1

    avg_brightness = sum(brightness_values) / len(brightness_values)

    row = {
        "filename": filename,
        "fps": fps,
        "ukupno_frejmova": ukupno_frejmova,
        "trajanje_videa": trajanje_videa,
        "sirina": sirina,
        "visina": visina,
        "avg_brightness": avg_brightness,
        "sharpness_score": np.mean(laplasian_variance),
        "motion_score": np.mean(frame_motion),
        "min_frame_motion": np.min(frame_motion),
        "max_frame_motion": np.max(frame_motion),
        "active_frame_ratio": active_frame/ukupno_frejmova
    }
    cap.release()
    return row


def analiza_svih_videa(video_folder, maska):
    rezultati = []
    for video_path in video_folder.iterdir():

        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        row = analiza_jednog_videa(video_path, maska)
        if row is not None:
            rezultati.append(row)

    return rezultati


def main():
    start_time = timer()
    video = VIDEO_PATH / "bez_pokreta1.mp4"
    maska = roi_maska(video)
    rezultati = analiza_svih_videa(VIDEO_PATH, maska)
    df = pd.DataFrame(rezultati)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    end_time = timer()
    print(f"Gascina sve radi kako treba, vreme potrebno za izvrsavanje {end_time-start_time}")


if __name__ == "__main__":
    main()
