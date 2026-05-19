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
OUT_PATH = ROOT / "data" / "processed" / "analiza_videa_step1.csv"
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

    print("ROI x:", my_roi.x)
    print("ROI y:", my_roi.y)
    x_min = min(my_roi.x).astype(int)
    x_max = max(my_roi.x).astype(int)
    y_min = min(my_roi.y).astype(int)
    y_max = max(my_roi.y).astype(int)
    print("x_min:", x_min.astype(int))
    print("x_max:", x_max.astype(int))
    print("y_min:", y_min.astype(int))
    print("y_max:", y_max.astype(int))

    maska = my_roi.get_mask(gray)

    povrsina = (x_max - x_min)*(y_max - y_min)

    plt.figure()
    plt.imshow(maska)
    plt.show()

    return maska, x_min, x_max, y_min, y_max, povrsina


def analiza_jednog_videa(video_path, maska, x_min, x_max, y_min, y_max, povrsina):

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

    brightness_step = round(fps)
    motion_step = 4
    #L = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    k=0
    prag = 1.18
    brightness_values = []
    laplasian_variance = []
    #motion_between_frames = []
    frame_motion = []
    active_frame = 0
    br_poredjenja = 0
    motion_area_score = []
    pixel_diff_threshold = 0.2
    area_threshold = 0.04

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        sivi_frejm = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bbox_frejm = sivi_frejm[y_min:y_max, x_min:x_max] * maska[y_min:y_max, x_min:x_max]

        brightness_values.append(bbox_frejm.mean())

        if k % brightness_step == 0:
            laplasian_variance.append(cv2.Laplacian(src=bbox_frejm, ddepth=-1, ksize=1).var())

        if k == 0:
            prethodni_frejm = bbox_frejm
            k=1
            continue

        if k % motion_step == 0:
            abs_razlika = abs(bbox_frejm.astype('float') - prethodni_frejm.astype('float'))
            motion_area_score.append(sum(abs_razlika > pixel_diff_threshold) / povrsina)
            frame_motion.append(np.mean(abs_razlika))
            active_frame += frame_motion[-1] > prag
            br_poredjenja += 1

        #TODO event detection -- izracunam za poslednjih 15 frejmova statistike

        prethodni_frejm = bbox_frejm

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
        "active_frame_ratio": active_frame/br_poredjenja,
        "mean_motion_area_score": np.mean(motion_area_score)
    }
    cap.release()
    return row


def analiza_svih_videa(video_folder, maska, x_min, x_max, y_min, y_max, povrsina):
    rezultati = []
    for video_path in video_folder.iterdir():

        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        row = analiza_jednog_videa(video_path, maska, x_min, x_max, y_min, y_max, povrsina)
        if row is not None:
            rezultati.append(row)

    return rezultati


def main():
    start_time = timer()

    start_time1 = timer()
    video = VIDEO_PATH / "bez_pokreta1.mp4"
    maska, x_min, x_max, y_min, y_max, povrsina = roi_maska(video)
    end_time1 = timer()

    """
    filename = video.name
    cap = cv2.VideoCapture(str(video))

    ret, frame = cap.read()

    sivi_frejm = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bbox_frejm = sivi_frejm[y_min:y_max, x_min:x_max] * maska[y_min:y_max, x_min:x_max]
    print(povrsina)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(maska)
    plt.subplot(1,2,2)
    plt.imshow(bbox_frejm)
    plt.show()
    """

    start_time2 = timer()
    rezultati = analiza_svih_videa(VIDEO_PATH, maska, x_min, x_max, y_min, y_max, povrsina)
    df = pd.DataFrame(rezultati)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    end_time2 = timer()

    end_time = timer()
    print(f"Gascina sve radi kako treba, ukupno vreme potrebno za izvrsavanje {end_time-start_time} \n"
          f"vreme potrebno za biranje ROI {end_time1-start_time1}\n"
          f"vreme potrebno za analiza svih videa {end_time2-start_time2}\n")


if __name__ == "__main__":
    main()
