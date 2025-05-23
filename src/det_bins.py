import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.feature import hog
from tqdm import tqdm

win_w, win_h = 64, 128
stride = 16


def sliding_windows(img):
    h, w = img.shape
    y_positions = list(range(0, h - win_h + 1, stride))
    if (h - win_h) % stride != 0:
        y_positions.append(h - win_h)
    x_positions = list(range(0, w - win_w + 1, stride))
    if (w - win_w) % stride != 0:
        x_positions.append(w - win_w)
    for y in y_positions:
        for x in x_positions:
            yield img[y : y + win_h, x : x + win_w]


def list_images(folder, exts={".jpg", ".jpeg", ".png", ".bmp"}):
    return [p for p in Path(folder).rglob("*") if p.suffix.lower() in exts]


def evaluate_det_curve(model_path, hog_params, thresholds):
    base = Path(__file__).resolve().parent.parent
    model = joblib.load(model_path)

    human_imgs = list_images(base / "dataset-big/Testing set/human")
    nonhuman_imgs = list_images(base / "dataset-big/Testing set/non-human")

    results = []

    for threshold in thresholds:
        miss = 0
        total_fp = 0
        total_windows = 0

        # Evaluate human
        for path in human_imgs:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape
            if h < win_h or w < win_w:
                img = cv2.resize(img, (win_w, win_h))
                patch = img
                f = hog(patch, **hog_params)
                if model.decision_function([f])[0] > threshold:
                    continue
                else:
                    miss += 1
                continue

            detected = False
            for patch in sliding_windows(img):
                f = hog(patch, **hog_params)
                if model.decision_function([f])[0] > threshold:
                    detected = True
                    break
            if not detected:
                miss += 1

        # Evaluate non-human
        for path in nonhuman_imgs:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape
            if h < win_h or w < win_w:
                img = cv2.resize(img, (win_w, win_h))
                patch = img
                f = hog(patch, **hog_params)
                total_windows += 1
                if model.decision_function([f])[0] > threshold:
                    total_fp += 1
                continue

            for patch in sliding_windows(img):
                f = hog(patch, **hog_params)
                total_windows += 1
                if model.decision_function([f])[0] > threshold:
                    total_fp += 1

        miss_rate = miss / len(human_imgs)
        fppw = total_fp / total_windows
        results.append((threshold, fppw, miss_rate))

    return results


def plot_det(det_curves, save_path):
    plt.figure(figsize=(8, 6))
    for label, points in det_curves.items():
        fppw = [x[1] for x in points]
        miss = [x[2] for x in points]
        plt.plot(fppw, miss, label=label, marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("False Positives Per Window (log scale)")
    plt.ylabel("Miss Rate (log scale)")
    plt.title("DET Curve: Effect of Orientation Bins")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Saved] DET curve at {save_path}")


if __name__ == "__main__":
    thresholds = np.linspace(-1.5, 1.5, 11)

    base = Path(__file__).resolve().parent.parent
    model_bin6 = base / "models/hog_svm_model_bin6_v2.pkl"
    model_bin9 = base / "models/hog_svm_model_v2.pkl"
    model_bin12 = base / "models/hog_svm_model_bin12_v2.pkl"
    

    hog_bin9 = {
        "orientations": 9,
        "pixels_per_cell": (8, 8),
        "cells_per_block": (2, 2),
        "block_norm": "L2-Hys",
    }

    hog_bin12 = hog_bin9.copy()
    hog_bin6 = hog_bin9.copy()
    hog_bin12["orientations"] = 12
    hog_bin6["orientations"] = 6

    print("[INFO] Evaluating bin=6 model...")
    res_6 = evaluate_det_curve(model_bin6, hog_bin6, thresholds)

    print("[INFO] Evaluating bin=9 model...")
    res_9 = evaluate_det_curve(model_bin9, hog_bin9, thresholds)

    print("[INFO] Evaluating bin=12 model...")
    res_12 = evaluate_det_curve(model_bin12, hog_bin12, thresholds)

    curves = {"bin=6": res_6, "bin=9" : res_9 ,"bin=12": res_12}

    plot_det(curves, base / "outputs/det_bins.png")
