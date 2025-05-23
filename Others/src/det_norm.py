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
            yield img[y:y + win_h, x:x + win_w]

def list_images(folder, exts={'.jpg', '.jpeg', '.png', '.bmp'}):
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
        true_positive = 0
        pbar = tqdm(total=len(human_imgs) + len(nonhuman_imgs), desc=f"Threshold {threshold:.2f}")

        for path in human_imgs:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                pbar.update(1)
                continue
            h, w = img.shape
            if h < win_h or w < win_w:
                img = cv2.resize(img, (win_w, win_h))
                f = hog(img, **hog_params)
                if model.decision_function([f])[0] > threshold:
                    true_positive += 1
                else:
                    miss += 1
                pbar.update(1)
                continue

            detected = False
            for patch in sliding_windows(img):
                f = hog(patch, **hog_params)
                if model.decision_function([f])[0] > threshold:
                    detected = True
                    break
            if detected:
                true_positive += 1
            else:
                miss += 1
            pbar.update(1)

        for path in nonhuman_imgs:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                pbar.update(1)
                continue
            h, w = img.shape
            if h < win_w or w < win_h:
                img = cv2.resize(img, (win_w, win_h))
                f = hog(img, **hog_params)
                total_windows += 1
                if model.decision_function([f])[0] > threshold:
                    total_fp += 1
                pbar.update(1)
                continue

            for patch in sliding_windows(img):
                f = hog(patch, **hog_params)
                total_windows += 1
                if model.decision_function([f])[0] > threshold:
                    total_fp += 1
            pbar.update(1)

        miss_rate = miss / len(human_imgs)
        fppw = total_fp / total_windows
        results.append((threshold, fppw, miss_rate))
        pbar.close()

    return results

def plot_det(det_curves, save_path):
    plt.figure(figsize=(8, 6))
    for label, points in det_curves.items():
        fppw = [x[1] for x in points]
        miss = [x[2] for x in points]
        plt.plot(fppw, miss, label=label, marker='o')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("False Positives Per Window (log scale)")
    plt.ylabel("Miss Rate (log scale)")
    plt.title("DET Curve: Effect of Normalization Techniques")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Saved] DET curve at {save_path}")

if __name__ == "__main__":
    thresholds = np.linspace(-1.5, 1.5, 11)

    # === Normalization experiment setup ===
    norms = [
        ("L1", "hog_svm_model_v2_L1norm.pkl"),
        ("L2", "hog_svm_model_v2_L2norm.pkl"),
        ("L2-Hys", "hog_svm_model_v2.pkl"),
    ]

    base = Path(__file__).resolve().parent.parent
    curves = {}


    for name, model_file in norms:
        model_path = base / "models" / model_file
        print(f"[INFO] Evaluating normalization={name}...")
        hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': name,
        }
        res = evaluate_det_curve(model_path, hog_params, thresholds)
        curves[f"norm={name}"] = res

    plot_det(curves, base / "outputs/det_norms2.png")