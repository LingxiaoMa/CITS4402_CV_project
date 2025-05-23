import cv2
import joblib
from pathlib import Path
from skimage.feature import hog
from tqdm import tqdm

# HOG 和窗口参数
win_w, win_h = 64, 128
stride = 16
threshold = 0
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
}

def list_images(path, exts={'.jpg', '.jpeg', '.png', '.bmp'}):
    return [p for p in Path(path).rglob("*") if p.suffix.lower() in exts]

def sliding_windows(img, stride=16):
    h, w = img.shape
    y_positions = list(range(0, h - win_h + 1, stride))
    if (h - win_h) % stride != 0:
        y_positions.append(h - win_h)
    x_positions = list(range(0, w - win_w + 1, stride))
    if (w - win_w) % stride != 0:
        x_positions.append(w - win_w)
    for y in y_positions:
        for x in x_positions:
            yield img[y:y+win_h, x:x+win_w]

def evaluate():
    base = Path(__file__).resolve().parent.parent
    model = joblib.load(base / "models/hog_svm_model_v2.pkl")

    human_dir = base / "dataset-big/Testing set/human"
    nonhuman_dir = base / "dataset-big/Testing set/non-human"
    human_imgs = list_images(human_dir)
    nonhuman_imgs = list_images(nonhuman_dir)

    total_human = len(human_imgs)
    total_nonhuman = len(nonhuman_imgs)
    total_fp = 0
    total_windows = 0
    miss_detected = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    print(f"Evaluating on {total_human} human and {total_nonhuman} non-human images...")

    # 人类图像：图像级 miss rate 和 TP 统计
    for path in tqdm(human_imgs, desc="Human eval"):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        if h < win_h or w < win_w:
            img = cv2.resize(img, (win_w, win_h))
            patch = img
            f = hog(patch, **HOG_PARAMS)
            if model.decision_function([f])[0] > threshold:
                true_positive += 1
            else:
                miss_detected += 1
                false_negative += 1
            continue

        detected = False
        for patch in sliding_windows(img):
            f = hog(patch, **HOG_PARAMS)
            if model.decision_function([f])[0] > threshold:
                detected = True
                break
        if detected:
            true_positive += 1
        else:
            miss_detected += 1
            false_negative += 1

    # 非人类图像：图像级 FP，窗口级 FP/FPPW
    for path in tqdm(nonhuman_imgs, desc="Non-human eval"):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        if h < win_h or w < win_w:
            img = cv2.resize(img, (win_w, win_h))
            patch = img
            f = hog(patch, **HOG_PARAMS)
            total_windows += 1
            if model.decision_function([f])[0] > threshold:
                total_fp += 1
                false_positive += 1
            else:
                true_negative += 1
            continue

        detected = False
        for patch in sliding_windows(img):
            f = hog(patch, **HOG_PARAMS)
            total_windows += 1
            if model.decision_function([f])[0] > threshold:
                total_fp += 1
                detected = True
        if detected:
            false_positive += 1
        else:
            true_negative += 1

    # 计算指标
    miss_rate = miss_detected / total_human
    fppw = total_fp / total_windows
    total_images = total_human + total_nonhuman
    accuracy = (true_positive + true_negative) / total_images
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0

    # 输出结果
    print("\n===== Evaluation Report =====")
    print(f"[Image-level] Miss Rate: {miss_rate:.4f}")
    print(f"[Window-level] FPPW: {fppw:.6f}")
    print(f"[Image-level] Accuracy: {accuracy:.4f}")
    print(f"[Image-level] Precision: {precision:.4f}")
    print(f"[Image-level] Recall: {recall:.4f}")

    # 写入日志
    log_path = base / "outputs/eval_log_v2.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("===== Evaluation Report =====\n")
        f.write(f"[Image-level] Miss Rate: {miss_rate:.4f}\n")
        f.write(f"[Window-level] FPPW: {fppw:.6f}\n")
        f.write(f"[Image-level] Accuracy: {accuracy:.4f}\n")
        f.write(f"[Image-level] Precision: {precision:.4f}\n")
        f.write(f"[Image-level] Recall: {recall:.4f}\n")

if __name__ == "__main__":
    evaluate()