import cv2
import joblib
import numpy as np
from pathlib import Path
from skimage.feature import hog
from tqdm import tqdm

HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
}

win_w , win_h = 64, 128
stride = 16
decision_threshold = 0 

base_path = Path(__file__).resolve().parent.parent
model_path = base_path / "models/hog_svm_model_v1.pkl"
neg_img_dir = base_path / "dataset-big/Training set/non-human"
output_dir = base_path / "data_processed/negative"
output_dir.mkdir(parents=True, exist_ok=True)

# load model
clf = joblib.load(model_path)
count = len(list(output_dir.glob("neg_*.jpg")))
count_original = count


for img_dir in tqdm(list(neg_img_dir.glob("*.*")), desc="Hard Negative Example Mining"):
    img = cv2.imread(str(img_dir), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    
    h,w = img.shape[:2]
    
    # if img is too small, resize
    if h < win_h or w < win_w:
        resized = cv2.resize(img, (win_w, win_h))
        features = hog(resized, **HOG_PARAMS)
        if clf.decision_function([features])[0] > decision_threshold:
            out_path = output_dir / f"neg_{count:06d}.jpg"
            cv2.imwrite(str(out_path), resized)
            count += 1
        continue
    
    # slide window and edge alignment
    y_positions = list(range(0, h - win_h + 1, stride))
    if (h - win_h) % stride != 0:
        y_positions.append(h - win_h)

    x_positions = list(range(0, w - win_w + 1, stride))
    if (w - win_w) % stride != 0:
        x_positions.append(w - win_w)

    for y in y_positions:
        for x in x_positions:
            patch = img[y:y + win_h, x:x + win_w]
            features = hog(patch, **HOG_PARAMS)
            if clf.decision_function([features])[0] > decision_threshold:
                out_path = output_dir / f"neg_{count:06d}.jpg"
                cv2.imwrite(str(out_path), patch)
                count += 1
                

log_path = base_path / "outputs/hard_negative_log.txt"
log_path.parent.mkdir(parents=True, exist_ok=True)
with open(log_path, "w") as log_file:
    log_file.write(f"Hard negative mining completed.\n")
    log_file.write(f"Model used: {model_path.name}\n")
    log_file.write(f"Negative source images: {len(list(neg_img_dir.glob('*.*')))}\n")
    log_file.write(f"Newly add {count-count_original} samples\n")
    log_file.write(f"Final negative sample count: {count}\n")

print(f"硬负样本挖掘完成，共新增样本至编号：{count}")