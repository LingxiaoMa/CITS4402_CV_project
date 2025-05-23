import os
import cv2
import joblib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.svm import LinearSVC
from skimage.feature import hog
from sklearn.metrics import classification_report

# HOG feature extraction parameters
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
}

def list_images(dir_path, valid_exts={".jpg", ".jpeg", ".png", ".bmp"}):
    # Recursively find all image files in a directory with valid extensions.
    return [p for p in Path(dir_path).rglob("*") if p.suffix.lower() in valid_exts]

def extract_features(image, label, size=(64,128)):
    # Extract HOG features from a list of images and assign them the given label.
    features = []
    labels = []
    for path in tqdm(image, desc=f"Extracting HOG for label {label}"):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, size)
        hog_feat = hog(img, **HOG_PARAMS)
        features.append(hog_feat)
        labels.append(label)
    return features, labels

def main():
    base_path = Path(__file__).resolve().parent.parent
    pos_dir = base_path / "data_processed/positive"
    neg_dir = base_path / "data_processed/negative"
    model_path = base_path / "models/hog_svm_model_v1.pkl"
    log_path = base_path / "outputs/train_log.txt"
    
    pos_images = list_images(pos_dir)
    neg_images = list_images(neg_dir)
    
    print(f"Loading {len(pos_images)} positive and {len(neg_images)} negative samples...")
    
    # Extract HOG features and labels for both positive and negative samples
    pos_feats, pos_labels = extract_features(pos_images, label=1)
    neg_feats, neg_labels = extract_features(neg_images, label=0)
    
    # Combine features and labels for the model training
    X = np.array(pos_feats + neg_feats)
    y = np.array(pos_labels + neg_labels)
    
    print("[Training] Fitting LinearSVC...")
    clf = LinearSVC(verbose=1, max_iter=10000)
    clf.fit(X, y)
    
    joblib.dump(clf, model_path)
    print(f"[Model Saved] {model_path}")
    
    # Evaluate the model on the training data
    preds = clf.predict(X)
    report = classification_report(y, preds, digits=4)
    print(report)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(report)
        
if __name__ == "__main__":
    main()