import os
import cv2
import random
from pathlib import Path
from tqdm import tqdm

output_summary_path = Path(__file__).resolve().parent.parent / "outputs/sample_count.txt"


def list_images_in_dir(dir_path, valid_exts={".jpg", ".jpeg", ".png", ".bmp"}):
    return [p for p in Path(dir_path).rglob("*") if p.suffix.lower() in valid_exts]


def resize(img, out_path, size):
    resized = cv2.resize(img, size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), resized)


def mirror_resize(input, output, size=(64, 128)):
    input = Path(input)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    images = list_images_in_dir(input)
    # images = random.sample(images, 10000)
    count = 0

    for img_path in tqdm(images, desc="Processing positive samples"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        out_name = output / f"pos_{count:06d}.jpg"
        resize(img, out_name, size)

        # 镜像图
        flipped = cv2.flip(img, 1)
        out_flip_name = output / f"pos_{count+1:06d}.jpg"
        resize(flipped, out_flip_name, size)
        count += 2

    print(f"[Positive] Total processed (with mirror): {count}")


def random_crop(img, crop_ratio=(0.25, 0.5)):
    h, w, _ = img.shape
    ch, cw = int(h * crop_ratio[1]), int(w * crop_ratio[0])
    if h < ch or w < cw:
        return None
    x = random.randint(0, w - cw)
    y = random.randint(0, h - ch)
    return img[y:y+ch, x:x+cw]

def random_crop_patches(input_dir, output_dir, target_size=(64,128), patches_per_image=10):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_images_in_dir(input_dir)
    # images = random.sample(images, 2000)
    collected = 0

    for img_path in tqdm(images, desc="Processing negative samples"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for i in range(patches_per_image):
            patch = random_crop(img, crop_ratio=(0.25, 0.5))
            if patch is None:
                continue
            resized_patch = cv2.resize(patch, target_size)
            out_path = output_dir / f"neg_{collected:06d}.jpg"
            cv2.imwrite(str(out_path), resized_patch)
            collected += 1

    print(f"[Negative] Total negative patches generated: {collected}")
    
    
def count_images(dir_path):
    return len(list_images_in_dir(dir_path))
    
def main():
    dataset_root = Path(__file__).resolve().parent.parent / "dataset-big"
    output_root = Path(__file__).resolve().parent.parent / "data_processed"

    positive_input = dataset_root / "Training set/human"
    negative_input = dataset_root / "Training set/non-human"
    positive_output = output_root / "positive"
    negative_output = output_root / "negative"


    mirror_resize(positive_input, positive_output)

    random_crop_patches(negative_input, negative_output, patches_per_image=10)
    
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary_path, "w") as f:
        pos_count = count_images(positive_output)
        neg_count = count_images(negative_output)
        f.write(f"Positive samples: {pos_count}\n")
        f.write(f"Negative samples: {neg_count}\n")
    print(f"[Summary] Written to {output_summary_path}")
    
    
if __name__ == "__main__":
    main()