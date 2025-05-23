import os
import cv2
import random
from pathlib import Path
from tqdm import tqdm

output_summary_path = Path(__file__).resolve().parent.parent / "outputs/sample_count.txt" # Set the output path


def list_images_in_dir(dir_path, valid_exts={".jpg", ".jpeg", ".png", ".bmp"}):
    # Traverse the directory and return a list of all valid image file paths
    return [p for p in Path(dir_path).rglob("*") if p.suffix.lower() in valid_exts]


def resize(img, out_path, size):
    # Resize the image and save it to the specified path.
    resized = cv2.resize(img, size)
    out_path.parent.mkdir(parents=True, exist_ok=True) # Ensure that the directory for saving the image exists, if it does not, create it automatically
    cv2.imwrite(str(out_path), resized) # Save the image to the directory


def mirror_resize(input, output, size=(64, 128)):
    # Resize all images in the specified directory and generate their horizontally flipped versions.
    input = Path(input)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True) 

    images = list_images_in_dir(input) # Get all images path in the directory
    count = 0

    for img_path in tqdm(images, desc="Processing positive samples"):
        img = cv2.imread(str(img_path))
        if img is None: # If the read process fail, go to next loop
            continue
            
        out_name = output / f"pos_{count:06d}.jpg" # Generate the save path for the resized image.
        resize(img, out_name, size) # Resize and save the image

        flipped = cv2.flip(img, 1) # Flip the image
        out_flip_name = output / f"pos_{count+1:06d}.jpg" # Generate the save path for the and flipped resized image.
        resize(flipped, out_flip_name, size) # Resize and save the image
        count += 2 # +2 mean we add origin image and flipped image

    print(f"[Positive] Total processed (with mirror): {count}")


def random_crop(img, crop_ratio=(0.25, 0.5)):
    # Randomly crop a portion of the image, with the crop ratio specified by crop_ratio.
    h, w, _ = img.shape # Get the height and width of the image 
    ch, cw = int(h * crop_ratio[1]), int(w * crop_ratio[0]) # Calculate the height and width of the cropping area based on the cropping ratio
    if h < ch or w < cw: # If the image itself is smaller than the cropping area, return None
        return None
    # Randomly select a point for cropping
    x = random.randint(0, w - cw) 
    y = random.randint(0, h - ch)
    return img[y:y+ch, x:x+cw] # Crop and return

def random_crop_patches(input_dir, output_dir, target_size=(64,128), patches_per_image=10):
    # Randomly crop several patches from each image, resize them, and save.
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_images_in_dir(input_dir) # Get the file paths of all image files in the input directory
    collected = 0 # Use it to name the image

    for img_path in tqdm(images, desc="Processing negative samples"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for i in range(patches_per_image):
            patch = random_crop(img, crop_ratio=(0.25, 0.5))
            if patch is None: # If the image itself is smaller than the cropping area, go to next image
                continue
            resized_patch = cv2.resize(patch, target_size) # Resize the patch to the target size.
            out_path = output_dir / f"neg_{collected:06d}.jpg" # Every image have a unique number
            cv2.imwrite(str(out_path), resized_patch)
            collected += 1 

    print(f"[Negative] Total negative patches generated: {collected}")
    
    
def count_images(dir_path):
    # Count the number of images in the directory.
    return len(list_images_in_dir(dir_path))
    
def main():

    # Define the input output directory.
    dataset_root = Path(__file__).resolve().parent.parent / "dataset-big"
    output_root = Path(__file__).resolve().parent.parent / "data_processed"
    
    positive_input = dataset_root / "Training set/human"
    negative_input = dataset_root / "Training set/non-human"
    positive_output = output_root / "positive"
    negative_output = output_root / "negative"

    # Process positive samples: resize and mirror.
    mirror_resize(positive_input, positive_output)
    # Process negative samples: randomly crop patches.
    random_crop_patches(negative_input, negative_output, patches_per_image=10)
    
    # Make a summary report
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary_path, "w") as f:
        pos_count = count_images(positive_output)
        neg_count = count_images(negative_output)
        f.write(f"Positive samples: {pos_count}\n")
        f.write(f"Negative samples: {neg_count}\n")
    print(f"[Summary] Written to {output_summary_path}")
    
    
if __name__ == "__main__":
    main()
