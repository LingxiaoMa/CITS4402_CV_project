import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import os
import numpy as np
import cv2
from pathlib import Path
from skimage.feature import hog
import joblib
import pandas as pd

# Sliding window and HOG parameters
win_w, win_h = 64, 128
stride = 16
threshold = 0
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
}

PYRAMID_SCALE = 1.5
PYRAMID_MIN_SIZE = (win_w, win_h)

class ImageGUI:
    def __init__(self, master):
        # The main GUI window
        self.master = master
        self.master.title("GUI")

        self.frame = tk.Frame(self.master)
        self.frame.pack(expand=True, padx=10, pady=10)
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        self.border = tk.Frame(self.frame, borderwidth=2, relief="groove")
        self.border.grid(row=0, column=0, sticky="nsew")

        self.canvas = tk.Canvas(self.border, height=370, width=700)
        self.canvas.grid(row=0, column=0, columnspan=3, sticky="nsew")

        self.scrollbar = tk.Scrollbar(self.border, orient="horizontal", command=self.canvas.xview)
        self.scrollbar.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.canvas.configure(xscrollcommand=self.scrollbar.set)

        self.image_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.image_frame, anchor="nw")

        self.button_frame = tk.Frame(self.border)
        self.button_frame.grid(row=2, column=0, columnspan=3, pady=10)

        self.load_button = tk.Button(self.button_frame, text="Load Folder", command=self.load_folder)
        self.load_button.pack(side=tk.LEFT, padx=70)

        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.predict)
        self.predict_button.pack(side=tk.LEFT, padx=70)

        self.original_images = []
        self.image_paths = [] 
        base = Path(__file__).resolve().parent
        self.model = joblib.load(base / "Others/models/hog_svm_model_v2.pkl")

    def load_folder(self):
        # Open a dialog to select a folder and load all supported images
        folder_path = filedialog.askdirectory(title="Select Folder")
        if not folder_path:
            return

        for widget in self.image_frame.winfo_children():
            widget.destroy()
        self.original_images.clear()
        self.image_paths.clear()
        # Get all image files in the folder
        image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)
                       if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        # Display each image in the GUI
        for image_file in image_files:
            img = Image.open(image_file)

            fixed_height = 300
            width, height = img.size
            new_width = int((fixed_height / height) * width)
            img = img.resize((new_width, fixed_height))

            photo = ImageTk.PhotoImage(img)

            image_container = tk.Frame(self.image_frame)
            image_container.pack(side=tk.LEFT, padx=10, pady=5)

            filename = os.path.basename(image_file)
            filename_label = tk.Label(image_container, text=filename, wraplength=150, justify="center")
            filename_label.pack()

            image_label = tk.Label(image_container, image=photo)
            image_label.image = photo
            image_label.pack()

            self.original_images.append(img)
            self.image_paths.append(image_file)

        self.image_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def sliding_windows(self, img, stride=16):
        # Generator to yield sliding window patches from the image
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

    def pyramid(self, image, scale=PYRAMID_SCALE, min_size=PYRAMID_MIN_SIZE):
        # Generator for image pyramid, yielding downscaled versions of the image
        yield image

        while True:
            h, w = image.shape[:2]
            new_w = int(w / scale)
            new_h = int(h / scale)
            if new_w < min_size[1] or new_h < min_size[0]:
                break

            image = cv2.resize(image, (new_w, new_h))
            yield image

    def predict(self):
        # Predict human/non-human for each loaded image and display results
        if not self.original_images:
            return
        # Clear previous displayed images
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        predictions = []

        for img_index, image_path in enumerate(self.image_paths):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            detected = False
            
            # Use image pyramid and sliding window to detect humans
            for resized_img in self.pyramid(img, PYRAMID_SCALE):
                h, w = resized_img.shape
                if h < win_h or w < win_w:
                    break

                for patch in self.sliding_windows(resized_img):
                    f = hog(patch, **HOG_PARAMS)
                    if self.model.decision_function([f])[0] > threshold:
                        detected = True
                        break
                if detected:
                    break

            result = 1 if detected else 0
            predictions.append({"filename": os.path.basename(image_path), "prediction": result})
            # Display the image and prediction result
            original_img = Image.open(image_path)
            fixed_height = 300
            width, height = original_img.size
            new_width = int((fixed_height / height) * width)
            original_img_resized = original_img.resize((new_width, fixed_height))
            photo = ImageTk.PhotoImage(original_img_resized)

            image_container = tk.Frame(self.image_frame)
            image_container.pack(side=tk.LEFT, padx=10, pady=5)

            filename_label = tk.Label(image_container, text=os.path.basename(image_path), wraplength=150, justify="center")
            filename_label.pack()

            image_label = tk.Label(image_container, image=photo)
            image_label.image = photo
            image_label.pack()

            result_label = tk.Label(image_container, text="Human" if detected else "Non-human", wraplength=150, justify="center")
            result_label.pack()
        # Save predictions to an Excel file
        df = pd.DataFrame(predictions)
        df.to_excel("predictions.xlsx", index=False)

        self.image_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))


if __name__ == "__main__":
    root = tk.Tk()
    gui = ImageGUI(root)
    root.mainloop()
