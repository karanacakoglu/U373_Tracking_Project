import cv2
import numpy as np
import tifffile as tiff
import os


def get_image_paths(data_path):
    # Klasördeki tüm .tif dosyalarını sıralı şekilde al
    files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.tif')])
    return files


def preprocess_frame(image_path):
    # 1. Tif dosyasını orijinal derinliğinde oku
    img = tiff.imread(image_path)

    # 2. Normalize et (0-255 arasına çek ki OpenCV işleyebilsin)
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. CLAHE uygula (Hücre çeperlerini belirginleştirir)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_norm)

    return img_enhanced