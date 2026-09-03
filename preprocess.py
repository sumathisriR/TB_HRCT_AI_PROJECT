import nibabel as nib
import numpy as np
from PIL import Image


def load_nifti(file_path):
    nii = nib.load(file_path)
    return nii.get_fdata()


def normalize_ct(volume):
    volume = volume.astype(np.float32)

    min_value = np.min(volume)
    max_value = np.max(volume)

    if max_value == min_value:
        return np.zeros_like(volume)

    return (volume - min_value) / (max_value - min_value)


def preprocess_ct(file_path):
    volume = load_nifti(file_path)
    return normalize_ct(volume)


def preprocess_image(image_file):
    image = Image.open(image_file).convert("L")
    image = image.resize((512, 512))

    image_array = np.array(image, dtype=np.float32) / 255.0

    return image_array