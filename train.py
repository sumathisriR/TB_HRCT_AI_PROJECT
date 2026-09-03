from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from models.model import UNet


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_ROOT / "data" / "images"
MASK_DIR = PROJECT_ROOT / "data" / "masks"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "tb_unet.pth"


class TBDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.images = sorted(self.image_dir.glob("*"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]

        # Dataset loading will be added after
        # the approved CT dataset is available.
        raise NotImplementedError(
            "Dataset loading will be enabled after approved data is added."
        )


def main():
    print("TB HRCT AI - Training Pipeline")
    print("-" * 40)

    print("Image folder:", IMAGE_DIR)
    print("Mask folder :", MASK_DIR)

    image_files = list(IMAGE_DIR.glob("*")) if IMAGE_DIR.exists() else []
    mask_files = list(MASK_DIR.glob("*")) if MASK_DIR.exists() else []

    print("Images found:", len(image_files))
    print("Masks found :", len(mask_files))

    if len(image_files) == 0 or len(mask_files) == 0:
        print()
        print("Training not started.")
        print("Approved CT images and corresponding masks are required.")
        return

    # Create dataset
    dataset = TBDataset(IMAGE_DIR, MASK_DIR)

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )

    # Create U-Net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)

    print()
    print("Model:", model.__class__.__name__)
    print("Device:", device)
    print("Dataset size:", len(dataset))

    print()
    print("Training pipeline is ready.")


if __name__ == "__main__":
    main()