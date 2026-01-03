import torch
from torch.utils.data import Dataset
import os
import tifffile as tiff
import numpy as np
import cv2
from pathlib import Path


class CellDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        # Sadece maskesi OLAN resimleri listeye ekleyelim
        self.valid_images = []
        all_img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])

        for img_name in all_img_files:
            # t047.tif -> 047
            file_number = "".join(filter(str.isdigit, img_name))

            # Maske klasöründe bu numaraya sahip bir dosya var mı bak
            # (man_seg047.tif veya t047.tif gibi)
            mask_options = [f"man_seg{file_number}.tif", img_name, f"{file_number}.tif"]
            found = False
            for opt in mask_options:
                if os.path.exists(os.path.join(mask_dir, opt)):
                    self.valid_images.append((img_name, opt))
                    found = True
                    break

        print(
            f"Toplam {len(all_img_files)} resimden {len(self.valid_images)} tanesi etiketli bulundu. Eğitim bunlar üzerinden yapılacak.")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_name, mask_name = self.valid_images[idx]

        image = tiff.imread(os.path.join(self.img_dir, img_name)).astype(np.float32)
        mask = tiff.imread(os.path.join(self.mask_dir, mask_name)).astype(np.float32)

        # Boyut eşitleme
        if image.shape != mask.shape:
            image = cv2.resize(image, (mask.shape[1], mask.shape[0]))

        # Normalize
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-7)
        mask = (mask > 0).astype(np.float32)

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

from model import UNet
import torch.optim as optim

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    img_dir = "data/PhC-C2DH-U373/01"
    mask_dir = "data/PhC-C2DH-U373/01_GT/SEG"
    train_dataset = CellDataset(img_dir=img_dir, mask_dir=mask_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()

    print("U-Net Eğitimi Başlıyor...")
    model.train()

    for epoch in range(0,30,+1):  # Hücreler belirgin olduğu için 20 epoch genelde yeter
        epoch_loss = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/20 - Loss: {epoch_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "unet_cell_seg.pth")
    print("Model 'unet_cell_seg.pth' olarak kaydedildi!")


if __name__ == "__main__":
    train()