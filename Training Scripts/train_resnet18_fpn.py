import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Local Config (Labeled training data)
DATASET_ROOT = Path(r"D:\JetRacer\Datasets\Training\segmentation_files")
IMAGESETS_DIR = DATASET_ROOT / "ImageSets" / "Segmentation"
JPEG_DIR = DATASET_ROOT / "JPEGImages"
MASK_DIR = DATASET_ROOT / "SegmentationClass"

OUT_DIR = Path(r"D:\JetRacer\Datasets\Training\seg_train\outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 320
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4

# NVIDIA GPU or CPU depending on availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# DATASET
# ----------------------------
class PaperDataset(Dataset):
    def __init__(self, split="train"):
        split_file = IMAGESETS_DIR / f"{split}.txt"
        self.ids = [x.strip() for x in split_file.read_text().splitlines() if x.strip()]
        
        self.transform = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5) if split == "train" else A.NoOp(),
            A.RandomBrightnessContrast(p=0.3) if split == "train" else A.NoOp(),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        image = np.array(Image.open(JPEG_DIR / f"{sample_id}.jpg").convert("RGB"))
        mask = np.array(Image.open(MASK_DIR / f"{sample_id}.png").convert("RGB"))
        
        # Convert RGB mask to class indices (0=background, 1=paper)
        mask = (np.any(mask > 0, axis=2)).astype(np.int64)
        
        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"].long()


# METRICS & LOOPS
def compute_iou(logits, targets):
    preds = torch.argmax(logits, dim=1)
    intersection = (preds & targets).sum().float()
    union = (preds | targets).sum().float()
    return (intersection + 1e-6) / (union + 1e-6)

def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    losses, ious = [], []

    for images, masks in tqdm(loader, leave=False):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        if is_train: optimizer.zero_grad()
        
        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, masks)
            if is_train:
                loss.backward()
                optimizer.step()
        
        losses.append(loss.item())
        ious.append(compute_iou(logits, masks).item())
        
    return np.mean(losses), np.mean(ious)

# MAIN EXECUTION
def main():
    print(f"Training on: {DEVICE}")
    
    train_loader = DataLoader(PaperDataset("train"), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PaperDataset("val"), batch_size=1, shuffle=False)

    model = smp.Unet(
        encoder_name="resnet18", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=2
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_iou = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Training
        t_loss, t_iou = run_epoch(model, train_loader, criterion, optimizer)
        # Validation - Added criterion here to fix the TypeError
        v_loss, v_iou = run_epoch(model, val_loader, criterion)
        
        print(f"Epoch {epoch:02d} | Train Loss: {t_loss:.4f} | Val IoU: {v_iou:.4f}")
        
        if v_iou > best_iou:
            best_iou = v_iou
            torch.save(model.state_dict(), OUT_DIR / "best_model.pth")
            print(f"  --> Saved new best model (IoU: {v_iou:.4f})")

    # EXPORT TO ONNX
    print("\nExporting to Jetson-compatible ONNX...")
    model.load_state_dict(torch.load(OUT_DIR / "best_model.pth"))
    model.eval()

    # Shape for Jetson compatibility
    # For some reason there has to be dummy input or else it won't work.
    # TODO: Further investigate this
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    onnx_path = OUT_DIR / "paper_resnet18_unet.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input_0"],
        output_names=["output_0"],
        opset_version=11, 
        do_constant_folding=True
    )

    (OUT_DIR / "labels.txt").write_text("background\npaper\n")
    print(f"Success! Model saved to: {onnx_path}")

if __name__ == "__main__":
    main()