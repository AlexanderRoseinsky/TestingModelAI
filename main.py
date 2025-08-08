import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.preprocessing import StandardScaler
import argparse

from torchvision.models import resnext50_32x4d

warnings.filterwarnings("ignore", message="X does not have valid feature names")

DATASET_ROOT = "dataset_v1"
IMG_DIR = "images"
LABEL_DIR = "labels"
TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, IMG_DIR, "train")
VAL_IMG_DIR = os.path.join(DATASET_ROOT, IMG_DIR, "val")
TEST_IMG_DIR = os.path.join(DATASET_ROOT, IMG_DIR, "test")
TRAIN_LABEL_PATH = os.path.join(DATASET_ROOT, LABEL_DIR, "train", "_train_annotations.csv")
VAL_LABEL_PATH = os.path.join(DATASET_ROOT, LABEL_DIR, "val", "_val_annotations.csv")
TEST_LABEL_PATH = os.path.join(DATASET_ROOT, LABEL_DIR, "test", "_test_annotations.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class GeoPoseDataset(Dataset):
    def __init__(self, image_dir, label_path, transform=None, scaler=None):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(label_path)
        self.transform = transform
        self.scaler = scaler

        required_cols = ['filename', 'lat', 'lon', 'alt']
        if not all(col in self.labels_df.columns for col in required_cols):
            raise ValueError(f"CSV должен содержать колонки: {required_cols}")
        if self.labels_df[required_cols[1:]].isna().any().any():
            print("Внимание: найдены NaN в координатах. Удаляем...")
            self.labels_df.dropna(subset=['lat', 'lon', 'alt'], inplace=True)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_name = row['filename']
        img_path = os.path.join(self.image_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Изображение не найдено: {img_path}") from e

        if self.transform:
            image = self.transform(image)

        target = np.array([row['lat'], row['lon'], row['alt']], dtype=np.float32)

        if self.scaler is not None:
            df = pd.DataFrame([target], columns=['lat', 'lon', 'alt'])
            target = self.scaler.transform(df)[0]

        return image, torch.tensor(target, dtype=torch.float32)

def get_model(output_size=3):
    model = resnext50_32x4d(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, output_size)
    return model.to(device)

from tqdm import tqdm


def train_model(model, train_loader, val_loader, num_epochs=25, lr=1e-4):
    criterion = nn.MSELoss()

    # Оптимизатор
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Косинусное затухание: плавно уменьшает lr
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    pin_memory = device.type == 'cuda'

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (images, targets) in progress_bar:
            images = images.to(device, non_blocking=pin_memory)
            targets = targets.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            avg_loss = running_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{avg_loss:.4f}'
            })

            if batch_idx % 50 == 0:
                print(f"\nEpoch [{epoch + 1}/{num_epochs}], Batch {batch_idx}, Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"\n Epoch [{epoch + 1}/{num_epochs}] Training Loss: {epoch_loss:.4f}")

        # Валидация
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device, non_blocking=pin_memory)
                    targets = targets.to(device, non_blocking=pin_memory)
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            print(f" Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_geo_model_new.pth")
                print(f" Best model saved with val loss: {val_loss:.4f}")

        # Обновляем lr
        scheduler.step()

    return model

def test_model(model, test_loader, scaler=None):
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0.0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            for i in range(min(3, images.size(0))):
                pred = outputs[i].cpu().numpy()
                true = targets[i].cpu().numpy()

                if scaler is not None:
                    pred = scaler.inverse_transform([pred])[0]
                    true = scaler.inverse_transform([true])[0]

                print(f"True: lat={true[0]:.4f}, lon={true[1]:.4f}, alt={true[2]:.2f} | "
                      f"Pred: lat={pred[0]:.4f}, lon={pred[1]:.4f}, alt={pred[2]:.2f}")

    avg_loss = test_loss / len(test_loader)
    print(f"Test MSE Loss: {avg_loss:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=90, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--test_only', action='store_true', help='Only run test')
    args = parser.parse_args()

    scaler = None
    if not args.test_only:
        print("Fitting StandardScaler on train + val labels...")
        train_df = pd.read_csv(TRAIN_LABEL_PATH)[['lat', 'lon', 'alt']]
        val_df = pd.read_csv(VAL_LABEL_PATH)[['lat', 'lon', 'alt']]
        all_labels = pd.concat([train_df, val_df], ignore_index=True)
        scaler = StandardScaler()
        scaler.fit(all_labels)
        print("Scaler fitted.")

    num_workers = 0 if device.type == 'cpu' else 4
    pin_memory = device.type == 'cuda'

    if not args.test_only:
        print("Loading training and validation data...")
        train_dataset = GeoPoseDataset(TRAIN_IMG_DIR, TRAIN_LABEL_PATH, transform=transform, scaler=scaler)
        val_dataset = GeoPoseDataset(VAL_IMG_DIR, VAL_LABEL_PATH, transform=transform, scaler=scaler)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)

        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    else:
        train_loader = None
        val_loader = None

    test_dataset = GeoPoseDataset(TEST_IMG_DIR, TEST_LABEL_PATH, transform=transform, scaler=scaler)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    print(f"Test size: {len(test_dataset)}")

    model = get_model(output_size=3)

    if not args.test_only:
        print(f"Starting training on {device}")
        model = train_model(model, train_loader, val_loader, num_epochs=args.epochs, lr=args.lr)
        print("Training completed.")
    else:
        try:
            model.load_state_dict(torch.load("best_geo_model_new_90.pth", map_location=device))
            print("Model loaded from 'best_geo_model.pth'")
        except FileNotFoundError:
            print("Best model file not found. Run training first.")
            return

    print("Running test...")
    test_model(model, test_loader, scaler=scaler)

if __name__ == "__main__":
    main()