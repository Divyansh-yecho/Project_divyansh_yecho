import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from config import (
    TRAIN_DIR, TEST_DIR, CLASS_NAMES,
    resize_x, resize_y, imagenet_mean, imagenet_std,
    batch_size, val_split, random_seed
)

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}

train_transforms = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

eval_transforms = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])


class BrainTumorDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.df.iloc[idx]['filepath']).convert('RGB')
        label = self.df.iloc[idx]['label']
        if self.transform:
            img = self.transform(img)
        return img, label


def build_dataframes():
    filepaths, labels = [], []
    for cls in CLASS_NAMES:
        folder = os.path.join(TRAIN_DIR, cls)
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepaths.append(os.path.join(folder, fname))
                labels.append(CLASS_TO_IDX[cls])

    full_train_df = pd.DataFrame({'filepath': filepaths, 'label': labels})

    test_fps, test_lbs = [], []
    for cls in CLASS_NAMES:
        folder = os.path.join(TEST_DIR, cls)
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_fps.append(os.path.join(folder, fname))
                test_lbs.append(CLASS_TO_IDX[cls])

    test_df = pd.DataFrame({'filepath': test_fps, 'label': test_lbs})

    train_df, val_df = train_test_split(
        full_train_df,
        test_size=val_split,
        stratify=full_train_df['label'],
        random_state=random_seed
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df


def build_dataloaders():
    train_df, val_df, test_df = build_dataframes()

    train_ds = BrainTumorDataset(train_df, transform=train_transforms)
    val_ds   = BrainTumorDataset(val_df,   transform=eval_transforms)
    test_ds  = BrainTumorDataset(test_df,  transform=eval_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, test_df


the_dataloader = build_dataloaders
