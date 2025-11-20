import h5py
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import matplotlib.pyplot as plt


class ISIC_DataModule:
    """
    Wrap of ISIC_HDF5 functionality to simplify usage.
    """
    def __init__(self, hdf5_train_path: str, hdf5_test_path: str, 
                 metadataCSV_train_path: str, metadataCSV_test_path: str,
                 transform=None, is_labelled: bool = True):
        """
        Args:
            hdf5_train/test_path (str): Path to the HDF5 file containing images of train/test splits.
            train/test_CSV (str): Path to the CSV containing metadata for train/test.
            transform (callable): Optional transforms to be applied on a sample.
            is_labelled (bool): Whether the dataset includes labels (for train/val).
        """
        
        # Test of the 
        train_df = pd.read_csv(metadataCSV_train_path)
        test_df  = pd.read_csv(metadataCSV_test_path)

        print(f"train_df shape: {train_df.shape}")
        print(f"test_df shape:  {test_df.shape}")

        # Example: split train_df into 80% train / 20% valid
        train_size = int(0.8 * len(train_df))
        valid_size = len(train_df) - train_size
        train_subset, valid_subset = random_split(
            train_df, 
            [train_size, valid_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_df_sub = train_df.iloc[train_subset.indices].reset_index(drop=True)
        valid_df_sub = train_df.iloc[valid_subset.indices].reset_index(drop=True)

        print(f"Train samples: {len(train_df_sub)}, Valid samples: {len(valid_df_sub)}")

        # Basic transforms for ResNet
        resnet_transforms = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # Create Datasets
        train_dataset = ISIC_HDF5_Dataset(
            df=train_df_sub, 
            hdf5_path=TRAIN_HDF5,
            transform=resnet_transforms,
            is_labelled=True
        )

        valid_dataset = ISIC_HDF5_Dataset(
            df=valid_df_sub,
            hdf5_path=TRAIN_HDF5,
            transform=resnet_transforms,
            is_labelled=True
        )

        test_dataset = ISIC_HDF5_Dataset(
            df=test_df,
            hdf5_path=TEST_HDF5,
            transform=resnet_transforms,
            is_labelled=False
        )


# ---------------------------
# HDF5 custom dataset
# ---------------------------
class ISIC_HDF5_Dataset(Dataset):
    """
    A PyTorch Dataset that loads images from an HDF5 file given a DataFrame of IDs.
    Applies image transforms suitable for ResNet50.
    """
    def __init__(self, df: pd.DataFrame, hdf5_path: str, transform=None, is_labelled: bool = True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'isic_id' and optionally 'target'.
            hdf5_path (str): Path to the HDF5 file containing images.
            transform (callable): Optional transforms to be applied on a sample.
            is_labelled (bool): Whether the dataset includes labels (for train/val).
        """
        self.df = df.reset_index(drop=True)
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.is_labelled = is_labelled

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        isic_id = row["isic_id"]
        
        # Load image from HDF5
        image_rgb = self._load_image_from_hdf5(isic_id)
        
        # Apply transforms (PIL-style transforms require converting np array to PIL, or we can do tensor transforms)
        if self.transform is not None:
            # Convert NumPy array (H x W x C) to a PIL Image
            import torchvision.transforms.functional as F_v
            image_pil = F_v.to_pil_image(image_rgb)
            image = self.transform(image_pil)
        else:
            # By default, convert it to a tensor (C x H x W)
            image = torch.from_numpy(image_rgb).permute(2, 0, 1).float()

        if self.is_labelled:
            label = row["target"]
            label = torch.tensor(label).float()
            return image, label, isic_id
        else:
            return image, isic_id

    def _load_image_from_hdf5(self, isic_id: str):
        """
        Loads and decodes an image from HDF5 by isic_id.
        Returns a NumPy array in RGB format (H x W x 3).
        """
        with h5py.File(self.hdf5_path, 'r') as hf:
            encoded_bytes = hf[isic_id][()]  # uint8 array

        # Decode the image bytes with OpenCV (returns BGR)
        image_bgr = cv2.imdecode(encoded_bytes, cv2.IMREAD_COLOR)
        # Convert to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb

# ----------------------------------------------------
# 2. DataFrames and Basic Preprocessing / Transforms
# ----------------------------------------------------
# -----------------------------
# 3. Data Setup (Train/Valid/Test)
# -----------------------------
TRAIN_METADATA_CSV = "data/new-train-metadata.csv"
TEST_METADATA_CSV  = "data/students-test-metadata.csv"
TRAIN_HDF5         = "data/train-image.hdf5"
TEST_HDF5          = "data/test-image.hdf5"

train_df = pd.read_csv(TRAIN_METADATA_CSV)
test_df  = pd.read_csv(TEST_METADATA_CSV)

print(f"train_df shape: {train_df.shape}")
print(f"test_df shape:  {test_df.shape}")

# Example: split train_df into 80% train / 20% valid
train_size = int(0.8 * len(train_df))
valid_size = len(train_df) - train_size
train_subset, valid_subset = random_split(
    train_df, 
    [train_size, valid_size],
    generator=torch.Generator().manual_seed(42)
)

train_df_sub = train_df.iloc[train_subset.indices].reset_index(drop=True)
valid_df_sub = train_df.iloc[valid_subset.indices].reset_index(drop=True)

print(f"Train samples: {len(train_df_sub)}, Valid samples: {len(valid_df_sub)}")

# Basic transforms for ResNet
resnet_transforms = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# Create Datasets
train_dataset = ISIC_HDF5_Dataset(
    df=train_df_sub, 
    hdf5_path=TRAIN_HDF5,
    transform=resnet_transforms,
    is_labelled=True
)

valid_dataset = ISIC_HDF5_Dataset(
    df=valid_df_sub,
    hdf5_path=TRAIN_HDF5,
    transform=resnet_transforms,
    is_labelled=True
)

test_dataset = ISIC_HDF5_Dataset(
    df=test_df,
    hdf5_path=TEST_HDF5,
    transform=resnet_transforms,
    is_labelled=False
)

print("Created train/valid/test datasets.")
