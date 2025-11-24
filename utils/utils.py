import h5py
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler
import torchvision.transforms.functional as F_v
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
import matplotlib.pyplot as plt


################################################################################
                                # TOOLS #
################################################################################

def create_balanced_split(metadataCSV_path: str, neg_multiplier: int = 10,
                          val_frac: float = 0.2, seed: int = 42):
    """
    Create a class-balanced dataset by retaining all positive samples and 
    randomly undersampling the negative class according to a specified ratio.
    After constructing the balanced subset, the function performs a stratified 
    train/validation split to maintain consistent class proportions in both sets.

    Parameters
    ----------
    metadataCSV_path : str
        Path to the metadata CSV file containing at least the columns:
        'isic_id' and 'target'. The 'target' column must contain 0/1 labels.
    
    neg_multiplier : int, default=10
        Number of negative samples to keep per positive sample.
    
    val_frac : float, default=0.2
        Fraction of the balanced dataset to allocate to the validation set.
    
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    train_df : pandas.DataFrame
        Balanced training subset with stratified class distribution.

    val_df : pandas.DataFrame
        Balanced validation subset with stratified class distribution.

    df_balanced : pandas.DataFrame
        The full balanced dataset prior to splitting.
    """

    df = pd.read_csv(metadataCSV_path)

    # 1. Separate positive and negative samples
    df_pos = df[df.target == 1]
    df_neg = df[df.target == 0]

    n_pos = len(df_pos)
    n_neg_needed = n_pos * neg_multiplier

    print(f"Positives: {n_pos}")
    print(f"Negatives available: {len(df_neg)}")
    print(f"Negatives to sample: {n_neg_needed}")

    # 2. Undersample negative class
    df_neg_sampled = df_neg.sample(n=n_neg_needed, random_state=seed)

    # 3. Combine positives with sampled negatives
    df_balanced = pd.concat([df_pos, df_neg_sampled], ignore_index=True)

    print(f"\nBalanced dataset size: {len(df_balanced)}")
    print(df_balanced['target'].value_counts())

    # 4. Stratified train/validation split
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        df_balanced,
        test_size=val_frac,
        random_state=seed,
        stratify=df_balanced['target']
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    return train_df, val_df, df_balanced

def plot_dataset_comparison_subplots(stats_dicts, labels=["Train", "Validation", "Test"]):
    """
    Plot comparison of multiple datasets in a single figure with subplots.

    Parameters:
    -----------
    stats_dicts : list of dict
        List of stats dictionaries returned by `analyze_dataset`.
    labels : list of str
        Labels for each dataset (e.g., ["Train", "Validation", "Test"]).
    """

    n_splits = len(stats_dicts)
    assert n_splits == len(labels), "Number of stats_dicts must match number of labels"

    # 1º - Image stats and class distribution
    fig, axes = plt.subplots(1,3, figsize=(15, 5))
    ax_class, ax_meanRGB, ax_stdRGB = axes

    # 1. Class distribution (only validation and train)
    for i, stats in enumerate(stats_dicts):
        class_counts = stats.get("class_counts")
        if class_counts:
            classes = sorted(class_counts.keys())
            counts = [class_counts[c] for c in classes]
            total = sum(counts)
            percentages = [c / total * 100 for c in counts]

            bars = ax_class.bar(np.array(classes) + i*0.2, percentages, width=0.2, label=labels[i])

            # Add the numbe of images on top of the bar
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax_class.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 1,  # un poco encima de la barra
                    str(count),
                    ha='center',
                    va='bottom',
                    fontsize=10
                )
    if any(s.get("class_counts") for s in stats_dicts):
        ax_class.set_xticks(classes)
        ax_class.set_xlabel("Class")
        ax_class.set_ylabel("Percentage (%)")
        ax_class.set_title("Class distribution comparison")
        ax_class.legend()

    # 2. Image statistics: Mean RGB
    rgb_labels = ["R", "G", "B"]
    width = 0.2
    x = np.arange(3)
    for i, stats in enumerate(stats_dicts):
        mean_rgb = stats["image_stats"]["mean_RGB"]
        ax_meanRGB.bar(x + i*width, mean_rgb, width=width, label=labels[i])
    ax_meanRGB.set_xticks(x + width)
    ax_meanRGB.set_xticklabels(rgb_labels)
    ax_meanRGB.set_ylabel("Mean value")
    ax_meanRGB.set_title("Image mean RGB comparison")
    ax_meanRGB.legend()

    # 3. Image statistics: Std RGB
    for i, stats in enumerate(stats_dicts):
        std_rgb = stats["image_stats"]["std_RGB"]
        ax_stdRGB.bar(x + i*width, std_rgb, width=width, label=labels[i])
    ax_stdRGB.set_xticks(x + width)
    ax_stdRGB.set_xticklabels(rgb_labels)
    ax_stdRGB.set_ylabel("Std value")
    ax_stdRGB.set_title("Image std RGB comparison")
    ax_stdRGB.legend()

    plt.tight_layout()
    plt.show()

    # 2º Metadatadata information
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_age, ax_sex, ax_anatom, ax_image_type = axes.flatten()

    # 1. Numeric: Age
    feature = "age_approx"
    x = np.arange(len(labels))
    width = 0.35
    for cls in [0,1]:  # clases
        means = []
        stds = []
        for stats in stats_dicts:
            cls_stats = stats["metadata"]["numeric"].get(feature)
            if cls_stats and str(cls) in cls_stats:
                means.append(cls_stats[cls]["mean"])
                stds.append(cls_stats[cls]["std"])
            else:
                means.append(0)
                stds.append(0)
        ax_age.bar(x + width*cls, means, width=width, yerr=stds, capsize=5, label=f"Class {cls}")
    ax_age.set_xticks(x + width/2)
    ax_age.set_xticklabels(labels)
    ax_age.set_ylabel("Age (mean ± std)")
    ax_age.set_title("Numeric metadata: Age comparison")
    ax_age.legend()

    # -----------------------------
    # 2. Categorical: Sex
    # -----------------------------
    feature = "sex"
    all_cats = set()
    for stats in stats_dicts:
        cat_stats = stats["metadata"]["categorical"].get(feature, {})
        for cls_key in cat_stats.keys():
            all_cats.update(cat_stats[cls_key].keys())
    all_cats = sorted(all_cats)
    n_cats = len(all_cats)
    width = 0.2
    x = np.arange(n_cats)

    for i, stats in enumerate(stats_dicts):
        cat_stats = stats["metadata"]["categorical"].get(feature, {})
        for cls in sorted(cat_stats.keys()):
            values = [cat_stats[cls].get(c,0)*100 for c in all_cats]
            ax_sex.bar(x + i*width, values, width=width, label=f"{labels[i]} Class {cls}")
    ax_sex.set_xticks(x + width*(len(stats_dicts)-1)/2)
    ax_sex.set_xticklabels(all_cats)
    ax_sex.set_ylabel("Percentage (%)")
    ax_sex.set_title("Categorical metadata: Sex")
    ax_sex.legend(fontsize=8)

    # -----------------------------
    # 3. Categorical: Anatomical site
    # -----------------------------
    feature = "anatom_site_general"
    all_cats = set()
    for stats in stats_dicts:
        cat_stats = stats["metadata"]["categorical"].get(feature, {})
        for cls_key in cat_stats.keys():
            all_cats.update(cat_stats[cls_key].keys())
    all_cats = sorted(all_cats)
    n_cats = len(all_cats)
    width = 0.2
    x = np.arange(n_cats)

    for i, stats in enumerate(stats_dicts):
        cat_stats = stats["metadata"]["categorical"].get(feature, {})
        for cls in sorted(cat_stats.keys()):
            values = [cat_stats[cls].get(c,0)*100 for c in all_cats]
            ax_anatom.bar(x + i*width, values, width=width, label=f"{labels[i]} Class {cls}")
    ax_anatom.set_xticks(x + width*(len(stats_dicts)-1)/2)
    ax_anatom.set_xticklabels(all_cats, rotation=45, ha="right")
    ax_anatom.set_ylabel("Percentage (%)")
    ax_anatom.set_title("Categorical metadata: Anatomical site")
    ax_anatom.legend(fontsize=8)

    # -----------------------------
    # 4. Categorical: Image type
    # -----------------------------
    feature = "image_type"
    all_cats = set()
    for stats in stats_dicts:
        cat_stats = stats["metadata"]["categorical"].get(feature, {})
        for cls_key in cat_stats.keys():
            all_cats.update(cat_stats[cls_key].keys())
    all_cats = sorted(all_cats)
    n_cats = len(all_cats)
    width = 0.2
    x = np.arange(n_cats)

    for i, stats in enumerate(stats_dicts):
        cat_stats = stats["metadata"]["categorical"].get(feature, {})
        for cls in sorted(cat_stats.keys()):
            values = [cat_stats[cls].get(c,0)*100 for c in all_cats]
            ax_image_type.bar(x + i*width, values, width=width, label=f"{labels[i]} Class {cls}")
    ax_image_type.set_xticks(x + width*(len(stats_dicts)-1)/2)
    ax_image_type.set_xticklabels(all_cats, rotation=45, ha="right")
    ax_image_type.set_ylabel("Percentage (%)")
    ax_image_type.set_title("Categorical metadata: Image type")
    ax_image_type.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

class MetadataProcessor:
    """
    Process and engineer clinical metadata for ISIC multimodal training.
    Improvements over the original version:
      - Normalization uses ONLY training statistics (no leakage).
      - Uses z-score scaling instead of per-column min-max.
      - Adds missing-value indicator features.
      - Keeps categorical encoding consistent across splits.
      - Adds more dermatology-inspired medical features.
    """

    def __init__(self, train_df):

        self.train_df = train_df.copy()

        # Feature groups
        self.numeric_features = [
            'age_approx', 'clin_size_long_diam_mm', 
            'tbp_lv_areaMM2', 'tbp_lv_perimeterMM', 'tbp_lv_minorAxisMM',
            'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_A', 'tbp_lv_Aext', 
            'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext',
            'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_deltaL', 'tbp_lv_deltaA', 'tbp_lv_deltaB',
            'tbp_lv_norm_border', 'tbp_lv_norm_color', 'tbp_lv_eccentricity',
            'tbp_lv_color_std_mean', 'tbp_lv_radial_color_std_max'
        ]

        self.categorical_features = [
            'sex', 'anatom_site_general', 'tbp_lv_location_simple', 
            'tbp_tile_type', 'image_type'
        ]

        # 1º - Compute training statistics (mean, std for numeric)
        self.numeric_means = train_df[self.numeric_features].apply(
            pd.to_numeric, errors='coerce'
        ).mean()

        self.numeric_stds = train_df[self.numeric_features].apply(
            pd.to_numeric, errors='coerce'
        ).std().replace(0, 1)   # prevent div-by-zero


        # 2º Build one-hot categorical maps for training data
        self.categorical_columns = {}
        for feature in self.categorical_features:
            dummies = pd.get_dummies(train_df[feature], prefix=feature, dummy_na=True)
            self.categorical_columns[feature] = dummies.columns.tolist()

    def process_metadata(self, df):
        """
        Main processing pipeline used during train/val/test.
        Returns a float32 NumPy array with consistent feature ordering.
        """
        processed = []

        # 1. NUMERIC FEATURES ((x - mean) / std) normalization
        for feature in self.numeric_features:
            if feature in df.columns:
                col = pd.to_numeric(df[feature], errors='coerce')

                # Missing-value indicator (0 = present, 1 = missing)
                processed.append(col.isna().astype(np.float32).values.reshape(-1, 1))

                # Fill missing with training mean
                col = col.fillna(self.numeric_means[feature])

                # Standardization: (x - mean) / std
                z = (col - self.numeric_means[feature]) / self.numeric_stds[feature]
                processed.append(
                    z.values.reshape(-1, 1).astype(np.float32)
                )

        # 2. CATEGORICAL FEATURES (consistent one-hot encoding)
        for feature in self.categorical_features:
            if feature in df.columns:
                dummies = pd.get_dummies(df[feature], prefix=feature, dummy_na=True)
                expected_cols = self.categorical_columns[feature]

                # Add missing columns with zeros
                for col in expected_cols:
                    if col not in dummies.columns:
                        dummies[col] = 0

                # Keep only expected columns (correct ordering)
                dummies = dummies[expected_cols]
                processed.append(dummies.values.astype(np.float32))


        # 3. MEDICAL FEATURES
        medical = self._medical_features(df)
        if medical:
            processed.extend(medical)
            
        # 4. Concatenate all features
        if processed:
            final = np.concatenate(processed, axis=1)
            return final.astype(np.float32)
        else:
            return np.zeros((len(df), 1), dtype=np.float32)


    def _medical_features(self, df):
        """
        Add domain-driven features used in dermatology literature.
        """
        medical = []


        # ---- Compactness (shape irregularity)
        if 'tbp_lv_areaMM2' in df.columns and 'tbp_lv_perimeterMM' in df.columns:
            area = pd.to_numeric(df['tbp_lv_areaMM2'], errors='coerce').fillna(0)
            per = pd.to_numeric(df['tbp_lv_perimeterMM'], errors='coerce').fillna(0)
            compact = 4 * np.pi * area / (per**2 + 1e-8)
            medical.append(compact.values.reshape(-1, 1).astype(np.float32))


        # ---- LAB color contrast
        if 'tbp_lv_deltaL' in df.columns and 'tbp_lv_deltaA' in df.columns:
            dL = pd.to_numeric(df['tbp_lv_deltaL'], errors='coerce').fillna(0)
            dA = pd.to_numeric(df['tbp_lv_deltaA'], errors='coerce').fillna(0)
            contrast = np.sqrt(dL**2 + dA**2)
            medical.append(contrast.values.reshape(-1, 1).astype(np.float32))


        # ---- Aspect ratio (minor/major axis)
        if 'tbp_lv_minorAxisMM' in df.columns and 'clin_size_long_diam_mm' in df.columns:
            minor = pd.to_numeric(df['tbp_lv_minorAxisMM'], errors='coerce').fillna(0)
            major = pd.to_numeric(df['clin_size_long_diam_mm'], errors='coerce').fillna(1)
            ratio = minor / (major + 1e-8)
            medical.append(ratio.values.reshape(-1, 1).astype(np.float32))


        # ---- Age groups (soft clinical binning)
        if 'age_approx' in df.columns:
            age = pd.to_numeric(df['age_approx'], errors='coerce').fillna(
                self.numeric_means['age_approx']
            )
            bins = [0, 30, 50, 70, 120]
            labels = ['young', 'middle', 'senior', 'elderly']
            age_group = pd.cut(age, bins=bins, labels=labels)
            dummies = pd.get_dummies(age_group, prefix='age_group', dummy_na=True)
            medical.append(dummies.values.astype(np.float32))

        return medical
    
class PositiveOversampler(Sampler):
    """
    Custom PyTorch Sampler to oversample positive class examples in a dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset containing a 'target' column in `dataset.metadata`.
    
    pos_multiplier : int, default=3
        Number of times to repeat each positive sample in the sampler
    """

    def __init__(self, dataset, pos_multiplier=3):
        self.dataset = dataset
        self.pos_multiplier = pos_multiplier
        self.indices = []
        for idx, row in dataset.metadata.iterrows():
            if row['target'] == 1:
                self.indices.extend([idx]*pos_multiplier)
            else:
                self.indices.append(idx)

    def __iter__(self):
        np.random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    
class ISIC_Multimodal_Dataset(Dataset):
    """
    PyTorch Dataset wrapper for ISIC HDF5 image data with associated clinical metadata.

    This class provides a unified interface to:
    - Load images stored in HDF5 files on-the-fly.
    - Preprocess and standardize clinical metadata.
    - Optionally apply dynamic image augmentations to positive (malignant) samples.
    - Compute and return detailed dataset statistics for analysis and comparison.

    Features:
    ---------
    1. Image loading:
        - Reads images efficiently from HDF5 datasets.
        - Supports torchvision transforms for preprocessing and augmentation.

    2. Metadata processing:
        - Numeric features are standardized using z-score (training statistics).
        - Missing values are handled via indicators and mean imputation.
        - Categorical features are one-hot encoded consistently with training data.
        - Additional domain-driven medical features can be included.

    3. Positive sample augmentation:
        - Dynamic augmentation applies random transformations during __getitem__.
        - Original images in HDF5 remain unmodified.
        - Useful for oversampling rare positive samples in imbalanced datasets.

    4. Dataset statistics and analysis:
        - Public method `get_dataset_stats()` computes:
            - Number of samples.
            - Class distribution and imbalance ratio (if labeled).
            - Image statistics (mean RGB, std RGB, average size).
            - Metadata statistics:
                - Numeric: mean, std, missing fraction.
                - Categorical: normalized distributions per class.
        - Enables easy comparison across train, validation, and test splits.

    Parameters:
    -----------
    hdf5_path : str
        Path to the HDF5 file containing image data.
    metadata_df : pd.DataFrame
        DataFrame containing image IDs, labels ('target'), and clinical metadata.
    image_transform : torchvision.transforms (callable), optional
        Preprocessing transforms applied to all images (e.g., resizing, normalization).
    data_augmentation_trans : torchvision.transforms (callable), optional
        Augmentation transforms applied dynamically to positive samples.
    augment_positives : bool, default=False
        If True, applies dynamic augmentation to positive samples during __getitem__.

    Usage:
    ------
    dataset = ISIC_Multimodal_Dataset(
        hdf5_path="data/images.h5",
        metadata_df=train_df,
        image_transform=preprocess_trans,
        data_augmentation_trans=augmentation_trans,
        augment_positives=True
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    Notes:
    ------
    - Original images are never modified; augmentations are applied dynamically.
    - `get_dataset_stats()` returns a comprehensive dictionary of dataset statistics,
      suitable for comparison across different splits.
    - Supports imbalanced datasets through dynamic positive augmentation.
    """
    def __init__(self, hdf5_path: str, metadata_df: pd.DataFrame,
                 image_transform, data_augmentation_trans = None,
                 augment_positives:bool=False):
        
        self.augment_trans = data_augmentation_trans    # transformation to data augmentation
        self.prep_trans = image_transform               # transformation base
        self.metadata_processor = MetadataProcessor(metadata_df)
        self.augment_positives = augment_positives    # apply dinamic transformations

        # 1º Read the metadata
        self.metadata = metadata_df
        self.metadata_ids = self.metadata["isic_id"].to_list()
        self.is_labelled = "target" in self.metadata.columns    # if dataset has a target

        # 2º Read the HDF5 file 
        self.hdf5_file = h5py.File(hdf5_path, "r")

        # 3º Preprocess the metadata
        self.metadata_features = self.metadata_processor.process_metadata(self.metadata)


    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        isic_id = row["isic_id"]
        
        # Load and transform image
        image_rgb = self._load_image_from_hdf5(isic_id)
        
        # Dynamic positive augmentation
        if self.augment_positives and self.is_labelled and row["target"] == 1:
            image_pil = F_v.to_pil_image(image_rgb)
            image_aug = self.augment_trans(image_pil)
            image = self.prep_trans(image_aug)  # apply base trans
        else:
            image_pil = F_v.to_pil_image(image_rgb)
            image = self.prep_trans(image_pil)

        metadata = torch.FloatTensor(self.metadata_features[idx])

        if self.is_labelled:
            label = torch.tensor(row["target"]).float()
            return image, metadata, label, isic_id
        else:
            return image, metadata, isic_id
    
    def _load_image_from_hdf5(self, isic_id):
        """Load image from already-open HDF5 file"""
        encoded_bytes = self.hdf5_file[isic_id][()]
        image_bgr = cv2.imdecode(encoded_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb
    
    def __del__(self):
        """Clean up: close HDF5 file"""
        if hasattr(self, 'hdf5_file') and self.hdf5_file:
            self.hdf5_file.close()
    
    # ---------------------------------------------------------------------------
    # ------ Analize funcitons ------
    # ---------------------------------------------------------------------------
    def get_dataset_stats(self, sample_size: int = 300):
        """
        Compute dataset statistics and metadata distributions per class.
        Returns a dictionary containing:
        - 'num_samples': total number of samples
        - 'class_counts': dict with counts per class (if labeled)
        - 'imbalance_ratio': ratio 0/1
        - 'image_stats': mean, std RGB
        - 'metadata': dict with numeric and categorical statistics per class
        """
        df = self.metadata
        stats = {}

        # Total number of samples
        stats["num_samples"] = len(df)

        # Class distribution
        if self.is_labelled:
            class_counts = df["target"].value_counts().sort_index()
            stats["class_counts"] = class_counts.to_dict()
            stats["imbalance_ratio"] = class_counts.get(0, 0) / max(class_counts.get(1, 1), 1)
            classes = sorted(df["target"].unique())
        else:
            stats["class_counts"] = None
            stats["imbalance_ratio"] = None
            classes = []

        # Image statistics (sample subset for speed)
        means, stds = [], []
        sample_ids = df["isic_id"].tolist()[:sample_size]

        for isic_id in sample_ids:
            img = self._load_image_from_hdf5(isic_id)

            if self.prep_trans:
                img_t = self.prep_trans(F_v.to_pil_image(img))
                img = img_t.permute(1, 2, 0).numpy()

            means.append(img.mean(axis=(0, 1)))
            stds.append(img.std(axis=(0, 1)))

        means = np.array(means)
        stds = np.array(stds)

        stats["image_stats"] = {
            "mean_RGB": means.mean(axis=0).tolist(),
            "std_RGB": stds.mean(axis=0).tolist()
        }

        # Metadata statistics per class
        metadata_info = {}
        categorical_vars = [c for c in self.metadata_processor.categorical_features if c in df.columns]
        numeric_vars = [n for n in self.metadata_processor.numeric_features if n in df.columns]

        # Numeric: mean, std, missing fraction per class
        numeric_stats = {}
        for col in numeric_vars:
            col_stats = {}
            if self.is_labelled:
                for cls in classes:
                    cls_data = pd.to_numeric(df[df["target"] == cls][col], errors="coerce")
                    col_stats[cls] = {
                        "mean": float(cls_data.mean()),
                        "std": float(cls_data.std()),
                        "missing_frac": float(cls_data.isna().mean())
                    }
            else:
                col_data = pd.to_numeric(df[col], errors="coerce")
                col_stats["all"] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "missing_frac": float(col_data.isna().mean())
                }
            numeric_stats[col] = col_stats
        metadata_info["numeric"] = numeric_stats

        # Categorical: proportion per class
        categorical_stats = {}
        for col in categorical_vars:
            cat_stat = {}
            if self.is_labelled:
                for cls in classes:
                    cls_data = df[df["target"] == cls][col]
                    cat_stat[cls] = (cls_data.value_counts(normalize=True) * 100).to_dict()
            else:
                cat_stat["all"] = (df[col].value_counts(normalize=True) * 100).to_dict()
            categorical_stats[col] = cat_stat
        metadata_info["categorical"] = categorical_stats

        stats["metadata"] = metadata_info

        return stats
