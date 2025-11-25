import h5py
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.cuda import amp
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
from torch.utils.data import Sampler
import torchvision.transforms.functional as F_v
from torch.utils.data import Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

def plot_stats(stats_dicts, labels=["Train", "Validation", "Test"]):
    n_splits = len(stats_dicts)
    assert n_splits == len(labels), "Number of stats_dicts must match number of labels"

    # 1º - Image stats and class distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_class, ax_meanRGB, ax_stdRGB = axes

    # 1. Class distribution
    for i, stats in enumerate(stats_dicts):
        class_counts = stats.get("class_counts")
        if class_counts:
            classes = sorted(class_counts.keys())
            counts = [class_counts[c] for c in classes]
            total = sum(counts)
            percentages = [c / total * 100 for c in counts]

            bars = ax_class.bar(np.array(classes) + i*0.2, percentages, width=0.2, label=labels[i])
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax_class.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 1,
                    str(count),
                    ha='center',
                    va='bottom',
                    fontsize=10
                )
    if any(s.get("class_counts") for s in stats_dicts):
        ax_class.set_xticks(classes)
        ax_class.set_xlabel("Class")
        ax_class.set_ylabel("Percentage (%)")
        ax_class.set_title("Class distribution")
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
    ax_meanRGB.set_title("Image mean RGB")
    ax_meanRGB.legend()

    # 3. Image statistics: Std RGB
    for i, stats in enumerate(stats_dicts):
        std_rgb = stats["image_stats"]["std_RGB"]
        ax_stdRGB.bar(x + i*width, std_rgb, width=width, label=labels[i])
    ax_stdRGB.set_xticks(x + width)
    ax_stdRGB.set_xticklabels(rgb_labels)
    ax_stdRGB.set_ylabel("Std value")
    ax_stdRGB.set_title("Image std RGB")
    ax_stdRGB.legend()

    plt.tight_layout()
    plt.show()

    # 2º - Metadata information
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_age, ax_sex, ax_anatom, ax_area = axes.flatten()

    # 1. Numeric: Age
    feature = "age_approx"
    x = np.arange(len(labels))
    width = 0.35
    for cls in [0, 1]:
        means, stds = [], []
        for stats in stats_dicts:
            cls_stats = stats["metadata"]["numeric"].get(feature)
            if cls_stats and cls in cls_stats:
                means.append(cls_stats[cls]["mean"])
                stds.append(cls_stats[cls]["std"])
            else:
                means.append(0)
                stds.append(0)
        ax_age.bar(x + width*cls, means, width=width, yerr=stds, capsize=5, label=f"Class {cls}")
    ax_age.set_xticks(x + width/2)
    ax_age.set_xticklabels(labels)
    ax_age.set_ylabel("Age (mean ± std)")
    ax_age.set_title("Age comparison")
    ax_age.legend()

    # 2. Categorical: Sex (percentage)
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
            total = sum(cat_stats[cls].values()) or 1
            values = [cat_stats[cls].get(c, 0)/total * 100 for c in all_cats]
            ax_sex.bar(x + i*width, values, width=width, label=f"{labels[i]} Class {cls}")
    ax_sex.set_xticks(x + width*(len(stats_dicts)-1)/2)
    ax_sex.set_xticklabels(all_cats)
    ax_sex.set_ylabel("Percentage (%)")
    ax_sex.set_title("Sex distribution")
    ax_sex.legend(fontsize=8)

    # 3. Categorical: Anatomical site (percentage)
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
            total = sum(cat_stats[cls].values()) or 1
            values = [cat_stats[cls].get(c, 0)/total * 100 for c in all_cats]
            ax_anatom.bar(x + i*width, values, width=width, label=f"{labels[i]} Class {cls}")
    ax_anatom.set_xticks(x + width*(len(stats_dicts)-1)/2)
    ax_anatom.set_xticklabels(all_cats, rotation=45, ha="right")
    ax_anatom.set_ylabel("Percentage (%)")
    ax_anatom.set_title("Anatomical site distribution")
    ax_anatom.legend(fontsize=8)

    # 4. Numeric: Lesion area
    feature = "tbp_lv_areaMM2"
    for i, stats in enumerate(stats_dicts):
        area_stats = stats["metadata"]["numeric"].get(feature, {})
        means = [area_stats[cls]["mean"] if cls in area_stats else 0 for cls in [0,1]]
        stds = [area_stats[cls]["std"] if cls in area_stats else 0 for cls in [0,1]]
        ax_area.bar([i*2, i*2+0.8], means, width=0.8, yerr=stds, capsize=5, label=labels[i])
    ax_area.set_xticks([i*2 + 0.4 for i in range(len(labels))])
    ax_area.set_xticklabels(labels)
    ax_area.set_ylabel("Lesion area (mm²)")
    ax_area.set_title("Lesion area comparison")
    ax_area.legend()

    plt.tight_layout()
    plt.show()

def get_best_metadata(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                      num_feat=0, verbose=False):
    # first get wich are the common features
    common_cols = list(set(train_df.columns) & set(test_df.columns))

    # colums with no usefull data
    exclude_metadata = { "isic_id", "target"
        # categorical data without relation with melanoma
        "patient_id","lesion_id","copyright_license","attribution","tbp_lv_x",
        # position of the tile on the body, no relation
        "tbp_lv_y","tbp_lv_z","image_type","tbp_tile_type","tbp_lv_location",
        # complex no direct relation with melanoma
        "tbp_lv_symm_2axis_angle","tbp_lv_stdLExt","tbp_lv_area_perim_ratio",  # "tbp_lv_location_simple"
        # data leackage if present
        "iddx_full","iddx_1","iddx_2","iddx_3","iddx_4","iddx_5",
        "mel_mitotic_index","mel_thick_mm","tbp_lv_dnn_lesion_confidence"}


    # get the metadata cols to maximize
    metadata_cols = [
        col for col in common_cols
        if col not in exclude_metadata and col != "target"
    ]
    metadata_cols = sorted(metadata_cols)

    # If we keep all features
    if num_feat <= 0 or num_feat >= len(metadata_cols):
        if verbose:
            print("\n✔ Using all metadata features")
        return metadata_cols

    X = train_df[metadata_cols].copy()
    y = train_df["target"].values

    # Fill missing values
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].fillna("unknown")
            # encode categories as integers (RF can handle it)
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        else:
            X[col] = X[col].fillna(X[col].median())

    # prepare the data
    X = X.astype(float).values

    # ----------------------------
    # Train Random Forest
    # ----------------------------
    params = {'n_estimators': np.linspace(50, 500, 10, dtype=int)}
    rf_grid = GridSearchCV(
        estimator=RandomForestClassifier(max_depth=4, random_state=42, n_jobs=-1),
        param_grid=params
    ).fit(X, y)

    # get importance scores
    importances = rf_grid.feature_importances_
    ranking = pd.Series(importances, index=metadata_cols).sort_values(ascending=False)
    best_features = ranking.head(num_feat).index.tolist()

    if verbose:
        print("\n✔ Top metadata features (Random Forest importance):")
        for i, f in enumerate(best_features, 1):
            print(f"{i:3d}. {f}")

    return best_features


class MetadataProcessor:
    """
    Metadata processor for ISIC multimodal training.
    Supports both numeric and categorical metadata.

    Features:
    - One-hot encode categorical variables (or 0/1 for binary)
    - Normalize numeric variables using mean/std from TRAIN
    - Guarantees consistent ordering between train/val/test
    """

    def __init__(self, feature_list: list):
        """
        feature_list : list of features to use (numeric or categorical)
        """
        self.features = feature_list
    
    def fit(self, train_df: pd.DataFrame):
        # Separate numeric and categorical
        self.numeric_features = [
            f for f in self.features if np.issubdtype(train_df[f].dtype, np.number)
        ]
        self.categorical_features = [
            f for f in self.features if f not in self.numeric_features
        ]

        # ----- Numeric normalization and missing indicators -----
        self.numeric_means = {}
        self.numeric_stds = {}
        for f in self.numeric_features:
            col = pd.to_numeric(train_df[f], errors='coerce')
            self.numeric_means[f] = col.mean()
            self.numeric_stds[f] = col.std() if col.std() > 0 else 1.0  # avoid div by zero

        # ----- Categorical one-hot mapping -----
        self.categorical_columns = {}
        for f in self.categorical_features:
            dummies = pd.get_dummies(
                train_df[f], prefix=f, dummy_na=True
            )
            self.categorical_columns[f] = dummies.columns.tolist()

    def process_metadata(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform metadata:
        - normalize numeric with missing indicators
        - one-hot encode categorical features
        - maintain consistent column ordering
        """
        processed = []

        # ----- Numeric -----
        for f in self.numeric_features:
            if f in df.columns:
                col = pd.to_numeric(df[f], errors='coerce')

                # Missing indicator (1 if missing, 0 if present)
                missing_indicator = col.isna().astype(np.float32).values.reshape(-1, 1)
                processed.append(missing_indicator)

                # Fill missing with training mean
                col = col.fillna(self.numeric_means[f])

                # Standardize
                z = ((col - self.numeric_means[f]) / self.numeric_stds[f]).values.reshape(-1, 1).astype(np.float32)
                processed.append(z)

        # ----- Categorical -----
        for f in self.categorical_features:
            if f in df.columns:
                dummies = pd.get_dummies(df[f], prefix=f, dummy_na=True)
                expected_cols = self.categorical_columns[f]

                # Add missing columns
                for col in expected_cols:
                    if col not in dummies.columns:
                        dummies[col] = 0

                # Reorder columns
                dummies = dummies[expected_cols]
                processed.append(dummies.values.astype(np.float32))

        # ----- Concatenate -----
        if processed:
            return np.concatenate(processed, axis=1)
        else:
            return np.zeros((len(df), 1), dtype=np.float32)
    
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
    
class TrainableModule():
    def __init__(self, criterion, device, patience, min_delta):

        self.device = device
        self.criterion = criterion
        self.to(device)
        self.optimizer = Adam(self.parameters(), lr=1e-4)
        self.best_model = None
        self.best_model_AUC = 0

        # metrics
        self.history = {}

        # Early stopping
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_state_dict = None

        # AMP
        self.scaler = amp.GradScaler() if device.type == "cuda" else None

    # -------------------------------------------------------------
    def _train_epoch(self, loader):
        self.train()
        total_loss = 0
        pos_labels = 0
        total_elem = 0

        for images, metadata, labels, _ in tqdm(loader, leave=False):
            images = images.to(self.device, non_blocking=True)
            metadata = metadata.to(self.device, non_blocking=True)
            total_elem += labels.size(0)
            pos_labels += (labels == 1).sum().item()
            labels = labels.float().to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type):  # AMP
                preds = self(images, metadata).view(-1)
                loss = self.criterion(preds, labels)

            # AMP scaled backward
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader), pos_labels / total_elem

    # -------------------------------------------------------------
    def _validate(self, loader):
        self.eval()
        total_loss = 0
        preds_list = []
        labels_list = []

        with torch.no_grad():
            # calculate the roc auc
            for images, metadata, labels, _ in loader:
                images = images.to(self.device, non_blocking=True)
                metadata = metadata.to(self.device, non_blocking=True)
                labels = labels.float().to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type=self.device.type):  # AMP
                    preds = self(images, metadata).view(-1)
                    loss = self.criterion(preds, labels)
                total_loss += loss.item()

                preds_list.append(preds.cpu())
                labels_list.append(labels.cpu())

        preds = torch.cat(preds_list).sigmoid().numpy()
        labels = torch.cat(labels_list).numpy()
        auc = roc_auc_score(labels, preds)

        return total_loss / len(loader), auc
    # -------------------------------------------------------------
    def pred_probs(self, loader, threshold=0.5):
        """
        Return predictions and labels for a given dataloader.

        Parameters
        ----------
        loader : DataLoader
            PyTorch DataLoader for validation/test.
        threshold : float, default=0.5
            Threshold for converting probabilities to binary predictions.

        Returns
        -------
        labels : np.ndarray
            Ground truth labels.
        preds : np.ndarray
            Predicted labels (binary).
        probs : np.ndarray
            Predicted probabilities.
        """
        self.eval()
        labels_list = []
        probs_list = []

        with torch.no_grad():
            for images, metadata, labels, _ in loader:
                images = images.to(self.device, non_blocking=True)
                metadata = metadata.to(self.device, non_blocking=True)
                labels = labels.float()

                preds_logits = self(images, metadata).view(-1)
                probs = torch.sigmoid(preds_logits)

                labels_list.append(labels)
                probs_list.append(probs.cpu())

        labels = torch.cat(labels_list).numpy()
        probs = torch.cat(probs_list).numpy()
        preds = (probs >= threshold).astype(int)

        return labels, preds, probs

    # -------------------------------------------------------------
    def _check_early_stopping(self, score):

        if self.best_score is None:
            self.best_score = score
            self.best_state_dict = {k: v.cpu().clone() for k, v in self.state_dict().items()}
            return False

        # Check if we improve the results over the last epoch and some margin
        if score <= self.best_score - self.min_delta:
            self.best_score = score
            self.best_state_dict = {k: v.cpu().clone() for k, v in self.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1

            # if we dont improve in n epoch we get out            
            if self.counter >= self.patience:
                return True

        return False

    # -------------------------------------------------------------
    def save(self, path):
        print(f"Saved model with val_auc {self.best_model_AUC }")
        torch.save(self.best_model, path)
    
class ISIC_Multimodal_Dataset(Dataset):
    """
    PyTorch Dataset wrapper for ISIC HDF5 images and clinical metadata.

    Provides:
    - On-the-fly image loading from HDF5 with optional torchvision transforms.
    - Standardized metadata handling: z-score scaling, missing-value indicators,
    mean imputation, and consistent one-hot encoding.
    - Optional dynamic augmentation applied only to positive (malignant) samples.
    - Utility method `get_dataset_stats()` for sample counts, class balance, image
    stats (RGB mean/std, size), and metadata summaries.

    Parameters:
        hdf5_path : str
            Path to HDF5 image file.
        metadata_df : pd.DataFrame
            Table with image IDs, labels, and metadata.
        image_transform : callable, optional
            Preprocessing transforms for all images.
        data_augmentation_trans : callable, optional
            Augmentations applied dynamically to positive samples.
        augment_positives : bool, default=False
            Enables positive-sample augmentation.

    Notes:
    - Original images remain unchanged; augmentations occur during __getitem__.
    - Stats from `get_dataset_stats()` are useful for analyzing train/val/test splits.
    """
    def __init__(self, hdf5_path: str, metadata_df: pd.DataFrame,
                 image_transform, metadata_processor: MetadataProcessor, 
                 data_augmentation_trans = None,
                 augment_positives:bool=False):
        
        self.augment_trans = data_augmentation_trans    # transformation to data augmentation
        self.prep_trans = image_transform               # transformation base
        self.metadata_processor = metadata_processor
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
        - 'class_counts': dict with counts per class (if labelled)
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

        # ---------------------
        # IMAGE STATISTICS
        # ---------------------
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

        # ---------------------
        # METADATA STATISTICS
        # ---------------------
        metadata_info = {}

        # Ensure attributes exist even if the processor doesn't define them
        categorical_vars = getattr(self.metadata_processor, "categorical_features", [])
        numeric_vars = getattr(self.metadata_processor, "numeric_features", [])

        # Only keep variables that exist in the raw metadata DataFrame
        categorical_vars = [c for c in categorical_vars if c in df.columns]
        numeric_vars = [n for n in numeric_vars if n in df.columns]

        # ---------------------
        # NUMERIC METADATA
        # ---------------------
        numeric_stats = {}

        for col in numeric_vars:
            col_stats = {}

            if self.is_labelled:
                for cls in classes:
                    col_data = pd.to_numeric(df[df["target"] == cls][col], errors="coerce")
                    col_stats[cls] = {
                        "mean": float(col_data.mean()),
                        "std": float(col_data.std()),
                        "missing_frac": float(col_data.isna().mean())
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

        # ---------------------
        # CATEGORICAL METADATA
        # ---------------------
        categorical_stats = {}

        for col in categorical_vars:
            col_stats = {}

            if self.is_labelled:
                for cls in classes:
                    cls_data = df[df["target"] == cls][col]
                    col_stats[cls] = (cls_data.value_counts(normalize=True) * 100).to_dict()
            else:
                col_stats["all"] = (df[col].value_counts(normalize=True) * 100).to_dict()

            categorical_stats[col] = col_stats

        metadata_info["categorical"] = categorical_stats

        stats["metadata"] = metadata_info

        return stats
