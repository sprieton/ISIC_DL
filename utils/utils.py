import h5py
import cv2
import os
import random
import numpy as np
import pandas as pd
import torch
from albumentations.core.transforms_interface import ImageOnlyTransform
import albumentations as A
from tqdm import tqdm
import torch.nn as nn
from torch.cuda import amp
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import optuna
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import joblib
import warnings



################################################################################
                                # TOOLS #
################################################################################
def make_sampler_weights(targets, p_target):
    "give the weights for the weighted sampler"
    n_pos = (targets == 1).sum()
    n_neg = (targets == 0).sum()
    # sampler class weights (w_neg=1)
    w_pos = (p_target / (1 - p_target)) * (n_neg / n_pos)
    class_weights = np.array([1.0, w_pos], dtype=float)
    sample_weights = np.array([class_weights[int(t)] for t in targets], dtype=float)
    
    return sample_weights

def remove_last_layer(model):
    """
    Erase the last layer of the model
    """
    last_linear_or_conv = None
    
    # Recorrer todos los módulos hijos
    for name, module in model.named_children():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            last_linear_or_conv = name
    
    if last_linear_or_conv is not None:
            setattr(model.backbone, last_linear_or_conv, nn.Identity())
    
    return model

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

class AdvancedHairAugmentation(ImageOnlyTransform):
    def __init__(self, hairs: int = 4, hairs_folder: str = "", always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.hairs = hairs
        self.hairs_folder = hairs_folder

    def apply(self, image, **params):
        n_hairs = random.randint(0, self.hairs)
        if n_hairs == 0:
            return image

        H, W, _ = image.shape

        # List hair images
        hair_files = [im for im in os.listdir(self.hairs_folder) if im.endswith(".png")]
        if not hair_files:
            return image

        for _ in range(n_hairs):
            # Load & random flip/rotate
            hair = cv2.imread(os.path.join(self.hairs_folder, random.choice(hair_files)))
            if hair.shape[2] == 4:
                hair = hair[:, :, :3]
            hair = cv2.cvtColor(hair, cv2.COLOR_BGR2RGB)

            # Random flip & rotation
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            hH, hW, _ = hair.shape

            # -------------------------------------------------------------------------------------
            # 1. Resize hair if bigger than target image
            # -------------------------------------------------------------------------------------
            if hH > H or hW > W:
                scale = min(H / hH, W / hW) * random.uniform(0.5, 0.9)
                new_hH = max(1, int(hH * scale))
                new_hW = max(1, int(hW * scale))
                hair = cv2.resize(hair, (new_hW, new_hH), interpolation=cv2.INTER_AREA)
                hH, hW = new_hH, new_hW

            # -------------------------------------------------------------------------------------
            # 2. Ensure valid placement
            # -------------------------------------------------------------------------------------
            if H - hH <= 0 or W - hW <= 0:
                continue  # Skip hair if still too big

            roi_y = random.randint(0, H - hH)
            roi_x = random.randint(0, W - hW)

            roi = image[roi_y:roi_y + hH, roi_x:roi_x + hW]

            # -------------------------------------------------------------------------------------
            # 3. Normal hair blending with mask
            # -------------------------------------------------------------------------------------
            img2gray = cv2.cvtColor(hair, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            fg = cv2.bitwise_and(hair, hair, mask=mask)
            blended = cv2.add(bg, fg)

            image[roi_y:roi_y + hH, roi_x:roi_x + hW] = blended

        return image

class Microscope(A.ImageOnlyTransform):
    def __init__(self, p=0.5, always_apply=False):
        super().__init__(always_apply, p)

    def apply(self, img):
        # Random skip based on probability
        if random.random() >= self.p:
            return img

        h, w = img.shape[:2]

        # --- Create a white mask (255 everywhere) ---
        circle_img = np.ones((h, w), dtype=np.uint8) * 255

        # --- Draw a black circle in the center ---
        center = (h // 2, w // 2)
        radius = random.randint(h // 2 - 3, h // 2 + 15)
        cv2.circle(circle_img, center, radius, 0, -1)

        # --- Convert to a binary mask (0 inside circle, 1 outside) ---
        mask = (circle_img == 0).astype(np.uint8)

        # --- Ensure the mask has 3 channels ---
        mask = np.repeat(mask[:, :, None], 3, axis=2)

        # --- Apply mask: darken the center region ---
        img_masked = img * mask

        return img_masked.astype(np.uint8)

class MetadataProcessor:
    """
    Processor for tabular metadata (numeric + categorical) with:
      - Multivariate imputation for numeric features (IterativeImputer)
      - Normalization (z‑score) of numeric features
      - Imputation of categorical NaNs (fill with "Missing") + one-hot encoding
      - Consistent output shape/order between train / val / test
    """

    def __init__(self, feature_list: list, categorical_as_str: bool = True, imputer_kwargs: dict = None):
        """
        feature_list: list of feature names (columns) to use.
        categorical_as_str: if True, convert categorical features to str before encoding (to handle NaNs).
        imputer_kwargs: arguments passed to IterativeImputer (e.g. random_state, max_iter).
        """
        self.features = feature_list
        self.categorical_as_str = categorical_as_str
        self.num_features = 0
        # default imputer parameters
        self.imputer_kwargs = imputer_kwargs or {'random_state': 0, 'max_iter': 10}

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        df = df[self.features]

        # Separate numeric and categorical
        self.numeric_features = [
            f for f in self.features if np.issubdtype(df[f].dtype, np.number)]
        self.categorical_features = [
            f for f in self.features if f not in self.numeric_features]

        # Fit imputer + scaler for numeric
        if self.numeric_features:
            self.imputer_ = IterativeImputer(**self.imputer_kwargs)
            self.imputer_.fit(df[self.numeric_features])

            # After imputing, compute scaler (on imputed values)
            imputed = self.imputer_.transform(df[self.numeric_features])
            self.scaler_ = StandardScaler().fit(imputed)

        # Prepare dummy columns for categorical features
        self.cat_dummy_columns_ = {}
        for f in self.categorical_features:
            ser = df[f].fillna("Missing").astype(str) if self.categorical_as_str else df[f]
            dummies = pd.get_dummies(ser, prefix=f)
            self.cat_dummy_columns_[f] = dummies.columns.tolist()

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df = df[self.features]

        parts = []

        # Numeric: impute then scale
        if self.numeric_features:
            num = df[self.numeric_features]
            num_imp = self.imputer_.transform(num)
            num_scaled = self.scaler_.transform(num_imp)
            parts.append(num_scaled.astype(np.float32))

        # Categorical: fillna + one-hot, ensure same columns
        for f in self.categorical_features:
            ser = df[f].fillna("Missing").astype(str) if self.categorical_as_str else df[f]
            dummies = pd.get_dummies(ser, prefix=f)
            cols = self.cat_dummy_columns_[f]

            # Reindex to ensure all expected columns exist; fill missing with 0
            dummies = dummies.reindex(columns=cols, fill_value=0)
            parts.append(dummies.values.astype(np.float32))

        if parts:
            concatenated_parts = np.concatenate(parts, axis=1)
            self.num_features = concatenated_parts.shape[1]
            return concatenated_parts
        else:
            return np.zeros((len(df), 0), dtype=np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

################################################################################
                            # TRAIN FUNCTIONS #
################################################################################

def pretrain_metadata_model(train_df,
                            metadata_processor=None,
                            n_splits=5,
                            save_path=None,
                            verbose=True):
    """
    Pretrains a metadata model using LightGBM + Optuna.
    If save_path exists, loads model/processor from disk instead of retraining.
    """

    # -------------------------------------------------
    # 1) LOAD FROM DISK IF AVAILABLE
    # -------------------------------------------------
    warnings.filterwarnings("ignore", category=UserWarning)
    if os.path.exists(save_path):
        if verbose:
            print(f"Loading pretrained metadata model from: {save_path}")

        saved = joblib.load(save_path)
        return saved

    X = metadata_processor.transform(train_df)
    y = train_df["target"].values

    # --------------------------
    # Optuna objective
    # --------------------------
    def objective(trial):
        params = {
            "objective": "binary",
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2),
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
        }

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        aucs = []

        for tr_idx, val_idx in kf.split(X, y):
            model = lgb.LGBMClassifier(**params, n_jobs=-1, verbosity=-1)

            model.fit(
                X[tr_idx],
                y[tr_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                eval_metric="auc",
            )

            preds = model.predict_proba(X[val_idx])[:, 1]
            aucs.append(roc_auc_score(y[val_idx], preds))

        mean_auc = float(np.mean(aucs))

        if verbose:
            print(f"Trial {trial.number:02d} → AUC: {mean_auc:.5f}")

        return mean_auc

    # --------------------------
    # Run search
    # --------------------------
    if verbose:
        print("\nSearching hyperparameters...\n")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=verbose)

    best_params = study.best_params

    if verbose:
        print("\nBest parameters:", best_params)
        print(f"Best CV AUC: {study.best_value:.5f}")

    # --------------------------
    # Train final model
    # --------------------------
    final_model = lgb.LGBMClassifier(**best_params,  verbosity=-1)
    final_model.fit(X, y)

    # -------------------------------------------------
    # 3) Save using joblib
    # -------------------------------------------------
    leaf_indices_train = final_model.predict(X, pred_leaf=True)
    leaf_dim = leaf_indices_train.max() + 1
    joblib.dump(final_model, save_path)

    return final_model
    
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
        self.lr = 1e-4
        self.optimizer = Adam(self.parameters(), lr=self.lr)
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

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.3)  # gain pequeño
                nn.init.zeros_(m.bias)

        # AMP
        # self.scaler = amp.GradScaler() if device.type == "cuda" else None

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

            # with torch.amp.autocast(device_type=self.device.type):  # AMP
            preds = self(images, metadata).view(-1)
            loss = self.criterion(preds, labels)

            # AMP scaled backward
            # if self.scaler is not None:
            #     self.scaler.scale(loss).backward()
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
            # else:
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

                # with torch.amp.autocast(device_type=self.device.type):  # AMP
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
        print(f"Saved model with val_auc {self.best_model_AUC}")
        torch.save(self.best_model, path)
    
################################################################################
                        # DATA PROCESSING #
################################################################################

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

    Notes:
    - Original images remain unchanged; augmentations occur during __getitem__.
    - Stats from `get_dataset_stats()` are useful for analyzing train/val/test splits.
    """
    def __init__(self, hdf5_path: str, metadata_df: pd.DataFrame,
                 image_transform, metadata_processor: MetadataProcessor):
        
        self.image_transform = image_transform
        self.metadata_processor = metadata_processor

        # 1º Read the metadata
        self.metadata = metadata_df
        self.metadata_ids = self.metadata["isic_id"].to_list()
        self.is_labelled = "target" in self.metadata.columns    # if dataset has a target

        # 2º Read the HDF5 file 
        self.hdf5_file = h5py.File(hdf5_path, "r")

        # 3º Preprocess the metadata
        self.metadata_features = self.metadata_processor.transform(self.metadata)


    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        isic_id = row["isic_id"]
        
        # Load and transform image
        image_rgb = self._load_image_from_hdf5(isic_id)
        image_aug = self.image_transform(image=image_rgb)['image']
        image = torch.from_numpy(image_aug).permute(2, 0, 1).float()

        # Metadata
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

            img = self.image_transform(image=img)["image"]
            img = torch.from_numpy(img).permute(2, 0, 1).float()

            means.append(img.mean(dim=(1, 2)).cpu().numpy())
            stds.append(img.std(dim=(1, 2)).cpu().numpy())

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
