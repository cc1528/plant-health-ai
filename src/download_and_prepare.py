import os
import shutil
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

#########################################################
# CONFIG
#########################################################

# Kaggle dataset slug (from Kaggle)
KAGGLE_DATASET = "charuchaudhry/plantvillage-tomato-leaf-dataset"
# We'll download it once, and then reuse what's already in data/tmp.

# Project root assumed: script is in plant-health-ai/src/
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR   = ROOT / "data"
TMP_DIR    = DATA_DIR / "tmp"          # where Kaggle unzips
RAW_DIR    = DATA_DIR / "raw"          # cleaned class folders go here
SPLITS_DIR = DATA_DIR / "splits"       # final train/val/test split

TRAIN_DIR = SPLITS_DIR / "train"
VAL_DIR   = SPLITS_DIR / "val"
TEST_DIR  = SPLITS_DIR / "test"

# IMPORTANT: in your case the images live under:
# data/tmp/plantvillage/plantvillage/<class_name>/
# So we point here:
PLANT_ROOT = TMP_DIR / "plantvillage" / "plantvillage"

# We only keep 3 classes for now
CLASS_NAME_MAP: Dict[str, str] = {
    "Tomato___healthy": "healthy",
    "Tomato___Early_blight": "early_blight",
    "Tomato___Late_blight": "late_blight",
}

# split ratios
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42


#########################################################
# UTILS
#########################################################

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run_kaggle_download():
    """
    Uses kaggle CLI to download + unzip the dataset into data/tmp.
    If you've already downloaded it manually, you can skip calling this.
    """
    ensure_dir(TMP_DIR)

    print(f"[INFO] Downloading dataset {KAGGLE_DATASET} into {TMP_DIR} ...")
    cmd = [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", str(TMP_DIR),
        "--unzip"
    ]
    subprocess.run(cmd, check=True)
    print("[INFO] Kaggle download + unzip completed.")


def list_images(folder: Path) -> List[Path]:
    imgs = []
    for fname in folder.iterdir():
        low = fname.name.lower()
        if low.endswith(".jpg") or low.endswith(".jpeg") or low.endswith(".png"):
            imgs.append(fname)
    return imgs


def collect_original_class_dirs() -> Dict[str, Path]:
    """
    Look inside PLANT_ROOT and return only the class folders we care about.
    e.g. { "Tomato___healthy": Path(.../Tomato___healthy), ... }
    """
    mapping: Dict[str, Path] = {}

    # sanity check
    if not PLANT_ROOT.exists():
        raise RuntimeError(
            f"Expected {PLANT_ROOT} to exist, but it doesn't. "
            "Check unzip path / folder nesting."
        )

    for item in PLANT_ROOT.iterdir():
        if item.is_dir() and item.name in CLASS_NAME_MAP:
            mapping[item.name] = item

    return mapping


def normalize_raw_folders(class_dirs: Dict[str, Path]):
    """
    Copy images from PLANT_ROOT/<original_class> into data/raw/<clean_label>/.
    After this:
      data/raw/healthy/*.jpg
      data/raw/early_blight/*.jpg
      data/raw/late_blight/*.jpg
    """
    print("[INFO] Normalizing raw class folders into data/raw/")
    ensure_dir(RAW_DIR)

    for original_name, src_dir in class_dirs.items():
        clean_label = CLASS_NAME_MAP[original_name]
        dest_dir = RAW_DIR / clean_label
        ensure_dir(dest_dir)

        imgs = list_images(src_dir)
        print(f"    {original_name} -> {clean_label}: {len(imgs)} images")

        for img_path in imgs:
            dest_path = dest_dir / img_path.name
            shutil.copyfile(img_path, dest_path)

    print("[INFO] Normalization complete.")


def split_list(
    items: List[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> Tuple[List[Path], List[Path], List[Path]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rnd = random.Random(seed)
    shuffled = items[:]
    rnd.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    # leftover goes to test
    n_test  = n - n_train - n_val

    train_files = shuffled[:n_train]
    val_files   = shuffled[n_train:n_train+n_val]
    test_files  = shuffled[n_train+n_val:]

    return train_files, val_files, test_files


def copy_split(files: List[Path], split_root: Path, class_label: str):
    """
    Copies a list of files into split_root/class_label/
    """
    dest_dir = split_root / class_label
    ensure_dir(dest_dir)

    for src_path in files:
        dest_path = dest_dir / src_path.name
        shutil.copyfile(src_path, dest_path)


def build_splits():
    """
    Take data/raw/<class_label> and create:
      data/splits/train/<class_label>/
      data/splits/val/<class_label>/
      data/splits/test/<class_label>/
    """
    print("[INFO] Creating train/val/test splits ...")

    ensure_dir(TRAIN_DIR)
    ensure_dir(VAL_DIR)
    ensure_dir(TEST_DIR)

    summary = []

    for class_label in CLASS_NAME_MAP.values():
        class_raw_dir = RAW_DIR / class_label
        imgs = list_images(class_raw_dir)

        train_files, val_files, test_files = split_list(
            imgs,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            seed=RANDOM_SEED,
        )

        copy_split(train_files, TRAIN_DIR, class_label)
        copy_split(val_files, VAL_DIR, class_label)
        copy_split(test_files, TEST_DIR, class_label)

        summary.append({
            "class_label": class_label,
            "total": len(imgs),
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        })

    print("\n[SUMMARY]")
    for row in summary:
        print(
            f"{row['class_label']:>12} | "
            f"total={row['total']:4d} "
            f"train={row['train']:4d} "
            f"val={row['val']:4d} "
            f"test={row['test']:4d}"
        )

    print("\n[OK] data/splits/ is ready for training 👌")


#########################################################
# MAIN
#########################################################

def main():
    print("=== STEP 1: make base folders ===")
    ensure_dir(DATA_DIR)
    ensure_dir(TMP_DIR)
    ensure_dir(RAW_DIR)
    ensure_dir(SPLITS_DIR)

    # If you've already downloaded once, you can comment out the next 2 lines.
    # Otherwise leave them so first-time users also get the data.
    print("=== STEP 2: download + unzip from Kaggle (if not already downloaded) ===")
    if not PLANT_ROOT.exists():
        run_kaggle_download()
    else:
        print("[INFO] Dataset already present, skipping download.")

    print("=== STEP 3: collect tomato folders from tmp ===")
    class_dirs = collect_original_class_dirs()
    if not class_dirs:
        raise RuntimeError(
            "Did not find expected tomato class folders in tmp/plantvillage/plantvillage. "
            "Check CLASS_NAME_MAP or folder nesting."
        )

    print("=== STEP 4: normalize into data/raw/ (healthy / early_blight / late_blight) ===")
    normalize_raw_folders(class_dirs)

    print("=== STEP 5: make train / val / test ===")
    build_splits()

    print("\nAll done. You can now train using data/splits/train, val, test ✅")


if __name__ == "__main__":
    main()
