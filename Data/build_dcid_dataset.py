import os
import random
from PIL import Image


def build_full_dataset(root, R, C, L, I, out_dir, val_split=0.15, seed=42):
    """
    Build a resized and RWDA-augmented version of the Drill Core Image Dataset (DCID).

    Parameters:
    - root (str): Root directory that contains the original high-resolution folders
                  (e.g., 'DCID-512-7' and 'noise-512-7').
    - R (int): Target resolution for output images (e.g., 32, 64, 128, 256).
    - C (int): Number of lithology categories. Options: 7 or 35.
    - L (float): Real-world data augmentation (RWDA) level, i.e., proportion of defective samples to inject (0.0–0.4).
    - I (str): RWDA injection scope:
        - 'N' = None (no injection),
        - 'T' = Train set only,
        - 'E' = Test set only,
        - 'A' = All (both train and test sets).
    - out_dir (str): Output base directory to save the generated dataset.
    - val_split (float): Proportion of the original train set to use as validation
                         (default: 0.15). Val is carved from train before any
                         defect injection, so val always contains clean images
                         unless I='A'.
    - seed (int): Random seed for reproducibility (default: 42).

    The original images (always read from 512x512 resolution) are resized to the
    target resolution R. Slightly defective samples are injected into the selected
    splits based on L and I parameters.

    Output folder structure:
        out_dir/
        └── DCID-{R}-{C}-{L}-{I}/
            ├── train/
            │   └── <class>/
            ├── val/
            │   └── <class>/
            └── test/
                └── <class>/

    Reference:
    Jia-Yu Li, Ji-Zhou Tang, Xian-Zheng Zhao, Bo Fan, Wen-Ya Jiang, Shun-Yao Song,
    Jian-Bing Li, Kai-Da Chen, Zheng-Guang Zhao.
    A large-scale, high-quality dataset for lithology identification: Construction
    and applications. Petroleum Science, 2025. ISSN 1995-8226.
    https://doi.org/10.1016/j.petsci.2025.04.013
    """

    assert I in {"N", "T", "E", "A"}, "I must be one of 'N', 'T', 'E', 'A'"
    assert C in {7, 35}, "C must be 7 or 35"
    assert 0.0 < val_split < 1.0, "val_split must be between 0 and 1"

    random.seed(seed)

    orig_dir   = os.path.join(root, f"DCID-512-{C}")
    noise_dir  = os.path.join(root, f"noise-512-{C}")
    target_dir = os.path.join(out_dir, f"Dataset-{C}_R-{R}_L-{L}_I-{I}")

    os.makedirs(target_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # STEP 1: Process the source test split → output test/               #
    # ------------------------------------------------------------------ #
    _process_split(
        split_name      = "test",
        src_orig_dir    = os.path.join(orig_dir,  "test"),
        src_noise_dir   = os.path.join(noise_dir, "test"),
        out_split_dir   = os.path.join(target_dir, "test"),
        R               = R,
        L               = L,
        inject          = I in {"E", "A"},
    )

    # ------------------------------------------------------------------ #
    # STEP 2: Load source train images per class, then split into        #
    #         train / val before any defect injection.                    #
    # ------------------------------------------------------------------ #
    src_train_dir  = os.path.join(orig_dir,  "train")
    src_noise_train = os.path.join(noise_dir, "train")

    print("Splitting source train set into train/val...")
    for cls in sorted(os.listdir(src_train_dir)):
        orig_cls_path  = os.path.join(src_train_dir, cls)
        if not os.path.isdir(orig_cls_path):
            continue

        all_imgs = sorted(os.listdir(orig_cls_path))
        random.shuffle(all_imgs)

        n_val   = int(len(all_imgs) * val_split)
        val_imgs   = all_imgs[:n_val]
        train_imgs = all_imgs[n_val:]

        # ---- Write val images (clean only, no defects unless I='A') ----
        out_val_cls = os.path.join(target_dir, "val", cls)
        os.makedirs(out_val_cls, exist_ok=True)
        _resize_and_save(orig_cls_path, val_imgs, out_val_cls, R)

        if I == "A":
            noise_cls_path = os.path.join(src_noise_train, cls)
            _inject_defects(noise_cls_path, out_val_cls, n_val, L, R)

        # ---- Write train images + optional defect injection ------------
        out_train_cls = os.path.join(target_dir, "train", cls)
        os.makedirs(out_train_cls, exist_ok=True)
        _resize_and_save(orig_cls_path, train_imgs, out_train_cls, R)

        if I in {"T", "A"}:
            noise_cls_path = os.path.join(src_noise_train, cls)
            _inject_defects(noise_cls_path, out_train_cls, len(train_imgs), L, R)

        print(f"  {cls}: {len(train_imgs)} train | {len(val_imgs)} val")

    print("train/val split finished.\n")

    # ------------------------------------------------------------------ #
    # STEP 3: Print final summary                                         #
    # ------------------------------------------------------------------ #
    print("=" * 45)
    print("Dataset summary")
    print("=" * 45)
    grand_total = 0
    for split in ["train", "val", "test"]:
        split_path = os.path.join(target_dir, split)
        split_total = 0
        for cls in sorted(os.listdir(split_path)):
            cls_path = os.path.join(split_path, cls)
            n = len(os.listdir(cls_path))
            print(f"  {split}/{cls}: {n}")
            split_total += n
        print(f"  {split} total: {split_total}\n")
        grand_total += split_total
    print(f"Grand total: {grand_total} images")
    print(f"Output:      {target_dir}")


# ------------------------------------------------------------------ #
# Helper functions                                                    #
# ------------------------------------------------------------------ #

def _resize_and_save(src_cls_path, img_list, out_cls_path, R):
    """Resize a list of images from src_cls_path and save to out_cls_path."""
    for img_name in img_list:
        src = os.path.join(src_cls_path, img_name)
        dst = os.path.join(out_cls_path, img_name)
        with Image.open(src) as im:
            im = im.resize((R, R), Image.BILINEAR)
            im.save(dst)


def _inject_defects(noise_cls_path, out_cls_path, n_clean, L, R):
    """Sample and inject defective images into out_cls_path."""
    if not os.path.isdir(noise_cls_path):
        print(f"  Warning: noise path not found: {noise_cls_path}")
        return

    n_defect   = int(L * n_clean)
    defect_imgs = os.listdir(noise_cls_path)
    sampled = (
        random.choices(defect_imgs, k=n_defect)
        if n_defect > len(defect_imgs)
        else random.sample(defect_imgs, k=n_defect)
    )
    for i, img_name in enumerate(sampled):
        src = os.path.join(noise_cls_path, img_name)
        dst = os.path.join(out_cls_path, f"defect_{i}_{img_name}")
        with Image.open(src) as im:
            im = im.resize((R, R), Image.BILINEAR)
            im.save(dst)


def _process_split(split_name, src_orig_dir, src_noise_dir, out_split_dir, R, L, inject):
    """Process a full source split (used for test)."""
    print(f"Processing {split_name} set | inject defect: {inject}")
    for cls in sorted(os.listdir(src_orig_dir)):
        orig_cls_path  = os.path.join(src_orig_dir,  cls)
        noise_cls_path = os.path.join(src_noise_dir, cls)
        out_cls_path   = os.path.join(out_split_dir, cls)
        os.makedirs(out_cls_path, exist_ok=True)

        all_imgs = os.listdir(orig_cls_path)
        _resize_and_save(orig_cls_path, all_imgs, out_cls_path, R)

        if inject:
            _inject_defects(noise_cls_path, out_cls_path, len(all_imgs), L, R)

    print(f"{split_name} set finished.\n")



def build_test_set(root, R, C, L, out_dir, seed=42, I = 'E'):
    """
    Build a resized and RWDA-augmented version of the Drill Core Image Dataset (DCID) - only the test-set.
    
    Parameters:
    - root (str): Root directory that contains the original high-resolution folders
                  (e.g., 'DCID-512-7' and 'noise-512-7').
    - R (int): Target resolution for output images (e.g., 32, 64, 128, 256).
    - C (int): Number of lithology categories. Options: 7 or 35.
    - L (float): Real-world data augmentation (RWDA) level, i.e., proportion of defective samples to inject (0.0–0.4).
    - I (str): RWDA injection scope:
        - 'N' = None (no injection),
        - 'T' = Train set only,
        - 'E' = Test set only,
        - 'A' = All (both train and test sets).
    - out_dir (str): Output base directory to save the generated dataset.
    - seed (int): Random seed for reproducibility (default: 42).

    The original images (always read from 512x512 resolution) are resized to the
    target resolution R. Slightly defective samples are injected into the selected
    splits based on L and I parameters.
    """
    assert C in {7, 35}, "C must be 7 or 35"

    random.seed(seed)

    orig_dir   = os.path.join(root, f"DCID-512-{C}")
    noise_dir  = os.path.join(root, f"noise-512-{C}")
    target_dir = os.path.join(out_dir, f"Dataset-{C}_R-{R}_L-{L}_I-{I}")

    os.makedirs(target_dir, exist_ok=True)
    _process_split(
    split_name      = "test",
    src_orig_dir    = os.path.join(orig_dir,  "test"),
    src_noise_dir   = os.path.join(noise_dir, "test"),
    out_split_dir   = os.path.join(target_dir, "test"),
    R               = R,
    L               = L,
    inject          = I in {"E", "A"},
    )


def build_train_val_dataset(root, R, C, L, I, out_dir, val_split=0.15, seed=42):
    """
    Build a resized and RWDA-augmented version of the Drill Core Image Dataset (DCID).

    Parameters:
    - root (str): Root directory that contains the original high-resolution folders
                  (e.g., 'DCID-512-7' and 'noise-512-7').
    - R (int): Target resolution for output images (e.g., 32, 64, 128, 256).
    - C (int): Number of lithology categories. Options: 7 or 35.
    - L (float): Real-world data augmentation (RWDA) level, i.e., proportion of defective samples to inject (0.0–0.4).
    - I (str): RWDA injection scope:
        - 'N' = None (no injection),
        - 'T' = Train set only,
        - 'E' = Test set only,
        - 'A' = All (both train and test sets).
    - out_dir (str): Output base directory to save the generated dataset.
    - val_split (float): Proportion of the original train set to use as validation
                         (default: 0.15). Val is carved from train before any
                         defect injection, so val always contains clean images
                         unless I='A'.
    - seed (int): Random seed for reproducibility (default: 42).

    The original images (always read from 512x512 resolution) are resized to the
    target resolution R. Slightly defective samples are injected into the selected
    splits based on L and I parameters.

    Output folder structure:
        out_dir/
        └── DCID-{R}-{C}-{L}-{I}/
            ├── train/
            │   └── <class>/
            ├── val/
            │   └── <class>/
            └── test/
                └── <class>/

    Reference:
    Jia-Yu Li, Ji-Zhou Tang, Xian-Zheng Zhao, Bo Fan, Wen-Ya Jiang, Shun-Yao Song,
    Jian-Bing Li, Kai-Da Chen, Zheng-Guang Zhao.
    A large-scale, high-quality dataset for lithology identification: Construction
    and applications. Petroleum Science, 2025. ISSN 1995-8226.
    https://doi.org/10.1016/j.petsci.2025.04.013
    """

    assert I in {"N", "T", "E", "A"}, "I must be one of 'N', 'T', 'E', 'A'"
    assert C in {7, 35}, "C must be 7 or 35"
    assert 0.0 < val_split < 1.0, "val_split must be between 0 and 1"

    random.seed(seed)

    orig_dir   = os.path.join(root, f"DCID-512-{C}")
    noise_dir  = os.path.join(root, f"noise-512-{C}")
    target_dir = os.path.join(out_dir, f"Dataset-{C}_R-{R}_L-{L}_I-{I}")

    os.makedirs(target_dir, exist_ok=True)

    src_train_dir  = os.path.join(orig_dir,  "train")
    src_noise_train = os.path.join(noise_dir, "train")

    print("Splitting source train set into train/val...")
    for cls in sorted(os.listdir(src_train_dir)):
        orig_cls_path  = os.path.join(src_train_dir, cls)
        if not os.path.isdir(orig_cls_path):
            continue

        all_imgs = sorted(os.listdir(orig_cls_path))
        random.shuffle(all_imgs)

        n_val   = int(len(all_imgs) * val_split)
        val_imgs   = all_imgs[:n_val]
        train_imgs = all_imgs[n_val:]

        # ---- Write val images (clean only, no defects unless I='A') ----
        out_val_cls = os.path.join(target_dir, "val", cls)
        os.makedirs(out_val_cls, exist_ok=True)
        _resize_and_save(orig_cls_path, val_imgs, out_val_cls, R)

        if I == "A":
            noise_cls_path = os.path.join(src_noise_train, cls)
            _inject_defects(noise_cls_path, out_val_cls, n_val, L, R)

        # ---- Write train images + optional defect injection ------------
        out_train_cls = os.path.join(target_dir, "train", cls)
        os.makedirs(out_train_cls, exist_ok=True)
        _resize_and_save(orig_cls_path, train_imgs, out_train_cls, R)

        if I in {"T", "A"}:
            noise_cls_path = os.path.join(src_noise_train, cls)
            _inject_defects(noise_cls_path, out_train_cls, len(train_imgs), L, R)

        print(f"  {cls}: {len(train_imgs)} train | {len(val_imgs)} val")

    print("train/val split finished.\n")


