import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import imagehash
import argparse
from skimage.metrics import structural_similarity as ssim


# --- Utility functions ---------------------------------------------------------

def open_grayscale_image(filepath):
    """Open an image as grayscale."""
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)


def image_similarity_score(img1, img2):
    """SSIM score between two images."""
    if img1 is None or img2 is None or img1.shape != img2.shape:
        return 0.0
    score = ssim(img1, img2)
    return score


def compute_image_hash(filepath, hash_func=imagehash.phash):
    """Compute a perceptual hash (phash by default) for an image file."""
    try:
        with Image.open(filepath) as img:
            return str(hash_func(img))
    except Exception:
        return None


def hamming_distance(hash1, hash2):
    """Compute the Hamming distance between two perceptual hashes."""
    try:
        h1, h2 = imagehash.hex_to_hash(hash1), imagehash.hex_to_hash(hash2)
        return h1 - h2
    except Exception:
        return np.inf


# --- Group flagging ------------------------------------------------------------

def flag_multi_image_groups(df):
    """
    Flag rows where (event_id, image_nr) occurs more than once.
    Adds column: 'multi_image_flag' = 'duplicate' or 'unique'.
    """
    df = df.copy()
    combo_counts = df.groupby(["event_id", "image_nr"])["image_nr"].transform("count")
    df["multi_image_flag"] = np.where(combo_counts > 1, "duplicate", "unique")
    return df


# --- Caching and comparison ----------------------------------------------------

def preload_group_images(group, method="ssim"):
    """
    Preload all images in a group once and store in a dict.
    Returns: {index: image data or image hash (depending on method)}
    """
    cache = {}
    for idx, row in group.iterrows():
        filepath = os.path.join(row.root, row.dataset_id, row.rec_path)
        try:
            if method == "ssim":
                cache[idx] = open_grayscale_image(filepath)
            elif method == "hash":
                cache[idx] = compute_image_hash(filepath)
        except Exception:
            cache[idx] = None
    return cache


def compare_images_in_group(group, cache, method="ssim", sim_threshold=0.99, hash_distance_threshold=0):
    """
    Compare images within a group using cached data (SSIM or hash).

    Returns
    -------
    list of (idx1, idx2, flag)
    """
    results = []
    for i, (idx1, row1) in enumerate(group.iterrows()):
        for idx2, row2 in group.iloc[i + 1 :].iterrows():
            # same dataset_id + rec_path
            if (row1.dataset_id == row2.dataset_id) and (row1.rec_path == row2.rec_path):
                results.append((idx1, idx2, "duplicate_paths"))
                continue

            img1, img2 = cache.get(idx1), cache.get(idx2)
            if img1 is None or img2 is None:
                continue

            if method == "ssim":
                similarity = image_similarity_score(img1, img2)
                if similarity >= sim_threshold:
                    results.append((idx1, idx2, "duplicate_visual"))
                    continue

            elif method == "hash":
                distance = hamming_distance(img1, img2)
                if distance <= hash_distance_threshold:
                    results.append((idx1, idx2, "duplicate_hash"))
                    continue
            
            results.append((idx1, idx2, "unknown"))
            

    return results


# --- Main orchestration --------------------------------------------------------

def flag_duplicate_images(df, method="ssim", sim_threshold=0.99, hash_distance_threshold=0):
    """
    Detect duplicate images within (event_id, image_nr) groups.

    Parameters
    ----------
    df : pandas.DataFrame
        Must include columns: ['root', 'dataset_id', 'rec_path', 'event_id', 'image_nr']
    method : str, optional
        'ssim' for pixel-level comparison (default), 'hash' for perceptual hashing.
    sim_threshold : float, optional
        SSIM similarity threshold.
    hash_distance_threshold : int, optional
        Max Hamming distance for hash-based duplicates.

    Returns
    -------
    pandas.DataFrame
        DataFrame with additional column 'duplicate_flag'.
    """
    assert method in ("ssim", "hash"), "method must be 'ssim' or 'hash'"

    df = flag_multi_image_groups(df)
    df = df.copy()
    df["duplicate_flag"] = "unique"

    multi_df = df[df["multi_image_flag"] == "duplicate"]
    grouped = multi_df.groupby(["event_id", "image_nr"])

    for (event_id, image_nr), group in tqdm(grouped, desc=f"Processing groups ({method})"):
        cache = preload_group_images(group, method=method)
        results = compare_images_in_group(
            group,
            cache,
            method=method,
            sim_threshold=sim_threshold,
            hash_distance_threshold=hash_distance_threshold,
        )

        for idx1, idx2, flag in results:
            df.loc[[idx1, idx2], "duplicate_flag"] = flag

    return df


# --- Example usage -------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for duplicate flagging')
    parser.add_argument('--root', default='Z:/marvel/marvel-fhnw/data/Poleno_25', type=str)
    parser.add_argument('--in_filename', default='data/combined/poleno/poleno_labels.csv', type=str)
    parser.add_argument('--out_filename', default='data/combined/poleno/poleno_labels_flagged.csv', type=str)
    args = parser.parse_args()

    import pandas as pd

    df_poleno = pd.read_csv(args.in_filename)
    df_poleno["root"] = args.root

    # 'ssim' (pixel-level) or 'hash' (fast perceptual)
    method = "hash"

    # Run duplicate detection
    df_poleno = flag_duplicate_images(df_poleno, method=method, sim_threshold=0.99, hash_distance_threshold=0)

    df_poleno = df_poleno.drop(columns="root")

    df_poleno.to_csv(args.out_filename, index=False)