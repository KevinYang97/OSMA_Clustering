import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore

# ===================== Global Configuration (Edit only this section) =====================
# Input file prefix (sylph_xxx_profiles.txt)
INPUT_PREFIX = "sylph"

# Supported preprocessing, distance metrics, and taxonomic levels
PRETREAT_METHODS = ["raw", "log", "clr", "hellinger", "arcsine", "zscore"]
DISTANCE_METHODS = ["euclidean", "jaccard", "braycurtis", "jensen_shannon"]
LEVELS = ["phylum", "class", "order", "family", "genus", "species"]

# CPU thread optimization
CPU_CORES = 60
os.environ["OMP_NUM_THREADS"] = str(CPU_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_CORES)

# ===================== Data Normalization Functions =====================
def normalize_abundance(df, method):
    """
    Apply 6 types of data transformation to abundance table
    - raw: no transformation
    - log: log(x + 1)
    - clr: centered log-ratio transformation
    - hellinger: Hellinger transformation
    - arcsine: arcsine square root transformation
    - zscore: Z-score normalization (per sample)
    """
    mat = df.copy().values
    eps = 1e-8

    if method == "raw":
        return df

    elif method == "log":
        mat = np.log1p(mat)
        return pd.DataFrame(mat, index=df.index, columns=df.columns)

    elif method == "clr":
        mat = mat + eps
        gm = np.exp(np.log(mat).mean(axis=1, keepdims=True))
        mat = np.log(mat / gm)
        return pd.DataFrame(mat, index=df.index, columns=df.columns)

    elif method == "hellinger":
        mat = np.sqrt(mat / (mat.sum(axis=1, keepdims=True) + eps))
        return pd.DataFrame(mat, index=df.index, columns=df.columns)

    elif method == "arcsine":
        mat = np.sqrt(mat / (mat.sum(axis=1, keepdims=True) + eps))
        mat = np.arcsin(mat)
        return pd.DataFrame(mat, index=df.index, columns=df.columns)

    elif method == "zscore":
        mat = zscore(mat, axis=1, nan_policy="omit")
        return pd.DataFrame(mat, index=df.index, columns=df.columns)

    else:
        raise ValueError(f"Unsupported normalization method: {method}")

# ===================== Distance Matrix Calculation =====================
def compute_distance_matrix(df, dist_method):
    """Calculate distance matrix using 4 common ecological metrics"""
    if dist_method == "jensen_shannon":
        # Normalize to proportions before JS distance calculation
        df_norm = df.div(df.sum(axis=1), axis=0) + 1e-8
        dist = pdist(df_norm, metric="jensenshannon")
    else:
        dist = pdist(df, metric=dist_method)

    mat = squareform(dist)
    return pd.DataFrame(mat, index=df.index, columns=df.index)

# ===================== Core Processing for One Taxonomic Level =====================
def process_single_level(level):
    """
    Read abundance profile ONCE, then run ALL normalizations & distances
    This drastically reduces I/O time
    """
    input_file = f"{INPUT_PREFIX}_{level}_profiles.txt"
    print(f"\n📂 Loading data for level: {level}")

    # ----------------------- KEY OPTIMIZATION -----------------------
    # Read file ONLY ONCE per level
    try:
        df = pd.read_csv(input_file, sep="\t", index_col=0)
        print(f"✅ Successfully loaded: {input_file}")
    except FileNotFoundError:
        print(f"❌ Missing input file: {input_file}, skipping...")
        return

    # Run ALL preprocessing + ALL distance methods
    for pretreat in PRETREAT_METHODS:
        # Normalize once
        df_norm = normalize_abundance(df, pretreat)

        # Calculate all distances for this normalized data
        for dist_method in DISTANCE_METHODS:
            output_file = f"{pretreat}_{dist_method}_distance_{level}.txt"
            print(f"▶ Calculating: {pretreat:8s} | {dist_method:15s} | {level:8s}")

            # Compute and save
            dist_df = compute_distance_matrix(df_norm, dist_method)
            dist_df.to_csv(output_file, sep="\t", float_format="%.6f")
            print(f"✅ Saved: {output_file}")

# ===================== Main Entry =====================
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 Batch Distance Matrix Calculation | Optimized I/O Version")
    print("📌 Strategy: Read each taxonomic level ONCE, compute all transforms/distances")
    print(f"🧮 Total tasks: {len(PRETREAT_METHODS) * len(DISTANCE_METHODS) * len(LEVELS)}")
    print("=" * 80)

    # Process each level once (FAST)
    for level in LEVELS:
        process_single_level(level)

    print("\n🎉 All distance matrix calculations completed successfully!")