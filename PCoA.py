import os
import pandas as pd
import numpy as np

# ===================== Global Configuration (Edit only this section) =====================
CPU_CORES = 60
N_PC_LIMIT = 100  # Max number of PCs to output
CALCULATE_R_EXPLAINED = False

# CPU thread optimization
os.environ["OMP_NUM_THREADS"] = str(CPU_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_CORES)
os.environ["MKL_NUM_THREADS"] = str(CPU_CORES)

# All combinations (MATCHES your distance matrix script)
PRETREAT_METHODS = ["raw", "log", "clr", "hellinger", "arcsine", "zscore"]
DISTANCE_METHODS = ["euclidean", "jaccard", "braycurtis", "jensenshannon"]
LEVELS = ["phylum", "class", "order", "family", "genus", "species"]

# ===================== PCoA Function (Vegan-compatible) =====================
def pcoa_vegan_compatible(distance_df):
    """
    PCoA implementation fully compatible with vegan::cmdscale in R
    Input: Distance matrix (DataFrame, n_samples × n_samples)
    Output: Coordinates, eigenvalues, variance explained
    """
    D = distance_df.values
    n = D.shape[0]

    # Double centering (same as vegan)
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D ** 2 @ H

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(B)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Keep only positive eigenvalues
    pos_mask = eigvals > 1e-10
    eigvals = eigvals[pos_mask]
    eigvecs = eigvecs[:, pos_mask]

    # Compute PCoA coordinates
    coords = eigvecs * np.sqrt(eigvals)

    # Variance explained
    r_exp = eigvals / eigvals.sum() if eigvals.sum() > 0 else np.zeros_like(eigvals)

    return coords, eigvals, r_exp

# ===================== Run PCoA for one combination =====================
def run_single_pcoa(pretreat, dist_method, level):
    input_file = f"{pretreat}_{dist_method}_distance_{level}.txt"
    output_coord = f"{pretreat}_{dist_method}_PCoA_coord_{level}.txt"
    output_exp = f"{pretreat}_{dist_method}_PCoA_exp_{level}.txt"

    print(f"▶ Running PCoA: {pretreat:8s} | {dist_method:12s} | {level:8s}")

    # Load distance matrix
    try:
        distance_df = pd.read_csv(input_file, sep="\t", index_col=0)
    except FileNotFoundError:
        print(f"❌ Missing: {input_file}\n")
        return

    # Run PCoA
    coords, eigvals, r_exp = pcoa_vegan_compatible(distance_df)
    n_pcs = min(N_PC_LIMIT, len(eigvals))

    # Save coordinates
    coord_df = pd.DataFrame(
        coords[:, :n_pcs],
        index=distance_df.index,
        columns=[f"PC{i+1}" for i in range(n_pcs)]
    )
    coord_df["Distance"] = dist_method.capitalize()
    coord_df["Pretreat"] = pretreat.capitalize()
    coord_df["Level"] = level.capitalize()
    coord_df.to_csv(output_coord, sep="\t", index_label="SampleID")

    # Save variance explained
    exp_data = {
        "PC": [f"PC{i+1}" for i in range(n_pcs)],
        "VarianceExplained": r_exp[:n_pcs],
        "CumulativeVarianceExplained": np.cumsum(r_exp[:n_pcs])
    }
    exp_df = pd.DataFrame(exp_data)
    exp_df["Distance"] = dist_method.capitalize()
    exp_df["Pretreat"] = pretreat.capitalize()
    exp_df["Level"] = level.capitalize()
    exp_df.to_csv(output_exp, sep="\t", index=False)

    print(f"✅ Saved coordinates: {output_coord}")
    print(f"✅ Saved variance:    {output_exp}\n")

# ===================== Batch run all combinations =====================
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 BATCH PCoA ANALYSIS FOR ALL DISTANCE MATRICES")
    print("📊 6 Normalizations × 4 Distances × 6 Taxonomic Levels")
    print(f"🧮 Total tasks: {len(PRETREAT_METHODS) * len(DISTANCE_METHODS) * len(LEVELS)}")
    print("=" * 80)

    # Auto-run all combinations
    for pretreat in PRETREAT_METHODS:
        for dist in DISTANCE_METHODS:
            for lv in LEVELS:
                run_single_pcoa(pretreat, dist, lv)

    print("🎉 All PCoA analyses completed successfully!")
