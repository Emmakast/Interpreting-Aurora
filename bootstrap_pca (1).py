import torch
import numpy as np

def run_bootstrap_validation(path="results/storm_calm_pooled_results.pt", n_boot=1000, n_components=3):
    print(f"--- Running Bootstrap Validation (N={n_boot}) ---")
    data = torch.load(path)
    
    # Load raw vectors saved in previous step
    if "raw_storm" not in data or "raw_calm" not in data:
        print("Error: Raw vectors not found in .pt file. Please re-run joint_pooled_pca.py with the 'raw' save update.")
        return

    X_storm = data["raw_storm"]
    X_calm = data["raw_calm"]
    
    # 1. Compute ORIGINAL PCA (The Reference)
    X_all = torch.cat([X_storm, X_calm], dim=0)
    mean, std = X_all.mean(0), X_all.std(0) + 1e-6
    # Compute full SVD on original data
    U_orig, S_orig, Vh_orig = torch.linalg.svd((X_all - mean) / std, full_matrices=False)
    
    # Store similarities for each component
    # shape: (n_boot, n_components)
    all_similarities = np.zeros((n_boot, n_components))
    
    for i in range(n_boot):
        # Resample indices with replacement
        idx_s = torch.randint(0, len(X_storm), (len(X_storm),))
        idx_c = torch.randint(0, len(X_calm), (len(X_calm),))
        
        X_boot = torch.cat([X_storm[idx_s], X_calm[idx_c]], dim=0)
        
        # Re-compute PCA on bootstrap sample
        mean_b, std_b = X_boot.mean(0), X_boot.std(0) + 1e-6
        U_b, S_b, Vh_boot = torch.linalg.svd((X_boot - mean_b) / std_b, full_matrices=False)
        
        # Compare top 'n_components'
        for k in range(n_components):
            # Dot product of Original Vector k vs Bootstrap Vector k
            # We take ABS because PCA sign is arbitrary (v and -v are the same axis)
            sim = torch.abs(torch.dot(Vh_orig[k], Vh_boot[k])).item()
            all_similarities[i, k] = sim
            
    # Print Results
    print("\n--- Stability Results (Mean Cosine Similarity) ---")
    for k in range(n_components):
        means = np.mean(all_similarities[:, k])
        stds = np.std(all_similarities[:, k])
        print(f"PC{k+1}: {means:.4f} ± {stds:.4f}")
        
        if means > 0.9:
            print(f"     -> Excellent stability. PC{k+1} is a robust structural feature.")
        elif means > 0.7:
            print(f"     -> Good stability. PC{k+1} is likely real but has some variance.")
        else:
            print(f"     -> Low stability. PC{k+1} might be noise or mixing with other PCs.")

if __name__ == "__main__":
    run_bootstrap_validation()