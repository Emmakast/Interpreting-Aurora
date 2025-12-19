import torch
import numpy as np

def run_seasonal_bootstrap(path="results/seasonal_pooled_pca.pt", n_boot=1000):
    print(f"--- Seasonal Bootstrap (N={n_boot}) ---")
    data = torch.load(path)
    
    if "raw_data" not in data:
        print("Error: 'raw_data' missing. Re-run seasonal_pca.py with saving enabled.")
        return

    # Combine all seasons into one tensor for PCA
    raw_dict = data["raw_data"]
    X_all = torch.cat(list(raw_dict.values()), dim=0)
    
    # Original PCA
    mean = X_all.mean(0)
    std = X_all.std(0) + 1e-6
    U, S, Vh_orig = torch.linalg.svd((X_all - mean) / std, full_matrices=False)
    
    # Bootstrap
    sims = np.zeros((n_boot, 3)) # Check top 3 PCs
    N = len(X_all)
    
    for i in range(n_boot):
        # Global resampling (we want to see if the DIRECTIONS are stable across the whole year)
        idx = torch.randint(0, N, (N,))
        X_boot = X_all[idx]
        
        m_b, s_b = X_boot.mean(0), X_boot.std(0) + 1e-6
        U_b, S_b, Vh_b = torch.linalg.svd((X_boot - m_b) / s_b, full_matrices=False)
        
        for k in range(3):
            sims[i, k] = torch.abs(torch.dot(Vh_orig[k], Vh_b[k])).item()
            
    print("\nResults:")
    for k in range(3):
        print(f"PC{k+1}: {np.mean(sims[:,k]):.4f} +/- {np.std(sims[:,k]):.4f}")

if __name__ == "__main__":
    run_seasonal_bootstrap()
