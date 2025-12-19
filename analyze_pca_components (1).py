import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

def plot_scatter(scores_dict, x_idx, y_idx, var_ratio, title, filename):
    plt.figure(figsize=(10, 8))
    
    # Define colors
    colors = {
        "Winter": "blue", "Spring": "green", "Summer": "red", "Fall": "orange",
        "storm": "red", "calm": "blue"
    }

    for label, tensor in scores_dict.items():
        s = tensor.numpy()
        color = colors.get(label, 'black')
        plt.scatter(s[:, x_idx], s[:, y_idx], label=label, color=color, alpha=0.5, edgecolors='none', s=20)
        
        # Centroid
        center = s.mean(axis=0)
        plt.scatter(center[x_idx], center[y_idx], c='black', marker='X', s=150, linewidth=1.5, edgecolors='white')
        plt.text(center[x_idx], center[y_idx], label, fontsize=12, fontweight='bold', ha='right')

    xlab = f"PC{x_idx+1} ({var_ratio[x_idx]*100:.1f}%)"
    ylab = f"PC{y_idx+1} ({var_ratio[y_idx]*100:.1f}%)"
    
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title, fontsize=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    plt.close()

def plot_loadings(components, title, filename):
    plt.figure(figsize=(12, 6))
    n_comps = min(3, components.shape[0])
    n_feats = components.shape[1]
    x = np.arange(n_feats)
    zone_size = n_feats // 3
    
    for i in range(n_comps):
        plt.plot(x, np.abs(components[i]), label=f"PC{i+1}")
        
    plt.axvline(zone_size, color='k', linestyle='--', alpha=0.5)
    plt.axvline(zone_size*2, color='k', linestyle='--', alpha=0.5)
    
    max_y = np.abs(components[:3]).max()
    plt.text(zone_size/2, max_y, "MEAN (State)", ha='center', fontweight='bold')
    plt.text(zone_size*1.5, max_y, "STD (Turbulence)", ha='center', fontweight='bold')
    plt.text(zone_size*2.5, max_y, "RMS (Energy)", ha='center', fontweight='bold')

    plt.xlabel("Feature Index")
    plt.ylabel("Absolute Loading")
    plt.title(f"Feature Importance: {title}", fontsize=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    plt.close()

def plot_contrastive_pair(score_dict, label_a, label_b, color_a, color_b, filename):
    """
    Calculates the separation vector between two classes in PCA space
    and plots their projection onto that vector.
    """
    if label_a not in score_dict or label_b not in score_dict:
        print(f"Skipping {label_a} vs {label_b}: labels not found in data.")
        return

    # Get scores (N, D)
    A = score_dict[label_a].float()
    B = score_dict[label_b].float()

    # Calculate separation vector (Difference of Centroids)
    # This represents the "Axis of Best Separation" between the two groups
    diff = A.mean(dim=0) - B.mean(dim=0)
    diff = diff / (diff.norm() + 1e-8) # Normalize

    # Project scores onto this separation axis
    proj_A = (A @ diff).numpy()
    proj_B = (B @ diff).numpy()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(proj_A, fill=True, color=color_a, label=f'{label_a}', alpha=0.5)
    sns.kdeplot(proj_B, fill=True, color=color_b, label=f'{label_b}', alpha=0.5)
    
    # Plot means
    plt.axvline(proj_A.mean(), color=color_a, linestyle='--', alpha=0.8)
    plt.axvline(proj_B.mean(), color=color_b, linestyle='--', alpha=0.8)

    plt.title(f"Contrastive Projection: {label_a} vs {label_b}", fontsize=20)
    plt.xlabel("Score along Separation Axis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to .pt results file")
    parser.add_argument("--mode", type=str, choices=["seasonal", "storm"], required=True)
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print("File not found.")
        return

    data = torch.load(args.file, map_location="cpu")
    
    # Normalize data structure
    scores_dict = {}
    if args.mode == "seasonal":
        scores_dict = data["scores"]
    else:
        scores_dict["storm"] = data["scores_storm"]
        scores_dict["calm"] = data["scores_calm"]
        
    var_ratio = data["explained_variance_ratio"].numpy()
    components = data["components"].numpy()

    # 1. Scatter Plots
    plot_scatter(scores_dict, 0, 1, var_ratio, 
                 f"{args.mode.title()} PCA: PC1 vs PC2", 
                 f"results/{args.mode}_pc1_pc2.png")

    plot_scatter(scores_dict, 1, 2, var_ratio, 
                 f"{args.mode.title()} PCA: PC2 vs PC3", 
                 f"results/{args.mode}_pc2_pc3.png")
                 
    # 2. Feature Loadings
    plot_loadings(components, f"{args.mode.title()} PCA Components", 
                  f"results/{args.mode}_loadings.png")

    # 3. Contrastive Plots (Specific Pairs)
    if args.mode == "seasonal":
        print("Generating Seasonal Contrastive Plots...")
        # Summer vs Winter (Temperature Axis)
        plot_contrastive_pair(scores_dict, "Summer", "Winter", "red", "blue", 
                              "results/seasonal_contrast_summer_winter.png")
        # Fall vs Spring (Transition Axis)
        plot_contrastive_pair(scores_dict, "Fall", "Spring", "orange", "green", 
                              "results/seasonal_contrast_fall_spring.png")
                              
    elif args.mode == "storm" and "proj_storm_contrastive" in data:
        # For Storms, we already have the pre-calculated projection
        print("Generating Storm Contrastive Plot...")
        plt.figure(figsize=(10, 6))
        p_storm = data["proj_storm_contrastive"].numpy()
        p_calm = data["proj_calm_contrastive"].numpy()
        
        sns.kdeplot(p_storm, fill=True, color='red', label='Storm')
        sns.kdeplot(p_calm, fill=True, color='blue', label='Calm')
        plt.title("Contrastive Projection (Storm vs Calm)", fontsize=20)
        plt.xlabel("Contrastive Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("results/storm_contrastive.png")
        print("Saved results/storm_contrastive.png")

if __name__ == "__main__":
    main()