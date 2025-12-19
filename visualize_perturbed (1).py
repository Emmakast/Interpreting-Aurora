import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to prediction_impact.pt")
    parser.add_argument("--output", default="results/storm_impact_visual.png")
    args = parser.parse_args()
    
    print(f"Loading {args.file}...")
    data = torch.load(args.file)
    
    # 1. Prepare Data
    # Raw output is approx Pa - 90000. Convert to hPa.
    # (Val + 90000) / 100 = Val/100 + 900
    
    msl_clean = data["msl_clean"].numpy()
    if msl_clean.ndim == 3: msl_clean = msl_clean[-1]
    msl_clean = msl_clean / 100.0 + 900.0

    msl_pert = data["msl_pert"].numpy()
    if msl_pert.ndim == 3: msl_pert = msl_pert[-1]
    msl_pert = msl_pert / 100.0 + 900.0
    
    diff = msl_clean - msl_pert
    
    lats = data["lats"].numpy()
    lons = data["lons"].numpy()
    
    # Shift to -180..180 for easier plotting over Atlantic
    # lons is 0..360. 
    # We want to roll the data so 0 deg is in the middle or handled correctly.
    # Actually, Cartopy handles 0..360 fine, but let's use contourf for smoothing.
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # 2. Setup Plot (Europe View)
    fig = plt.figure(figsize=(24, 7))
    proj = ccrs.PlateCarree()
    extent = [-40, 40, 30, 75] 

    def plot_panel(ax, field, title, cmap, vmin=None, vmax=None):
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black', facecolor='none')
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, edgecolor='black', facecolor='none')
        
        # Use contourf for smoothing
        # We need to handle the cyclic point or just plot. 
        # Since we zoom in, we don't need full global cyclic point, but we need continuity at 0.
        
        levels = np.linspace(vmin, vmax, 20)
        im = ax.contourf(lon_grid, lat_grid, field, levels=levels, transform=ccrs.PlateCarree(), 
                         cmap=cmap, extend='both')
        
        # Add contours lines
        if "Difference" not in title:
            ax.contour(lon_grid, lat_grid, field, transform=ccrs.PlateCarree(),
                       levels=np.arange(960, 1040, 4), colors='black', linewidths=0.5, alpha=0.5)
            
        plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
        ax.set_title(title, fontsize=16, fontweight='bold')
        return ax

    # Panel 1: Original
    ax1 = fig.add_subplot(1, 3, 1, projection=proj)
    # Use dynamic limits based on data to ensure visibility
    vmin = min(msl_clean.min(), msl_pert.min())
    vmax = max(msl_clean.max(), msl_pert.max())
    print(f"Dynamic Plot Limits: vmin={vmin:.2f}, vmax={vmax:.2f}")
    
    plot_panel(ax1, msl_clean, "Original Forecast (Clean)", "coolwarm_r", vmin=vmin, vmax=vmax)
    
    # Panel 2: Perturbed
    ax2 = fig.add_subplot(1, 3, 2, projection=proj)
    plot_panel(ax2, msl_pert, "Perturbed Forecast (Relevant Pixels Masked)", "coolwarm_r", vmin=vmin, vmax=vmax)
    
    # Panel 3: Difference
    ax3 = fig.add_subplot(1, 3, 3, projection=proj)
    # Red = Pressure INCREASED (Storm weakened/filled up)
    # Blue = Pressure DECREASED
    limit = np.percentile(np.abs(diff), 99.5)
    plot_panel(ax3, diff, "Impact (Clean - Perturbed)", "RdBu_r", vmin=-limit, vmax=limit)
    
    plt.suptitle("LRP Validation: Impact of Masking on Great Storm Forecast", fontsize=20, y=0.95)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {args.output}")

if __name__ == "__main__":
    main()