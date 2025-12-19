import argparse
import torch
import xarray as xr
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from aurora import AuroraSmallPretrained, Batch, Metadata

# Reuse your existing settings
VAR_MAP = {
    "2t": "2m_temperature", "10u": "10m_u_component_of_wind", "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure", "t": "temperature", "u": "u_component_of_wind",
    "v": "v_component_of_wind", "q": "specific_humidity", "z": "geopotential",
}

def load_batch(zarr_path: str, static_path: str, date_str: str) -> Batch:
    # Standard loading logic (same as before)
    ds = xr.open_zarr(zarr_path, consolidated=True)
    target_time = pd.to_datetime(f"{date_str}T06:00:00")
    request_times = [target_time - timedelta(hours=6), target_time]

    try:
        frame = ds.sel(time=request_times, method="nearest").load()
        frame = frame.sortby("time")
        frame = frame.isel(latitude=slice(0, 720), longitude=slice(0, 1440))
        if frame.time.size < 2: return None
    except Exception: return None

    static = xr.open_dataset(static_path, engine="netcdf4")
    if "valid_time" in static.dims: static = static.isel(valid_time=0)
    static = static.interp(latitude=frame.latitude, longitude=frame.longitude)
    static = static.transpose("latitude", "longitude")

    surf = {k: torch.from_numpy(frame[VAR_MAP[k]].transpose("time", "latitude", "longitude").values).unsqueeze(0).float() for k in ["2t", "10u", "10v", "msl"]}
    atmos = {k: torch.from_numpy(frame[VAR_MAP[k]].transpose("time", "level", "latitude", "longitude").values).unsqueeze(0).float() for k in ["t", "u", "v", "q", "z"]}

    return Batch(surf, {"z": torch.from_numpy(static["z"].values).float(), "slt": torch.from_numpy(static["slt"].values).float(), "lsm": torch.from_numpy(static["lsm"].values).float()}, atmos, Metadata(torch.from_numpy(frame.latitude.values), torch.from_numpy(frame.longitude.values), tuple(pd.to_datetime(frame.time.values).to_pydatetime()), tuple(int(l) for l in frame.level.values)))

def extract_latents(model, batch):
    activ = {}
    def hook(_, __, out): activ["x"] = out[0] if isinstance(out, tuple) else out
    h = model.backbone.register_forward_hook(hook)
    with torch.no_grad(): model(batch)
    h.remove()
    return activ["x"]

def get_us_mask(lat, lon):
    lon_grid, lat_grid = torch.meshgrid(lon, lat, indexing="xy")
    lon_grid = lon_grid % 360
    return ((lat_grid >= 24) & (lat_grid <= 50) & (lon_grid >= 235) & (lon_grid <= 295))

def spatial_pool(latents, mask):
    if latents.ndim == 4:
        latents, mask = latents.unsqueeze(1), mask.unsqueeze(1)
    B, T, H, W, C = latents.shape
    latents, mask = latents.reshape(B * T, H, W, C), mask.reshape(B * T, H, W)
    feats = []
    for x, m in zip(latents, mask):
        x = x[m]
        feats.append(torch.cat([x.mean(0), x.std(0), torch.sqrt((x ** 2).mean(0))]))
    return torch.stack(feats)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--static-path", required=True)
    parser.add_argument("--winter", nargs='+', help="List of Winter dates")
    parser.add_argument("--spring", nargs='+', help="List of Spring dates")
    parser.add_argument("--summer", nargs='+', help="List of Summer dates")
    parser.add_argument("--fall", nargs='+', help="List of Fall dates")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AuroraSmallPretrained()
    model.load_checkpoint()
    model.eval().to(device)

    seasons = {
        "Winter": args.winter,
        "Spring": args.spring,
        "Summer": args.summer,
        "Fall": args.fall
    }
    
    # 1. Collect Pooled Vectors
    data_store = {}
    colors = {"Winter": "blue", "Spring": "green", "Summer": "red", "Fall": "orange"}

    print("Extracting vectors for all seasons...")
    for name, dates in seasons.items():
        if not dates: continue
        print(f"  Processing {name} ({len(dates)} dates)...")
        season_vectors = []
        
        for d in dates:
            batch = load_batch(args.zarr_path, args.static_path, d)
            if batch is None: continue
            
            batch = batch.to(device)
            lat = extract_latents(model, batch)
            
            # Shape Handling
            if lat.ndim == 3:
                T, N, C = lat.shape
                lat_size = batch.metadata.lat.shape[0]
                lat = lat.view(T, lat_size, N//lat_size, C).unsqueeze(0)
            elif lat.ndim == 4: lat = lat.permute(0, 2, 3, 1)
            elif lat.ndim == 5: lat = lat.permute(0, 1, 3, 4, 2)
            
            H, W = lat.shape[-3], lat.shape[-2]
            
            # Masking
            lat_in, lon_in = batch.metadata.lat.to(device), batch.metadata.lon.to(device)
            lat_sub = lat_in[np.linspace(0, lat_in.shape[0]-1, H).astype(int)]
            lon_sub = lon_in[np.linspace(0, lon_in.shape[0]-1, W).astype(int)]
            mask = get_us_mask(lat_sub, lon_sub).to(device)
            
            if not mask.any(): continue
            
            if lat.ndim == 4: mask = mask.unsqueeze(0)
            else: mask = mask.unsqueeze(0).unsqueeze(0).expand(lat.shape[:2] + mask.shape)
            
            pooled = spatial_pool(lat, mask)
            season_vectors.append(pooled)
            del batch, lat, pooled
            torch.cuda.empty_cache()
            
        if season_vectors:
            data_store[name] = torch.cat(season_vectors)

    if not data_store:
        print("Error: No data found.")
        return

    # 2. Joint PCA
    print("Running Joint PCA...")
    # Combine all seasons into one matrix
    X_all = torch.cat(list(data_store.values()), dim=0)
    
    # Normalize (CRITICAL STEP)
    mean = X_all.mean(dim=0, keepdim=True)
    std = X_all.std(dim=0, keepdim=True) + 1e-6
    X_all_norm = (X_all - mean) / std
    
    # PCA
    U, S, Vh = torch.linalg.svd(X_all_norm, full_matrices=False)
    
    # Calculate Explained Variance
    explained_var = (S ** 2) / (X_all.shape[0] - 1)
    explained_var_ratio = explained_var / explained_var.sum()

    # 3. Project, Plot, and Collect Data for Saving
    plt.figure(figsize=(10, 8))
    
    saved_scores = {} # Dictionary to store scores for the .pt file

    # We need to project each season separately to color them
    start_idx = 0
    for name, X in data_store.items():
        n_samples = X.shape[0]
        
        # Normalize using GLOBAL mean/std (to keep them in the same space)
        X_norm = (X - mean) / std
        
        # Project using GLOBAL Vh
        scores_tensor = X_norm @ Vh.T
        
        # Store for saving (keep as CPU tensor)
        saved_scores[name] = scores_tensor.cpu()
        
        # Convert to numpy for plotting
        scores = scores_tensor.cpu().numpy()
        
        # Plot
        plt.scatter(scores[:, 0], scores[:, 1], c=colors[name], label=name, alpha=0.6, edgecolors='none')
        
        # Centroid
        center = scores.mean(axis=0)
        plt.scatter(center[0], center[1], c='black', marker='X', s=150, linewidth=1.5, edgecolors='white')
        plt.text(center[0], center[1], name, fontsize=12, fontweight='bold')
        
        start_idx += n_samples

    plt.xlabel(f"PC1 ({explained_var_ratio[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({explained_var_ratio[1]*100:.1f}%)")
    plt.title("Seasonal PCA (Pooled Latents) - US Region")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/seasonal_pooled_pca.png")
    print("Saved plot to results/seasonal_pooled_pca.png")
    
    # 4. Save .pt file
    save_path = "results/seasonal_pooled_pca.pt"
    print(f"Saving data to {save_path}...")
    torch.save({
        "components": Vh.cpu(),
        "explained_variance": explained_var.cpu(),
        "explained_variance_ratio": explained_var_ratio.cpu(),
        "scores": saved_scores,
        "global_mean": mean.cpu(),
        "global_std": std.cpu(),
        "raw_data": {k: v.cpu() for k, v in data_store.items()}
    }, save_path)
    print("Done.")

if __name__ == "__main__":
    main()