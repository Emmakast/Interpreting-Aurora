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

# --- SETTINGS ---
VAR_MAP = {
    "2t": "2m_temperature", "10u": "10m_u_component_of_wind", "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure", "t": "temperature", "u": "u_component_of_wind",
    "v": "v_component_of_wind", "q": "specific_humidity", "z": "geopotential",
}

def load_batch(zarr_path, static_path, date_str):
    # Standard loading logic
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
    if latents.ndim == 4: latents, mask = latents.unsqueeze(1), mask.unsqueeze(1)
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
    parser.add_argument("--storm-dates", nargs="+", required=True)
    parser.add_argument("--calm-dates", nargs="+", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AuroraSmallPretrained()
    model.load_checkpoint()
    model.eval().to(device)

    # 1. Collect Pooled Vectors
    data_store = {"storm": [], "calm": []}
    
    print("Extracting vectors...")
    for label, dates in [("storm", args.storm_dates), ("calm", args.calm_dates)]:
        for d in dates:
            batch = load_batch(args.zarr_path, args.static_path, d)
            if batch is None: continue
            
            batch = batch.to(device)
            lat = extract_latents(model, batch)
            
            # Shape handling
            if lat.ndim == 3:
                T, N, C = lat.shape
                lat_size = batch.metadata.lat.shape[0]
                lat = lat.view(T, lat_size, N//lat_size, C).unsqueeze(0)
            elif lat.ndim == 4: lat = lat.permute(0, 2, 3, 1)
            elif lat.ndim == 5: lat = lat.permute(0, 1, 3, 4, 2)
            
            H, W = lat.shape[-3], lat.shape[-2]
            
            lat_in, lon_in = batch.metadata.lat.to(device), batch.metadata.lon.to(device)
            lat_sub = lat_in[np.linspace(0, lat_in.shape[0]-1, H).astype(int)]
            lon_sub = lon_in[np.linspace(0, lon_in.shape[0]-1, W).astype(int)]
            mask = get_us_mask(lat_sub, lon_sub).to(device)
            
            if not mask.any(): continue
            if lat.ndim == 4: mask = mask.unsqueeze(0)
            else: mask = mask.unsqueeze(0).unsqueeze(0).expand(lat.shape[:2] + mask.shape)
            
            pooled = spatial_pool(lat, mask)
            data_store[label].append(pooled)
            del batch, lat, pooled
            torch.cuda.empty_cache()

    if not data_store["storm"] or not data_store["calm"]:
        print("Error: Missing data.")
        return

    X_storm = torch.cat(data_store["storm"])
    X_calm = torch.cat(data_store["calm"])
    X_all = torch.cat([X_storm, X_calm], dim=0)

    # 2. Joint Normalization
    mean = X_all.mean(dim=0, keepdim=True)
    std = X_all.std(dim=0, keepdim=True) + 1e-6
    X_all_norm = (X_all - mean) / std

    # 3. Joint PCA (For Scatter Plots)
    U, S, Vh = torch.linalg.svd(X_all_norm, full_matrices=False)
    explained_var = (S ** 2) / (X_all.shape[0] - 1)
    explained_var_ratio = explained_var / explained_var.sum()
    
    # Project Scores
    scores = X_all_norm @ Vh.T
    scores_storm = scores[:X_storm.shape[0]]
    scores_calm = scores[X_storm.shape[0]:]

    # 4. Contrastive Projection (For Barcode Plot)
    # Direction = Mean(Storm) - Mean(Calm)
    X_storm_norm = (X_storm - mean) / std
    X_calm_norm = (X_calm - mean) / std
    
    contrastive_dir = X_storm_norm.mean(0) - X_calm_norm.mean(0)
    contrastive_dir = contrastive_dir / contrastive_dir.norm()
    
    proj_storm = X_storm_norm @ contrastive_dir
    proj_calm = X_calm_norm @ contrastive_dir

    # 5. Save Results
    os.makedirs("results", exist_ok=True)
    torch.save({
        "components": Vh.cpu(),
        "explained_variance_ratio": explained_var_ratio.cpu(),
        "scores_storm": scores_storm.cpu(),
        "scores_calm": scores_calm.cpu(),
        "proj_storm_contrastive": proj_storm.cpu(),
        "proj_calm_contrastive": proj_calm.cpu(),
        "feature_loadings": Vh.cpu(), 
        "raw_storm": X_storm.cpu(),
    "raw_calm": X_calm.cpu(),
    }, "results/storm_calm_pooled_results.pt")
    
    print("Analysis complete. Saved to results/storm_calm_pooled_results.pt")

if __name__ == "__main__":
    main()