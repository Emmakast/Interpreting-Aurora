import argparse
import torch
import xarray as xr
import pandas as pd
import numpy as np
from datetime import timedelta
from aurora import AuroraSmallPretrained, Batch, Metadata
import os

VAR_MAP = {
    "2t": "2m_temperature", "10u": "10m_u_component_of_wind", "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure", "t": "temperature", "u": "u_component_of_wind",
    "v": "v_component_of_wind", "q": "specific_humidity", "z": "geopotential",
}

def load_batch_from_zarr(zarr_path: str, static_path: str, date_str: str) -> Batch:
    ds = xr.open_zarr(zarr_path, consolidated=True)
    target_time = pd.to_datetime(f"{date_str}T06:00:00")
    request_times = [target_time - timedelta(hours=6), target_time]
    
    frame = ds.sel(time=request_times, method="nearest").load()
    frame = frame.sortby("time").isel(latitude=slice(0, 720))

    static = xr.open_dataset(static_path, engine="netcdf4")
    if "valid_time" in static.dims: static = static.isel(valid_time=0)
    static = static.interp(latitude=frame.latitude, longitude=frame.longitude)
    static = static.transpose("latitude", "longitude").isel(latitude=slice(0, 720))

    return Batch(
        surf_vars={
            "2t": torch.from_numpy(frame[VAR_MAP["2t"]].values).unsqueeze(0).float(),
            "10u": torch.from_numpy(frame[VAR_MAP["10u"]].values).unsqueeze(0).float(),
            "10v": torch.from_numpy(frame[VAR_MAP["10v"]].values).unsqueeze(0).float(),
            "msl": torch.from_numpy(frame[VAR_MAP["msl"]].values).unsqueeze(0).float(),
        },
        static_vars={
            "z": torch.from_numpy(static["z"].values[None, None]).float(),
            "slt": torch.from_numpy(static["slt"].values[None, None]).float(),
            "lsm": torch.from_numpy(static["lsm"].values[None, None]).float(),
        },
        atmos_vars={
            "t": torch.from_numpy(frame[VAR_MAP["t"]].values).unsqueeze(0).float(),
            "u": torch.from_numpy(frame[VAR_MAP["u"]].values).unsqueeze(0).float(),
            "v": torch.from_numpy(frame[VAR_MAP["v"]].values).unsqueeze(0).float(),
            "q": torch.from_numpy(frame[VAR_MAP["q"]].values).unsqueeze(0).float(),
            "z": torch.from_numpy(frame[VAR_MAP["z"]].values).unsqueeze(0).float(),
        },
        metadata=Metadata(
            lat=torch.from_numpy(frame.latitude.values),
            lon=torch.from_numpy(frame.longitude.values),
            time=tuple(pd.to_datetime(frame.time.values).to_pydatetime()),
            atmos_levels=tuple(int(lvl) for lvl in frame.level.values),
        ),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--static-path", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--lrp-file", required=True)
    parser.add_argument("--output", type=str, default="results/prediction_impact.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading Aurora...")
    model = AuroraSmallPretrained()
    model.load_checkpoint()
    model.eval().to(device)

    print(f"Loading data for {args.date}...")
    raw_batch = load_batch_from_zarr(args.zarr_path, args.static_path, args.date)
    
    # Handle missing stats
    surf_stats = model.surf_stats
    if not surf_stats:
        print("WARNING: model.surf_stats is empty. Using approximate stats.")
        surf_stats = {
            "2t": (288.0, 15.0), 
            "10u": (0.0, 6.0),   
            "10v": (0.0, 6.0),   
            "msl": (101325.0, 1500.0), 
        }

    # Preprocess exactly like in training/LRP
    batch = model.batch_transform_hook(raw_batch)
    batch = batch.normalise(surf_stats=surf_stats).crop(model.patch_size)
    
    print(f"Loading LRP map {args.lrp_file}...")
    lrp_data = torch.load(args.lrp_file, map_location="cpu")
    
    # Reconstruct Global Relevance
    surf_keys = ["2t", "10u", "10v", "msl"]
    atmos_keys = ["t", "u", "v", "q", "z"]
    first_key = next(k for k in surf_keys if k in lrp_data)
    total_relevance = torch.zeros_like(lrp_data[first_key][0,0])
    
    for k in surf_keys + atmos_keys:
        if k in lrp_data:
            rel = lrp_data[k].float().abs()
            dims_to_sum = list(range(rel.ndim - 2)) 
            total_relevance += rel.sum(dim=dims_to_sum)
            
    # --- SECTOR RESTRICTION (N. Atlantic) ---
    print("Applying Sector Mask...")
    # Lat 20N (280) to 80N (40)
    # Lon -60E (1200) to 45E (180)
    valid_lat_slice = slice(40, 280) 
    sector_mask = torch.zeros_like(total_relevance)
    sector_mask[valid_lat_slice, 0:180] = 1.0     
    sector_mask[valid_lat_slice, 1200:1440] = 1.0 
    total_relevance = total_relevance * sector_mask

    flat_rel = total_relevance.flatten()
    k_pixels = int(0.10 * flat_rel.numel())
    threshold = torch.topk(flat_rel, k_pixels).values[-1]
    
    mask_top_k = (total_relevance >= threshold).float().to(device)
    
    print(f"Masks created. Perturbing {k_pixels} pixels.")

    # Helper to clean/perturb batches
    def perturb_batch(b, mask_2d):
        p_surf = {}; p_atmos = {}
        # Zero out input pixels
        for k, v in b.surf_vars.items(): p_surf[k] = v * (1 - mask_2d[None, None, :, :])
        for k, v in b.atmos_vars.items(): p_atmos[k] = v * (1 - mask_2d[None, None, None, :, :])
        return Batch(p_surf, b.static_vars, p_atmos, b.metadata)

    # 1. Clean Prediction
    clean_batch = Batch(
        surf_vars={k: v.to(device) for k, v in batch.surf_vars.items()},
        static_vars={k: v.squeeze(0).squeeze(0).to(device) for k, v in batch.static_vars.items()},
        atmos_vars={k: v.to(device) for k, v in batch.atmos_vars.items()},
        metadata=batch.metadata
    )
    with torch.no_grad():
        pred_clean = model(clean_batch)
        msl_raw = pred_clean.surf_vars['msl']
        print(f"DEBUG: Raw MSL shape: {msl_raw.shape}")
        print(f"DEBUG: Raw MSL stats: {msl_raw.min():.4f}, {msl_raw.max():.4f}")
        
        # Skip denormalization for now as it produces huge values
        # We will handle scaling in visualization
        pred_clean.surf_vars['msl'] = msl_raw
    
    # 2. Perturbed Prediction (LRP)
    batch_rel = perturb_batch(clean_batch, mask_top_k)
    with torch.no_grad():
        pred_pert = model(batch_rel)
        # Skip denormalization
    
    # Extract MSL Pressure (Index -1 is the forecast step)
    msl_clean = pred_clean.surf_vars['msl'].squeeze().cpu()
    msl_pert = pred_pert.surf_vars['msl'].squeeze().cpu()
    
    # Save for visualization
    print(f"Saving predictions to {args.output}...")
    torch.save({
        "msl_clean": msl_clean,
        "msl_pert": msl_pert,
        "mask_lrp": mask_top_k.cpu(),
        "lats": batch.metadata.lat.cpu(),
        "lons": batch.metadata.lon.cpu()
    }, args.output)
    print("Done.")

if __name__ == "__main__":
    main()
