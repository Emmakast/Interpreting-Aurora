import argparse
import torch
import xarray as xr
import pandas as pd
import numpy as np
import random
from datetime import timedelta
from aurora import AuroraSmallPretrained, Batch, Metadata
import os

VAR_MAP = {
    "2t": "2m_temperature", "10u": "10m_u_component_of_wind", "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure", "t": "temperature", "u": "u_component_of_wind",
    "v": "v_component_of_wind", "q": "specific_humidity", "z": "geopotential",
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")

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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-random", type=int, default=10, help="Number of random runs per pct")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading Aurora...")
    model = AuroraSmallPretrained()
    model.load_checkpoint()
    model.eval().to(device)

    print(f"Loading data for {args.date}...")
    raw_batch = load_batch_from_zarr(args.zarr_path, args.static_path, args.date)
    
    surf_stats = model.surf_stats
    if not surf_stats:
        surf_stats = {"2t": (288.0, 15.0), "10u": (0.0, 6.0), "10v": (0.0, 6.0), "msl": (101325.0, 1500.0)}

    batch = model.batch_transform_hook(raw_batch)
    batch = batch.normalise(surf_stats=surf_stats).crop(model.patch_size)
    
    print(f"Loading LRP map {args.lrp_file}...")
    lrp_data = torch.load(args.lrp_file, map_location="cpu")
    
    # 1. Global Relevance
    surf_keys = ["2t", "10u", "10v", "msl"]
    atmos_keys = ["t", "u", "v", "q", "z"]
    first_key = next(k for k in surf_keys if k in lrp_data)
    total_relevance = torch.zeros_like(lrp_data[first_key][0,0])
    
    for k in surf_keys + atmos_keys:
        if k in lrp_data:
            rel = lrp_data[k].float().abs()
            dims_to_sum = list(range(rel.ndim - 2)) 
            total_relevance += rel.sum(dim=dims_to_sum)
            
    # 2. Sector Masks
    valid_lat_slice = slice(40, 280) 
    sector_mask_binary = torch.zeros_like(total_relevance)
    sector_mask_binary[valid_lat_slice, 0:180] = 1.0     
    sector_mask_binary[valid_lat_slice, 1200:1440] = 1.0 
    eval_mask = sector_mask_binary.to(device)
    
    masked_relevance = total_relevance * sector_mask_binary
    flat_rel = masked_relevance.flatten()
    valid_indices = torch.nonzero(sector_mask_binary.flatten(), as_tuple=True)[0]
    
    # 3. Output Mask (Europe)
    lats = batch.metadata.lat
    lons = batch.metadata.lon
    lat_grid = lats.unsqueeze(1).repeat(1, len(lons))
    lon_grid = lons.unsqueeze(0).repeat(len(lats), 1)
    europe_mask = (
        (lat_grid <= 72) & (lat_grid >= 30) & 
        ((lon_grid <= 45) | (lon_grid >= 348))
    ).float().to(device)

    # 4. Loop
    percentages = [0.01]
    
    print(f"\n--- Running Perturbation Analysis (Random N={args.n_random}) ---")
    print(f"{'Pct':<6} | {'LRP MSE':<12} | {'Rand Mean':<12} | {'Rand Std':<10} | {'Ratio':<6}")
    print("-" * 65)

    def perturb_batch(b, mask_2d):
        p_surf = {}; p_atmos = {}
        for k, v in b.surf_vars.items(): p_surf[k] = v * (1 - mask_2d[None, None, :, :])
        for k, v in b.atmos_vars.items(): p_atmos[k] = v * (1 - mask_2d[None, None, None, :, :])
        return Batch(p_surf, b.static_vars, p_atmos, b.metadata)

    def get_mse(pred_clean, pred_pert):
        diff = (pred_clean.surf_vars['msl'] - pred_pert.surf_vars['msl'])
        # Use europe_mask to evaluate impact on the target region (Europe)
        diff = diff * europe_mask[None, None, :, :]
        mse = (diff ** 2).sum() / europe_mask.sum()
        return mse.item()

    # Clean Pass
    clean_batch = Batch(
        surf_vars={k: v.to(device) for k, v in batch.surf_vars.items()},
        static_vars={k: v.squeeze(0).squeeze(0).to(device) for k, v in batch.static_vars.items()},
        atmos_vars={k: v.to(device) for k, v in batch.atmos_vars.items()},
        metadata=batch.metadata
    )
    with torch.no_grad():
        pred_clean = model(clean_batch)

    results_to_save = {}


    # Calculate total pixels IN THE SECTOR
    sector_pixel_count = valid_indices.numel()
    print(f"Total pixels in N. Atlantic Sector: {sector_pixel_count}")

    for pct in percentages:
        # Mask percentage relative to the SECTOR, not the globe
        k_count = int(pct * sector_pixel_count)
        
        print(f"Masking {k_count} pixels ({pct*100}%) ...")
        
        # A. LRP Pass (Deterministic)
        threshold = torch.topk(flat_rel, k_count).values[-1]
        mask_lrp = (masked_relevance >= threshold).float().to(device)
        with torch.no_grad():
            pred_lrp = model(perturb_batch(clean_batch, mask_lrp))
        mse_lrp = get_mse(pred_clean, pred_lrp)
        
        # B. Random Loop
        rand_mses = []
        for i in range(args.n_random):
            perm = torch.randperm(valid_indices.size(0))
            idx_rand = valid_indices[perm[:k_count]]
            mask_rand = torch.zeros_like(flat_rel)
            mask_rand[idx_rand] = 1.0
            mask_rand = mask_rand.reshape(total_relevance.shape).to(device)
            
            with torch.no_grad():
                pred_rnd = model(perturb_batch(clean_batch, mask_rand))
            rand_mses.append(get_mse(pred_clean, pred_rnd))
            
        avg_rand = np.mean(rand_mses)
        std_rand = np.std(rand_mses)
        ratio = mse_lrp / (avg_rand + 1e-8)
        
        print(f"{pct*100:>4.0f}%  | {mse_lrp:.4e}   | {avg_rand:.4e}   | {std_rand:.4e} | {ratio:.2f}x")
        
        # Save 10% data for plotting
        if pct == 0.10:
            results_to_save = {
                "msl_clean": pred_clean.surf_vars['msl'].squeeze().cpu(),
                "msl_pert": pred_lrp.surf_vars['msl'].squeeze().cpu(),
                "mask_lrp": mask_lrp.cpu(),
                "lats": lats.cpu(),
                "lons": lons.cpu(),
                "ratio": ratio,
                "mse_lrp": mse_lrp,
                "mse_rnd_mean": avg_rand
            }

    print("-" * 65)
    print(f"Saving 10% visualization data to {args.output}...")
    torch.save(results_to_save, args.output)
    print("Done.")

if __name__ == "__main__":
    main()
