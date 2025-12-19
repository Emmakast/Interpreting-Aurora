import matplotlib
matplotlib.use('Agg') 

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import argparse
import os
import time

# Try to import Cartopy
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
    try:
        import cartopy.config
        cartopy.config['data_dir'] = os.path.join(os.getcwd(), 'cartopy_data')
    except ImportError:
        os.environ.setdefault('CARTOPY_DATA_DIR', os.path.join(os.getcwd(), 'cartopy_data'))
except Exception as e:
    HAS_CARTOPY = False
    print("WARNING: Cartopy not found. Plotting without map projections.")

def plot_mega_grid(file_path, output_path=None, no_map=False, device="cpu", region=None):
    start_time = time.time()
    
    # 1. Load Data
    print(f"Loading {file_path} on {device}...")
    data = torch.load(file_path, map_location=device)
    
    # Force CPU for numpy conversion for coordinates
    lat = data.get("lat").cpu() 
    lon = data.get("lon").cpu()
    
    # 2. Setup Extent & Slicing
    region = region or (-15.0, 45.0, 30.0, 75.0)
    extent, lat_slice, lon_slice, lon_reorder = _compute_extent(lat, lon, region)
    
    use_map = HAS_CARTOPY and not no_map
    if use_map:
        proj = ccrs.PlateCarree()
        subplot_kw = {'projection': proj}
        data_extent = extent 
    else:
        subplot_kw = {}
        data_extent = [region[0], region[1], region[3], region[2]]

    # 3. Setup Grid Layout (4 Rows, 5 Columns)
    # Row 0: Surface LRP
    # Row 1: Surface Predictions  <-- MOVED UP
    # Row 2: Atmos LRP            <-- MOVED DOWN
    # Row 3: Atmos Predictions
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 5, figure=fig, height_ratios=[1, 1, 1, 1], wspace=0.1, hspace=0.3)
    
    # Helper for LRP plotting
    def plot_lrp_var(key, row_idx, col_idx, title, col_span=1):
        if key not in data: return
        ax = fig.add_subplot(gs[row_idx, col_idx : col_idx+col_span], **subplot_kw)
        
        tensor = data[key].to(device).float()
        
        # Aggregate: SUM for LRP
        if tensor.ndim > 2:
            flatten_dims = list(range(tensor.ndim - 2))
            heatmap = tensor.sum(dim=flatten_dims)
        else:
            heatmap = tensor
            
        heatmap = heatmap.cpu().numpy()
        
        # Apply strict coordinate logic
        if lon_reorder is not None: heatmap = heatmap[:, lon_reorder]
        if lat_slice is not None: heatmap = heatmap[lat_slice]
        if lon_slice is not None: heatmap = heatmap[:, lon_slice]
            
        # Limits (symmetric)
        abs_max = np.abs(heatmap).max()
        limit = np.percentile(np.abs(heatmap), 99.5)
        if limit == 0: limit = abs_max if abs_max > 0 else 1.0
        
        _plot_panel(ax, heatmap, data_extent, limit, use_map, cmap="seismic", is_prediction=False)
        ax.set_title(title, fontsize=16)

    # Helper for Prediction plotting
    def plot_pred_var(key, row_idx, col_idx, title, cmap="viridis", col_span=1, func=None):
        if key not in data: return
        ax = fig.add_subplot(gs[row_idx, col_idx : col_idx+col_span], **subplot_kw)
        
        t = data[key].to(device).float()
        # Aggregate: MEAN for prediction
        if t.ndim > 2:
            t = t.mean(dim=tuple(range(t.ndim - 2)))
        
        arr = t.cpu().numpy()
        
        # Apply strict coordinate logic
        if lon_reorder is not None: arr = arr[:, lon_reorder]
        if lat_slice is not None: arr = arr[lat_slice]
        if lon_slice is not None: arr = arr[:, lon_slice]
        
        # Apply custom transform function (e.g., Kelvin to Celsius)
        if func: arr = func(arr)
            
        _plot_panel(ax, arr, data_extent, None, use_map, cmap=cmap, is_prediction=True)
        ax.set_title(title, fontsize=16)

    # --- ROW 0: SURFACE LRP RELEVANCE ---
    print("Plotting Surface LRP...")
    plot_lrp_var("2t", 0, 0, "Surface T Relevance (2m)")
    plot_lrp_var("10u", 0, 1, "Surface U Relevance (10m)")
    plot_lrp_var("10v", 0, 2, "Surface V Relevance (10m)")
    plot_lrp_var("msl", 0, 4, "Surface Pressure Relevance (MSL)")

    # --- ROW 1: SURFACE PREDICTIONS ---
    print("Plotting Surface Predictions...")
    
    # 1. Temp 2m
    def to_celsius(x): return x - 273.15 if x.mean() > 200 else x
    plot_pred_var("pred_2t", 1, 0, "2m Temp (°C)", "coolwarm", func=to_celsius)
    
    # 2. Wind 10m (Combined)
    if "pred_10u" in data and "pred_10v" in data:
        ax = fig.add_subplot(gs[1, 1:3], **subplot_kw)
        u = data["pred_10u"].to(device).float().squeeze()
        v = data["pred_10v"].to(device).float().squeeze()
        if u.ndim > 2: u, v = u.mean(0), v.mean(0)
        wspd = torch.sqrt(u**2 + v**2).cpu().numpy()
        
        if lon_reorder is not None: wspd = wspd[:, lon_reorder]
        if lat_slice is not None: wspd = wspd[lat_slice]
        if lon_slice is not None: wspd = wspd[:, lon_slice]
            
        _plot_panel(ax, wspd, data_extent, None, use_map, cmap="viridis", is_prediction=True)
        ax.set_title("10m Wind Speed (m/s)")

    # 3. MSL
    def to_hpa(x): return x / 100
    plot_pred_var("pred_msl", 1, 4, "MSL Pressure (hPa)", "cividis", func=to_hpa)


    # --- ROW 2: ATMOS LRP RELEVANCE ---
    print("Plotting Atmos LRP...")
    plot_lrp_var("t", 2, 0, "Atmos T Relevance")
    plot_lrp_var("u", 2, 1, "Atmos U Relevance")
    plot_lrp_var("v", 2, 2, "Atmos V Relevance")
    plot_lrp_var("q", 2, 3, "Atmos Q Relevance")
    plot_lrp_var("z", 2, 4, "Atmos Z Relevance")


    # --- ROW 3: ATMOS PREDICTIONS ---
    print("Plotting Atmos Predictions...")
    
    # 1. Temp
    plot_pred_var("pred_t", 3, 0, "Atmos Temp Mean (°C)", "coolwarm", func=to_celsius)
    
    # 2. Wind Atmos (Combined)
    if "pred_u" in data and "pred_v" in data:
        ax = fig.add_subplot(gs[3, 1:3], **subplot_kw)
        u = data["pred_u"].to(device).float()
        v = data["pred_v"].to(device).float()
        if u.ndim > 2: u, v = u.mean(dim=(0,1)), v.mean(dim=(0,1))
        elif u.ndim > 2: u, v = u.mean(0), v.mean(0)
        
        wspd = torch.sqrt(u**2 + v**2).cpu().numpy()
        
        if lon_reorder is not None: wspd = wspd[:, lon_reorder]
        if lat_slice is not None: wspd = wspd[lat_slice]
        if lon_slice is not None: wspd = wspd[:, lon_slice]
            
        _plot_panel(ax, wspd, data_extent, None, use_map, cmap="viridis", is_prediction=True)
        ax.set_title("Atmos Wind Speed Mean (m/s)")
        
    # 3. Humidity
    def to_gkg(x): return x * 1000
    plot_pred_var("pred_q", 3, 3, "Atmos Humidity (g/kg)", "BuGn", func=to_gkg)
    
    # 4. Geopotential
    def to_m(x): return x / 9.80665
    plot_pred_var("pred_z", 3, 4, "Geopotential Height (m)", "cividis", func=to_m)

    plt.suptitle("LRP Relevance (Rows 1 & 3) vs Predictions (Rows 2 & 4)", fontsize=24, y=0.95)
    
    if output_path is None:
        output_path = file_path.replace(".pt", "_reordered.png")
        
    print(f"Saving to {output_path}...")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print("Done.")

# --- HELPER FUNCTIONS ---

def _compute_extent(lat_tensor, lon_tensor, region):
    if lat_tensor is None or lon_tensor is None: return None, None, None, None
    lat_vals = np.asarray(lat_tensor.squeeze().cpu())
    lon_vals = np.asarray(lon_tensor.squeeze().cpu())
    lon_wrapped = ((lon_vals + 180) % 360) - 180
    lon_order = np.argsort(lon_wrapped)
    lon_sorted = lon_wrapped[lon_order]
    lon_min, lon_max, lat_min, lat_max = region
    lat_idx = np.where((lat_vals >= lat_min) & (lat_vals <= lat_max))[0]
    lon_idx = np.where((lon_sorted >= lon_min) & (lon_sorted <= lon_max))[0]
    if len(lat_idx) == 0 or len(lon_idx) == 0: return None, None, None, lon_order
    lat_slice = slice(lat_idx[0], lat_idx[-1] + 1)
    lon_slice = slice(lon_idx[0], lon_idx[-1] + 1)
    extent = [float(lon_sorted[lon_slice.start]), float(lon_sorted[lon_slice.stop-1]), float(lat_vals[lat_slice.stop-1]), float(lat_vals[lat_slice.start])]
    return extent, lat_slice, lon_slice, lon_order

def _plot_panel(ax, heatmap, extent, limit, use_map, cmap="seismic", is_prediction=False):
    vmin, vmax = (None, None) if is_prediction else (-limit, limit)
    interp = 'bilinear' if is_prediction else 'nearest'
    if use_map:
        try:
            ax.coastlines(resolution='110m', color='black', linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        except Exception: pass
        im = ax.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', extent=extent, transform=ccrs.PlateCarree(), interpolation=interp)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        im = ax.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', extent=extent, aspect='auto', interpolation=interp)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
    return im

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to the .pt file")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-map", action="store_true")
    parser.add_argument("--lon-min", type=float, default=-15.0)
    parser.add_argument("--lon-max", type=float, default=45.0)
    parser.add_argument("--lat-min", type=float, default=30.0)
    parser.add_argument("--lat-max", type=float, default=75.0)
    args = parser.parse_args()
    region = (args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    plot_mega_grid(args.file, no_map=args.no_map, device=args.device, region=region)