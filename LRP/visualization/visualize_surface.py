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

def plot_surface_grid(file_path, output_path=None, no_map=False, device="cpu", region=None):
    """
    Create a 2x4 grid plot:
    - Top row: Surface predictions (2t, 10u, 10v, msl)
    - Bottom row: Surface LRP relevance (2t, 10u, 10v, msl)
    """
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

    # 3. Setup Grid Layout (2 Rows, 4 Columns)
    # Row 0: Surface Predictions (2t, 10u, 10v, msl)
    # Row 1: Surface LRP (2t, 10u, 10v, msl)
    fig = plt.figure(figsize=(20, 7))
    gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1, 1], wspace=0.15, hspace=0.08)
    
    # Helper for LRP plotting
    def plot_lrp_var(key, row_idx, col_idx, title):
        if key not in data: 
            print(f"  Warning: Key '{key}' not found in data")
            return
        ax = fig.add_subplot(gs[row_idx, col_idx], **subplot_kw)
        
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
        ax.set_title(title, fontsize=14)

    # Helper for Prediction plotting
    def plot_pred_var(key, row_idx, col_idx, title, cmap="viridis", func=None):
        if key not in data: 
            print(f"  Warning: Key '{key}' not found in data")
            return
        ax = fig.add_subplot(gs[row_idx, col_idx], **subplot_kw)
        
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
        ax.set_title(title, fontsize=14)

    # Transform functions
    def to_celsius(x): return x - 273.15 if x.mean() > 200 else x
    def to_hpa(x): return x / 100

    # --- ROW 0: SURFACE PREDICTIONS ---
    print("Plotting Surface Predictions...")
    plot_pred_var("pred_2t", 0, 0, "2m Temperature (°C)", "coolwarm", func=to_celsius)
    plot_pred_var("pred_10u", 0, 1, "10m U Wind (m/s)", "RdBu_r")
    plot_pred_var("pred_10v", 0, 2, "10m V Wind (m/s)", "RdBu_r")
    plot_pred_var("pred_msl", 0, 3, "MSL Pressure (hPa)", "cividis", func=to_hpa)

    # --- ROW 1: SURFACE LRP RELEVANCE ---
    print("Plotting Surface LRP...")
    plot_lrp_var("2t", 1, 0, "2m Temp LRP Relevance")
    plot_lrp_var("10u", 1, 1, "10m U Wind LRP Relevance")
    plot_lrp_var("10v", 1, 2, "10m V Wind LRP Relevance")
    plot_lrp_var("msl", 1, 3, "MSL Pressure LRP Relevance")

    # Extract date from filename for title
    basename = os.path.basename(file_path)
    date_str = basename.replace("lrp_europe_", "").replace(".pt", "")
    
    if output_path is None:
        output_path = file_path.replace(".pt", "_surface_grid.pdf")
        
    print(f"Saving to {output_path}...")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f}s")
    return output_path

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
    plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.02)
    return im

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 2x4 grid: Surface Predictions (top) vs Surface LRP (bottom)")
    parser.add_argument("file", type=str, help="Path to the .pt file")
    parser.add_argument("--output", type=str, default=None, help="Output path for the plot")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-map", action="store_true")
    parser.add_argument("--lon-min", type=float, default=-15.0)
    parser.add_argument("--lon-max", type=float, default=45.0)
    parser.add_argument("--lat-min", type=float, default=30.0)
    parser.add_argument("--lat-max", type=float, default=75.0)
    args = parser.parse_args()
    region = (args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    plot_surface_grid(args.file, output_path=args.output, no_map=args.no_map, device=args.device, region=region)
