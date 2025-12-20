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

def plot_levels(file_path, var_name, output_path=None, no_map=False, device="cpu", region=None):
    start_time = time.time()
    
    print(f"Loading {file_path} on {device}...")
    data = torch.load(file_path, map_location=device)
    
    # 1. Identify Atmospheric Tensor
    if var_name not in data:
        print(f"Error: Variable '{var_name}' not found. Available: {list(data.keys())}")
        return
    
    atmos_tensor = data[var_name].to(device).float()
    
    # Handle dims (Batch, Time, Levels, H, W) -> (Levels, H, W)
    if atmos_tensor.ndim == 5: atmos_tensor = atmos_tensor[0, 0]
    
    num_levels = atmos_tensor.shape[0] # Should be 13
    
    # 2. Identify Corresponding Surface Tensor
    # Map 't' -> '2t', 'u' -> '10u', etc.
    surf_map = {'t': '2t', 'u': '10u', 'v': '10v', 'z': 'msl', 'q': None}
    surf_key = surf_map.get(var_name)
    
    surf_tensor = None
    if surf_key and surf_key in data:
        surf_tensor = data[surf_key].to(device).float()
        # Surface might be (Batch, Time, 1, H, W) or (Batch, Time, H, W) or (1, H, W)
        if surf_tensor.ndim == 5: surf_tensor = surf_tensor[0, 0, 0] 
        elif surf_tensor.ndim == 4: surf_tensor = surf_tensor[0, 0]
        elif surf_tensor.ndim == 3: surf_tensor = surf_tensor[0]
        print(f"Found surface variable '{surf_key}' to compare.")
    else:
        print(f"No corresponding surface variable found for '{var_name}'.")

    # 3. Setup Grid
    # Force CPU for coordinates
    lat = data.get("lat").cpu() 
    lon = data.get("lon").cpu()
    
    region = region or (-15.0, 45.0, 30.0, 75.0)
    extent, lat_slice, lon_slice, lon_reorder = _compute_extent(lat, lon, region)
    
    use_map = HAS_CARTOPY and not no_map
    subplot_kw = {'projection': ccrs.PlateCarree()} if use_map else {}
    data_extent = extent if use_map else [region[0], region[1], region[3], region[2]]

    # 3 Rows x 5 Cols = 15 slots. 
    # Slots 0-12: Atmos Levels. 
    # Slot 13: Empty
    # Slot 14 (Bottom Right): Surface.
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 5, figure=fig, wspace=0.1, hspace=0.2)
    
    # Calculate Global Color Limits (considering both Atmos and Surface)
    # This ensures "Red" means the same intensity at 10km high as it does on the ground.
    combined_data = atmos_tensor.cpu().numpy()
    if surf_tensor is not None:
        combined_data = np.concatenate([combined_data, surf_tensor.cpu().numpy()[None, ...]], axis=0)
        
    abs_max = np.abs(combined_data).max()
    limit = np.percentile(np.abs(combined_data), 99.9)
    if limit == 0: limit = abs_max if abs_max > 0 else 1.0
    print(f"Color limit set to +/- {limit:.2e}")

    # Plot 13 Atmos Levels
    for level_idx in range(num_levels):
        row = level_idx // 5
        col = level_idx % 5
        ax = fig.add_subplot(gs[row, col], **subplot_kw)
        
        heatmap = atmos_tensor[level_idx].cpu().numpy()
        heatmap = _process_heatmap(heatmap, lon_reorder, lat_slice, lon_slice)
        
        title = f"Lvl {level_idx}"
        if level_idx == 0: title += " (Top)"
        if level_idx == num_levels - 1: title += " (Bottom Atmos)"
        
        _plot_panel(ax, heatmap, data_extent, limit, use_map, title)

    # Plot Surface (if exists)
    if surf_tensor is not None:
        ax = fig.add_subplot(gs[2, 4], **subplot_kw)
        
        heatmap = surf_tensor.cpu().numpy()
        heatmap = _process_heatmap(heatmap, lon_reorder, lat_slice, lon_slice)
        
        _plot_panel(ax, heatmap, data_extent, limit, use_map, f"SURFACE ({surf_key})")
        

    plt.suptitle(f"Vertical Structure: {var_name.upper()} (Levels 0-12) + Surface", fontsize=16, y=0.95)
    
    if output_path is None:
        output_path = file_path.replace(".pt", f"_levels_surf_{var_name}.png")
        
    print(f"Saving to {output_path}...")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print("Done.")

def _process_heatmap(heatmap, lon_reorder, lat_slice, lon_slice):
    if lon_reorder is not None: heatmap = heatmap[:, lon_reorder]
    if lat_slice is not None: heatmap = heatmap[lat_slice]
    if lon_slice is not None: heatmap = heatmap[:, lon_slice]
    return heatmap

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

def _plot_panel(ax, heatmap, extent, limit, use_map, title):
    vmin, vmax = -limit, limit
    if use_map:
        try:
            ax.coastlines(resolution='110m', color='black', linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        except Exception: pass
        im = ax.imshow(heatmap, cmap="seismic", vmin=vmin, vmax=vmax, origin='upper', extent=extent, transform=ccrs.PlateCarree(), interpolation='nearest')
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        im = ax.imshow(heatmap, cmap="seismic", vmin=vmin, vmax=vmax, origin='upper', extent=extent, aspect='auto', interpolation='nearest')
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to the .pt file")
    parser.add_argument("--var", type=str, default="t", help="Variable to plot (t, u, v, q, z)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-map", action="store_true")
    
    args = parser.parse_args()
    
    plot_levels(args.file, args.var, no_map=args.no_map, device=args.device)
