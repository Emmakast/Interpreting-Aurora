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

def plot_levels_grid(file_path, var_name, output_path=None, no_map=False, device="cpu", region=None, levels=None):
    """
    Plot selected atmospheric levels in a 2x4 grid.
    Default levels: 0, 2, 4, 6, 8, 10, 12 (every other level) + surface in last slot
    """
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
    
    num_levels = atmos_tensor.shape[0]
    
    # Default levels: every other level (0, 2, 4, 6, 8, 10, 12)
    if levels is None:
        levels = list(range(0, min(13, num_levels), 2))
    
    # 2. Identify Corresponding Surface Tensor
    surf_map = {'t': '2t', 'u': '10u', 'v': '10v', 'z': 'msl', 'q': None}
    surf_key = surf_map.get(var_name)
    
    surf_tensor = None
    if surf_key and surf_key in data:
        surf_tensor = data[surf_key].to(device).float()
        if surf_tensor.ndim == 5: surf_tensor = surf_tensor[0, 0, 0] 
        elif surf_tensor.ndim == 4: surf_tensor = surf_tensor[0, 0]
        elif surf_tensor.ndim == 3: surf_tensor = surf_tensor[0]
        print(f"Found surface variable '{surf_key}' to compare.")
    else:
        print(f"No corresponding surface variable found for '{var_name}'.")

    # 3. Setup Grid
    lat = data.get("lat").cpu() 
    lon = data.get("lon").cpu()
    
    region = region or (-15.0, 45.0, 30.0, 75.0)
    extent, lat_slice, lon_slice, lon_reorder = _compute_extent(lat, lon, region)
    
    use_map = HAS_CARTOPY and not no_map
    subplot_kw = {'projection': ccrs.PlateCarree()} if use_map else {}
    data_extent = extent if use_map else [region[0], region[1], region[3], region[2]]

    # 2 Rows x 4 Cols = 8 slots (7 levels + 1 surface)
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.12, hspace=0.08)
    
    # Calculate Global Color Limits
    combined_data = atmos_tensor[levels].cpu().numpy()
    if surf_tensor is not None:
        combined_data = np.concatenate([combined_data, surf_tensor.cpu().numpy()[None, ...]], axis=0)
        
    abs_max = np.abs(combined_data).max()
    # Use 99th percentile for better color contrast (99.9 was too high for some dates)
    limit = np.percentile(np.abs(combined_data), 99)
    if limit == 0: limit = abs_max if abs_max > 0 else 1.0
    print(f"Color limit set to +/- {limit:.2e}")

    # Plot selected levels (7 levels fit in slots 0-6)
    for i, level_idx in enumerate(levels[:7]):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col], **subplot_kw)
        
        heatmap = atmos_tensor[level_idx].cpu().numpy()
        heatmap = _process_heatmap(heatmap, lon_reorder, lat_slice, lon_slice)
        
        title = f"Level {level_idx}"
        if level_idx == 0: title += " (Top)"
        if level_idx == num_levels - 1: title += " (Bottom)"
        
        # Show colorbar on all panels
        _plot_panel(ax, heatmap, data_extent, limit, use_map, title, show_colorbar=True)

    # Plot Surface in the 8th slot (row 1, col 3)
    if surf_tensor is not None:
        ax = fig.add_subplot(gs[1, 3], **subplot_kw)
        heatmap = surf_tensor.cpu().numpy()
        heatmap = _process_heatmap(heatmap, lon_reorder, lat_slice, lon_slice)
        _plot_panel(ax, heatmap, data_extent, limit, use_map, f"Surface ({surf_key})", show_colorbar=True)

    # Extract date from filename for title
    basename = os.path.basename(file_path)
    date_str = basename.replace("lrp_europe_", "").replace(".pt", "")
    
    
    if output_path is None:
        output_path = file_path.replace(".pt", f"_levels_{var_name}_grid.pdf")
        
    print(f"Saving to {output_path}...")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f}s")
    return output_path

# --- HELPER FUNCTIONS ---

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

def _plot_panel(ax, heatmap, extent, limit, use_map, title, show_colorbar=True):
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
    ax.set_title(title, fontsize=14)
    if show_colorbar:
        plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.02)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot selected atmospheric levels in 2x4 grid")
    parser.add_argument("file", type=str, help="Path to the .pt file")
    parser.add_argument("--var", type=str, default="u", help="Variable to plot (t, u, v, q, z)")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-map", action="store_true")
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 2, 4, 6, 8, 10, 12], 
                        help="Levels to plot (default: 0 2 4 6 8 10 12)")
    
    args = parser.parse_args()
    
    plot_levels_grid(args.file, args.var, output_path=args.output, no_map=args.no_map, 
                     device=args.device, levels=args.levels)
