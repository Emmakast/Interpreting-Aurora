import argparse
import xarray as xr
import pandas as pd
import numpy as np
import sys

# Lat Conversion: Index = (90 - degrees) * 4
# 50N -> (90-50)*4 = 160
# 24N -> (90-24)*4 = 264
LAT_SLICE = slice(160, 264) 

# Lon Conversion: Index = degrees * 4
# 235E -> 235*4 = 940
# 295E -> 295*4 = 1180
LON_SLICE = slice(940, 1180)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", type=str, required=True)
    parser.add_argument("--year", type=int, default=2005)
    parser.add_argument("--count", type=int, default=30)
    # Adjusted thresholds for Regional US data
    parser.add_argument("--storm-thresh", type=float, default=18.0) 
    parser.add_argument("--calm-thresh", type=float, default=12.0)
    args = parser.parse_args()

    print(f"--- Fast Scanning {args.year} (US REGION ONLY) ---", file=sys.stderr)
    print(f"    Region Indices: Lat {LAT_SLICE}, Lon {LON_SLICE}", file=sys.stderr)
    
    # 1. Define timestamps
    target_times = pd.date_range(
        start=f"{args.year}-01-01 12:00", 
        end=f"{args.year}-12-31 12:00", 
        freq="D"
    )

    try:
        ds = xr.open_zarr(args.zarr_path, consolidated=True)
        
        # 2. Select ONLY the US Region
        print(f"  Loading data for {len(target_times)} days...", file=sys.stderr)
        
        subset = ds.sel(time=target_times, method="nearest").isel(
            latitude=LAT_SLICE, 
            longitude=LON_SLICE
        )
        
        subset = subset.load()
        print(f"  Data loaded. Computing US-specific wind speeds...", file=sys.stderr)

        # 3. Calculate Wind Speed Magnitude
        u2 = np.square(subset["10m_u_component_of_wind"].values)
        v2 = np.square(subset["10m_v_component_of_wind"].values)
        ws = np.sqrt(u2 + v2) # Shape: (365, H, W) of the US Box
        
        # 4. Get Max Wind per Day inside the US Box
        daily_max_ws = np.max(ws, axis=(1, 2))
        
        measured_days = []
        for i, t in enumerate(target_times):
            date_obj = pd.to_datetime(t).date()
            speed = float(daily_max_ws[i])
            measured_days.append((date_obj, speed))
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # 5. Sort and Filter
    measured_days.sort(key=lambda x: x[1], reverse=True)

    storms = [x for x in measured_days if x[1] >= args.storm_thresh]
    calms = [x for x in measured_days if x[1] <= args.calm_thresh]
    calms.sort(key=lambda x: x[1]) # Weakest first

    selected_storms = storms[:args.count]
    selected_calms  = calms[:args.count]

    # Log results
    print(f"\n--- RESULTS (US Region) ---", file=sys.stderr)
    if selected_storms:
        print(f"  Top US Storm: {selected_storms[0][0]} ({selected_storms[0][1]:.2f} m/s)", file=sys.stderr)
        print(f"  Weakest selected storm: {selected_storms[-1][1]:.2f} m/s", file=sys.stderr)
    if selected_calms:
        print(f"  Calmest US Day: {selected_calms[0][0]} ({selected_calms[0][1]:.2f} m/s)", file=sys.stderr)
        print(f"  Windiest selected calm: {selected_calms[-1][1]:.2f} m/s", file=sys.stderr)

    # Output
    storm_str = " ".join(d[0].strftime("%Y-%m-%d") for d in sorted(selected_storms, key=lambda x: x[0]))
    calm_str = " ".join(d[0].strftime("%Y-%m-%d") for d in sorted(selected_calms, key=lambda x: x[0]))

    print(f"STORM_DATES=\"{storm_str}\"")
    print(f"CALM_DATES=\"{calm_str}\"")

if __name__ == "__main__":
    main()
