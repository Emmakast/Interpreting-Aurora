# # import argparse
# # from datetime import date, timedelta

# # def get_all_days_in_year(year):
# #     """Returns a set of all date objects in the given year."""
# #     start_date = date(year, 1, 1)
# #     end_date = date(year, 12, 31)
# #     delta = end_date - start_date
# #     return {start_date + timedelta(days=i) for i in range(delta.days + 1)}

# # def get_2005_storm_dates():
# #     """
# #     Returns the specific Peak Intensity or Landfall dates for the 
# #     record-breaking 28 storms of the 2005 Atlantic Season.
# #     Source: NHC Best Track Data.
# #     """
# #     storm_dates = set()

# #     --- THE 28 NAMED/RECOGNIZED STORMS OF 2005 ---
    
# #     1. EARLY SEASON (June/July)
# #     storm_dates.add(date(2005, 6, 11))  # TS Arlene (Landfall FL/AL)
# #     storm_dates.add(date(2005, 6, 28))  # TS Bret (Mexico landfall)
# #     storm_dates.add(date(2005, 7, 5))   # Hurricane Cindy (Landfall LA)
# #     storm_dates.add(date(2005, 7, 10))  # Hurricane Dennis (Major Landfall FL)
# #     storm_dates.add(date(2005, 7, 16))  # Hurricane Emily (Cat 5 Peak Caribbean)
# #     storm_dates.add(date(2005, 7, 24))  # TS Gert (Mexico Landfall)
# #     storm_dates.add(date(2005, 7, 27))  # TS Franklin (Atlantic - Peak)

# #     2. PEAK SEASON (August/September)
# #     storm_dates.add(date(2005, 8, 4))   # TS Harvey (Atlantic - Peak)
# #     storm_dates.add(date(2005, 8, 16))  # Hurricane Irene (Atlantic - Peak)
# #     storm_dates.add(date(2005, 8, 22))  # TS Jose (Mexico Landfall)
# #     storm_dates.add(date(2005, 8, 25))  # Hurricane Katrina (Florida Landfall)
# #     storm_dates.add(date(2005, 8, 29))  # Hurricane Katrina (Gulf Landfall - THE BIG ONE)
# #     storm_dates.add(date(2005, 8, 31))  # TS Lee (Atlantic - Peak)
# #     storm_dates.add(date(2005, 9, 5))   # Hurricane Maria (Atlantic - Peak)
# #     storm_dates.add(date(2005, 9, 8))   # Hurricane Nate (Atlantic - Peak)
# #     storm_dates.add(date(2005, 9, 14))  # Hurricane Ophelia (NC Coast Impact)
# #     storm_dates.add(date(2005, 9, 19))  # Hurricane Philippe (Atlantic - Peak)
# #     storm_dates.add(date(2005, 9, 24))  # Hurricane Rita (Landfall TX/LA)

# #     3. LATE SEASON (October)
# #     storm_dates.add(date(2005, 10, 4))  # Hurricane Stan (Mexico Impact)
# #     storm_dates.add(date(2005, 10, 4))  # Unnamed Subtropical Storm (Atlantic)
# #     storm_dates.add(date(2005, 10, 5))  # TS Tammy (Florida Impact)
# #     storm_dates.add(date(2005, 10, 19)) # Hurricane Wilma (Peak Intensity 882mb)
# #     storm_dates.add(date(2005, 10, 24)) # Hurricane Wilma (Florida Landfall)
    
# #     4. THE GREEK ALPHABET (Late Oct - Jan)
# #     NOTE: Many of these did not hit the US. If you crop to US, these may look "Calm".
# #     storm_dates.add(date(2005, 10, 9))  # Hurricane Vince (Spain Landfall!)
# #     storm_dates.add(date(2005, 10, 23)) # TS Alpha (Dominican Republic)
# #     storm_dates.add(date(2005, 10, 29)) # Hurricane Beta (Nicaragua)
# #     storm_dates.add(date(2005, 11, 18)) # TS Gamma (Honduras)
# #     storm_dates.add(date(2005, 11, 23)) # TS Delta (Canary Islands!)
# #     storm_dates.add(date(2005, 12, 2))  # Hurricane Epsilon (Atlantic)
# #     storm_dates.add(date(2005, 12, 30)) # TS Zeta (Atlantic)

# #     return storm_dates

# # def pick_evenly(dates, k):
# #     """Selects k dates evenly distributed from a list of dates."""
# #     if k <= 0: return []
# #     dates = sorted(dates)
# #     if len(dates) <= k: return dates
# #     step = (len(dates) - 1) / (k - 1)
# #     return [dates[int(round(i * step))] for i in range(k)]

# # def main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--storm-csv", type=str, help="Not used in this mode, but kept for compatibility")
# #     parser.add_argument("--year", type=int, default=2005, help="Year to analyze (Must be 2005 for this script)")
# #     parser.add_argument("--max-calm", type=int, default=None, help="Maximum number of calm days to output")
# #     parser.add_argument("--even-calm", action="store_true", help="If set, picks calm days evenly across the year")
# #     args = parser.parse_args()

# #     if args.year != 2005:
# #         print("ERROR: This script is hardcoded for the 2005 records. Please set --year 2005.")
# #         return

# #     1. Get all days
# #     all_days = get_all_days_in_year(2005)
    
# #     2. Get the official list of 28 Storms
# #     storm_days = get_2005_storm_dates()
    
# #     3. Calm days = All Days MINUS Storm Days
# #     calm_days = sorted(all_days - storm_days)

# #     4. Cap/Distribute Calm Days
# #     target_calm = args.max_calm if args.max_calm is not None else len(storm_days)
    
# #     if target_calm > 0 and len(calm_days) > target_calm:
# #         calm_days = pick_evenly(calm_days, target_calm if args.even_calm else target_calm)

# #     5. Output
# #     storm_dates_str = " ".join(d.strftime("%Y-%m-%d") for d in sorted(storm_days))
# #     calm_dates_str  = " ".join(d.strftime("%Y-%m-%d") for d in calm_days)

# #     print(f"STORM_DATES=\"{storm_dates_str}\"")
# #     print(f"CALM_DATES=\"{calm_dates_str}\"")

# # if __name__ == "__main__":
# #     main()
# import argparse
# import xarray as xr
# import pandas as pd
# import numpy as np
# import sys
# import os
# from datetime import date, timedelta

# # --- 1. DATE DEFINITIONS ---

# def get_2005_storm_dates():
#     """
#     Returns the specific dates for the record-breaking 2005 Atlantic Season.
#     """
#     storm_dates = set()
#     # 1. EARLY SEASON (June/July)
#     storm_dates.add(date(2005, 6, 11))  # TS Arlene (Landfall FL/AL)
#     storm_dates.add(date(2005, 6, 28))  # TS Bret (Mexico landfall)
#     storm_dates.add(date(2005, 7, 5))   # Hurricane Cindy (Landfall LA)
#     storm_dates.add(date(2005, 7, 10))  # Hurricane Dennis (Major Landfall FL)
#     storm_dates.add(date(2005, 7, 16))  # Hurricane Emily (Cat 5 Peak Caribbean)
#     storm_dates.add(date(2005, 7, 24))  # TS Gert (Mexico Landfall)
#     storm_dates.add(date(2005, 7, 27))  # TS Franklin (Atlantic - Peak)

#     # 2. PEAK SEASON (August/September)
#     storm_dates.add(date(2005, 8, 4))   # TS Harvey (Atlantic - Peak)
#     storm_dates.add(date(2005, 8, 16))  # Hurricane Irene (Atlantic - Peak)
#     storm_dates.add(date(2005, 8, 22))  # TS Jose (Mexico Landfall)
#     storm_dates.add(date(2005, 8, 25))  # Hurricane Katrina (Florida Landfall)
#     storm_dates.add(date(2005, 8, 29))  # Hurricane Katrina (Gulf Landfall - THE BIG ONE)
#     storm_dates.add(date(2005, 8, 31))  # TS Lee (Atlantic - Peak)
#     storm_dates.add(date(2005, 9, 5))   # Hurricane Maria (Atlantic - Peak)
#     storm_dates.add(date(2005, 9, 8))   # Hurricane Nate (Atlantic - Peak)
#     storm_dates.add(date(2005, 9, 14))  # Hurricane Ophelia (NC Coast Impact)
#     storm_dates.add(date(2005, 9, 19))  # Hurricane Philippe (Atlantic - Peak)
#     storm_dates.add(date(2005, 9, 24))  # Hurricane Rita (Landfall TX/LA)

#             # 3. LATE SEASON (October)
#     storm_dates.add(date(2005, 10, 4))  # Hurricane Stan (Mexico Impact)
#     storm_dates.add(date(2005, 10, 4))  # Unnamed Subtropical Storm (Atlantic)
#     storm_dates.add(date(2005, 10, 5))  # TS Tammy (Florida Impact)
#     storm_dates.add(date(2005, 10, 19)) # Hurricane Wilma (Peak Intensity 882mb)
#     storm_dates.add(date(2005, 10, 24)) # Hurricane Wilma (Florida Landfall)
#     return storm_dates

# def get_all_days_in_year(year):
#     start = date(year, 1, 1)
#     return {start + timedelta(days=i) for i in range(365)}

# def pick_evenly(dates, k):
#     """Selects k dates evenly distributed from a list."""
#     if k <= 0 or not dates: return []
#     dates = sorted(dates)
#     if len(dates) <= k: return dates
#     step = (len(dates) - 1) / (k - 1)
#     return [dates[int(round(i * step))] for i in range(k)]

# # --- 2. VALIDATION LOGIC ---

# def is_seasonally_valid(d, label):
#     if label == 'storm': return True
#     return 6 <= d.month <= 10

# def verify_wind_intensity(zarr_path, date_obj, label):
#     try:
#         date_str = date_obj.strftime("%Y-%m-%d")
#         target_t2 = pd.to_datetime(f"{date_str}T12:00:00")
        
#         ds = xr.open_zarr(zarr_path, consolidated=True)
#         try:
#             # Load specific time
#             frame = ds.sel(time=target_t2, method="nearest").load()
#         except KeyError:
#             return False

#         # --- CRITICAL FIX: FOCUS ON NORTH ATLANTIC ONLY ---
#         # Assuming Data is (Lat: 90 to -90, Lon: 0 to 360)
#         # Lat index 0-300 covers approx 90N to 15N (Northern Hemisphere)
#         # Lon index 1000-1440 covers approx 250E to 360E (Atlantic/Americas)
#         # You may need to adjust indices based on your exact grid, 
#         # but restricting to the Northern Hemisphere is the most important step.
        
#         frame_region = frame.isel(latitude=slice(0, 360)) # Only Northern Hemisphere
        
#         u = frame_region["10m_u_component_of_wind"].values
#         v = frame_region["10m_v_component_of_wind"].values
#         max_ws = np.max(np.sqrt(u**2 + v**2))

#         # --- LOGGING ---
#         print(f"  [Check] {date_str} ({label}): Max Wind (N. Hem) = {max_ws:.2f} m/s", file=sys.stderr)

#         if label == 'storm':
#             # 18.0 m/s is strong enough to be significant in grid data
#             # Do NOT require > 40.0 (that is extremely rare in 0.25deg data)
#             if max_ws < 20.0:
#                 print(f"    -> REJECTED (Storm too weak)", file=sys.stderr)
#                 return False
                
#         elif label == 'calm':
#             # Now that we ignore the Southern Ocean, we can expect lower winds.
#             # 18.0 m/s is a reasonable cutoff for "No Hurricane present"
#             if max_ws > 15.0: 
#                 print(f"    -> REJECTED (Too windy for calm)", file=sys.stderr)
#                 return False
            
#         return True

#     except Exception as e:
#         print(f"  [Error] {date_str}: {e}", file=sys.stderr)
#         return False
# # --- 3. MAIN SCRIPT ---

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--zarr-path", type=str, required=True)
#     parser.add_argument("--year", type=int, default=2005)
#     parser.add_argument("--max-calm", type=int, default=30)
#     args = parser.parse_args()

#     # LOGS GO TO STDERR
#     print(f"--- Preparing Dates for {args.year} ---", file=sys.stderr)

#     all_days = get_all_days_in_year(args.year)
#     storm_dates = get_2005_storm_dates()
#     candidate_calm = sorted(list(all_days - storm_dates))

#     print(f"Verifying storms...", file=sys.stderr)
#     final_storms = []
#     for d in sorted(storm_dates):
#         if verify_wind_intensity(args.zarr_path, d, 'storm'):
#             final_storms.append(d)
    
#     # ... inside main() ...
    
#     print(f"Verifying calm...", file=sys.stderr)
#     seasonal_calm = [d for d in candidate_calm if is_seasonally_valid(d, 'calm')]
    
#     # INCREASE CANDIDATE POOL: Check 5x the needed amount
#     subset_calm = pick_evenly(seasonal_calm, args.max_calm * 2) 
    
#     final_calm = []
#     for d in subset_calm:
#         if len(final_calm) >= args.max_calm: break
        
#         # Print progress to stderr so you can see it working
#         print(f"  Checking {d}...", file=sys.stderr)
        
#         if verify_wind_intensity(args.zarr_path, d, 'calm'):
#             final_calm.append(d)

#     # FINAL OUTPUT GOES TO STDOUT (This is the only thing eval sees)
#     storm_str = " ".join(d.strftime("%Y-%m-%d") for d in final_storms)
#     calm_str = " ".join(d.strftime("%Y-%m-%d") for d in sorted(final_calm))

#     print(f"STORM_DATES=\"{storm_str}\"")
#     print(f"CALM_DATES=\"{calm_str}\"")

# if __name__ == "__main__":
#     main()
import argparse
import xarray as xr
import pandas as pd
import numpy as np
import sys

# --- SETTINGS: MATCHING YOUR PCA SCRIPT ---
# Your PCA code uses: Lat 24-50, Lon 235-295
# ERA5 0.25 deg grid usually starts at 90N (Index 0) and 0E (Index 0)

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
        # This aligns perfectly with your PCA analysis mask
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