# Opening the Black Box of AI Weather Forecasting: Latent Space Dynamics and Attribution in the Aurora Model

This repository contains the code and methodology for the study **"Opening the Black Box of AI Weather Forecasting: Latent Space Dynamics and Attribution in the Aurora Model"**. This project investigates the internal latent representations of the Aurora AI weather model using:
- **Principal Component Analysis (PCA)** to disentangle physical regimes (e.g., Storm vs. Calm, Seasonal Cycles), and
- **Layer-wise Relevance Propagation (LRP)** to attribute specific predictions to input features.

--- 

## 🛠️ Installation & Prerequisites

Before running the code, ensure you have the following installed:

**Aurora Model**: The repository assumes the Aurora model environment is set up.

**Dependencies**:
```
pip install zennit cartopy xarray torch numpy matplotlib
```

---

## ⚠️ Data Requirements

The code was originally executed on the **Snellius supercomputer**. It requires access to **ERA5 reanalysis data in Zarr format**.

*Important*: Replace `--zarr-path` in the commands below with the path to your local ERA5 dataset.

---

## 1. PCA Analysis (Latent Space Dynamics)

Perform PCA on Aurora's latent space to identify structural modes of variability.

### Step 1: Prepare Dates

Identify specific dates that correspond to "Storm" and "Calm" regimes based on wind speed thresholds.

```bash
python prepare_pca_dates.py \
    --zarr-path /path/to/era5.zarr \
    --year 2005 \
    --storm-thresh 24 \
    --calm-thresh 18
```

- `--storm-thresh`: Wind speed threshold for storms (default: 24 m/s).
- `--calm-thresh`: Wind speed threshold for calm days (default: ≤18 m/s).

### Step 2: Run PCA

Once the dates are generated, project latent representations onto principal components.

- **For Storm/Calm Analysis**:
    ```bash
    python storm_pca.py \
        --zarr-path /path/to/era5.zarr \
        --static-path /path/to/static_data.nc \
        --storm-dates "2005-01-08 2005-01-09 ..." \
        --calm-dates "2005-06-15 2005-06-16 ..." \
        --output results/storm_pca.pt
    ```

- **For Seasonal Analysis**:
    Generate lists of dates for each season:
    ```bash
    cat <<EOF > get_dates.py
    from datetime import date
    import sys

    SEASONS = {
        "WINTER": [1, 2, 12], "SPRING": [3, 4, 5],
        "SUMMER": [6, 7, 8],  "FALL":   [9, 10, 11]
    }

    def get_all_dates(year, months):
        candidates = []
        for m in months:
            days = 28 if m == 2 else 30 if m not in [1,3,5,7,8,10,12] else 31
            for d in range(1, days + 1):
                candidates.append(date(year, m, d))
        candidates.sort()
        return " ".join([d.strftime("%Y-%m-%d") for d in candidates])

    print("WINTER_DATES=\"" + get_all_dates(2005, SEASONS["WINTER"]) + "\"")
    print("SPRING_DATES=\"" + get_all_dates(2005, SEASONS["SPRING"]) + "\"")
    print("SUMMER_DATES=\"" + get_all_dates(2005, SEASONS["SUMMER"]) + "\"")
    print("FALL_DATES=\""   + get_all_dates(2005, SEASONS["FALL"])   + "\"")
    EOF

    eval $(python get_dates.py)
    rm get_dates.py

    python seasonal_pca.py \
        --zarr-path /path/to/era5.zarr \
        --static-path /path/to/static_data.nc \
        --winter "$WINTER_DATES" \
        --spring "$SPRING_DATES" \
        --summer "$SUMMER_DATES" \
        --fall "$FALL_DATES" \
        --output results/seasonal_pca.pt
    ```

### Step 3: Analysis & Visualization

Analyze variance and visualize the PC components.

```bash
python analyze_pca_components.py \
    --file results/storm_pca.pt \
    --mode storm  # or 'seasonal'
```

### Step 4: Bootstrap Validation

Validate the stability of PCA components and store results in the `results/` directory.

- **Validate Storm PCA**:
    ```bash
    python bootstrap_pca.py
    ```
- **Validate Seasonal PCA**:
    ```bash
    python bootstrap_pca_seasonal.py
    ```

---

## 2. LRP Attribution (Interpretability)

Leverage Layer-wise Relevance Propagation to interpret specific predictions.

### Step 1: Run LRP

Compute relevance maps for a specific date (e.g., the Great Storm of 1987).

```bash
python aurora_lrp_europe_1.py \
    --zarr-path /path/to/era5.zarr \
    --static-path /path/to/static_data.nc \
    --date 1987-10-16 \
    --output results/lrp_europe_1987-10-16.pt
```

### Step 2: Visualization

- **Surface Variables**:
    ```bash
    python visualize_lrp_1.py \
        --file results/lrp_europe_1987-10-16.pt \
        --lat-min 30 --lat-max 70 \
        --lon-min -20 --lon-max 40 \
        --device cpu
    ```

- **Vertical Levels**:
    ```bash
    python visualize_levels.py \
        --file results/lrp_europe_1987-10-16.pt \
        --var u \
        --device cpu
    ```

Add `--no-map` to disable mapping if `cartopy` is not installed.

### Step 3: Perturbation Validation

Quantitatively validate LRP maps by perturbing the most relevant pixels and analyzing the forecast impact.

```bash
python validation_lrp.py \
    --zarr-path /path/to/era5.zarr \
    --static-path /path/to/static_data.nc \
    --date 1987-10-16 \
    --lrp-file results/lrp_europe_1987-10-16.pt \
    --output results/validation_impact.pt \
    --seed 42 \
    --n-random 10
```

---

## 📂 Repository Structure
- `prepare_pca_dates.py`: Prepares dates for PCA based on weather regimes.
- `storm_pca.py`: Executes PCA for storm/calm analysis.
- `seasonal_pca.py`: Executes PCA for seasonal analysis.
- `analyze_pca_components.py`: Analyzes variance of PCA components.
- `bootstrap_pca.py`: Performs bootstrap validation for storm PCA.
- `bootstrap_pca_seasonal.py`: Performs bootstrap validation for seasonal PCA.
- `aurora_lrp_europe_1.py`: Runs LRP to compute relevance maps.
- `visualize_lrp_1.py`: Visualizes surface relevance maps.
- `visualize_levels.py`: Visualizes atmospheric variables at vertical levels.
- `validation_lrp.py`: Validates LRP maps via perturbation.

---


## 🔗 References

- [Aurora Model Documentation](https://github.com/microsoft/aurora)
- [ERA5 Reanalysis Dataset](https://cds.climate.copernicus.eu/)

---

## 👩‍💻 Author
Emma Kast
