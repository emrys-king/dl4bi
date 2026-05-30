"""
TROPOMI Ozone Pipeline — Single Spatial Snapshot
=================================================
Downloads TROPOMI L2 ozone data for a single date from the Copernicus Data
Space Ecosystem (CDSE), quality-filters it, and saves a flat point cloud of
(lon, lat, ozone_du) suitable for direct use as GP training data.

Census-tract overlays are deferred to a later stage applied to GP predictions.

Setup (once, in your conda environment on nvidia7)
--------------------------------------------------
    pip install requests xarray netCDF4 h5py pandas numpy tqdm

Credentials
-----------
    export CDSE_USER="your@email.com"
    export CDSE_PASSWORD="yourpassword"
    (Register free at https://dataspace.copernicus.eu)

Usage
-----
    python tropomi_pipeline.py
"""

# =============================================================================
# Configuration
# =============================================================================

CDSE_USER     = "emrys.king25@imperial.ac.uk"   # or set directly, e.g. "your@email.com"
CDSE_PASSWORD = "QFFN8w-VJj@n2Hvo"   # or set directly

# The date to download and process
DATE = "2021-07-15"

# Continental US bounding box [min_lon, min_lat, max_lon, max_lat]
# Reduce this during development (e.g. one state) to limit download time.
# Example for California only: [-124.5, 32.5, -114.0, 42.0]
BBOX = [-125.0, 24.0, -66.0, 50.0]

# TROPOMI QA threshold — 0.5 is the team's recommendation for ozone
QA_THRESHOLD = 0.5

# Where to store files
DATA_DIR = "sm_kernels/data"
RAW_DIR  = "data/raw_tropomi"    # downloaded NetCDF files

# =============================================================================
# Imports
# =============================================================================

import os
import re
import time
import logging
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
# Step 1: Authenticate with CDSE
# =============================================================================

def cdse_token(user: str, password: str) -> str:
    """Obtain a short-lived access token from the CDSE identity service."""
    resp = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
        "/protocol/openid-connect/token",
        data={
            "grant_type": "password",
            "client_id":  "cdse-public",
            "username":   user,
            "password":   password,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]

# =============================================================================
# Step 2: Search the CDSE catalogue
# =============================================================================

def search_tropomi(date: str, bbox: list, token: str) -> list[dict]:
    """Return a list of TROPOMI L2 ozone products covering bbox on date.

    Each product dict contains 'Id', 'Name', and 'ContentLength'.
    Typically 3–5 files cover CONUS on a given day (one per orbit).
    """
    lon0, lat0, lon1, lat1 = bbox
    footprint = (
        f"POLYGON(({lon0} {lat0},{lon1} {lat0},"
        f"{lon1} {lat1},{lon0} {lat1},{lon0} {lat0}))"
    )
    params = {
        "$filter": (
            "Collection/Name eq 'SENTINEL-5P' and "
            "Attributes/OData.CSC.StringAttribute/any("
            "  att:att/Name eq 'productType' and "
            "  att/OData.CSC.StringAttribute/Value eq 'L2__O3____') and "
            f"ContentDate/Start ge {date}T00:00:00.000Z and "
            f"ContentDate/Start le {date}T23:59:59.999Z and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;{footprint}')"
        ),
        "$orderby": "ContentDate/Start",
        "$top": 20,
    }
    resp = requests.get(
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products",
        params=params,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    resp.raise_for_status()
    products = resp.json().get("value", [])
    log.info(f"Found {len(products)} TROPOMI files for {date}")
    return products

# =============================================================================
# Step 3: Download a file
# =============================================================================

def download_file(product: dict, dest_dir: str, token: str) -> Path:
    """Download one TROPOMI product to dest_dir.

    Skips the download if the file already exists at the correct size,
    so re-running the script is safe.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / product["Name"]

    if dest.exists() and dest.stat().st_size == product.get("ContentLength", 0):
        log.info(f"  Already downloaded: {product['Name']}")
        return dest

    url = (
        "https://download.dataspace.copernicus.eu/odata/v1/"
        f"Products({product['Id']})/$value"
    )
    with requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        stream=True,
        timeout=300,
    ) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=product["Name"][:55], leave=False,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    return dest

# =============================================================================
# Step 4: Process one NetCDF file → flat point cloud
# =============================================================================

def process_file(nc_path: Path, bbox: list) -> pd.DataFrame:
    """Extract valid ozone pixels from one TROPOMI orbit file.

    Returns a DataFrame with columns [lon, lat, ozone_du], one row per
    valid TROPOMI pixel within the bounding box.  Returns an empty
    DataFrame if the file contains no valid pixels in the study area.

    Variable notes
    --------------
    ozone_total_column : float32 [mol/m²]
        Total atmospheric ozone column. Multiply by 1/4.4615e-4 to get
        Dobson Units (DU), the conventional unit for ozone column amount.
    qa_value : float32 [0–1]
        Retrieval quality flag. Use >= 0.5 as recommended by the TROPOMI
        L2 ozone product documentation (section 4.7).
    latitude/longitude : float32 [degrees]
        Pixel centre coordinates. Shape is [time, scanline, pixel].
        The time dimension always has length 1 per file.
    """
    try:
        ds = xr.open_dataset(nc_path, group="PRODUCT", engine="netcdf4")
    except Exception as e:
        log.warning(f"Could not open {nc_path.name}: {e}")
        return pd.DataFrame()

    try:
        # Squeeze out the time dimension (length 1 per orbit file),
        # leaving shape [scanline, pixel] for all arrays.
        ozone = ds["ozone_total_vertical_column"].values.squeeze()   # [scanline, pixel]
        qa    = ds["qa_value"].values.squeeze()
        lat   = ds["latitude"].values.squeeze()
        lon   = ds["longitude"].values.squeeze()
    except KeyError as e:
        log.warning(f"Missing variable {e} in {nc_path.name}")
        return pd.DataFrame()
    finally:
        ds.close()

    # Build a boolean mask: quality filter AND bounding box
    lon0, lat0, lon1, lat1 = bbox
    mask = (
        (qa  >= QA_THRESHOLD) &
        (lon >= lon0) & (lon <= lon1) &
        (lat >= lat0) & (lat <= lat1)
    )

    n_valid = mask.sum()
    log.info(f"  {nc_path.name}: {n_valid:,} valid pixels")

    if n_valid == 0:
        return pd.DataFrame()

    # Convert mol/m² → Dobson Units
    ozone_du = ozone[mask] / 4.4615e-4

    return pd.DataFrame({
        "lon":      lon[mask].ravel(),
        "lat":      lat[mask].ravel(),
        "ozone_du": ozone_du.ravel(),
    })

# =============================================================================
# Step 5: Main pipeline
# =============================================================================

def run_pipeline():
    # --- Credentials ---
    user     = CDSE_USER     or os.environ.get("CDSE_USER")
    password = CDSE_PASSWORD or os.environ.get("CDSE_PASSWORD")
    if not user or not password:
        raise RuntimeError(
            "CDSE credentials not set. "
            "Set CDSE_USER and CDSE_PASSWORD as environment variables, "
            "or edit the config section at the top of this file."
        )

    log.info(f"Date: {DATE}  |  BBox: {BBOX}")

    # --- Auth ---
    token = cdse_token(user, password)

    # --- Search ---
    products = search_tropomi(DATE, BBOX, token)
    if not products:
        log.error(f"No TROPOMI files found for {DATE}. "
                  "Check the date and bounding box.")
        return

    # --- Download + process ---
    frames = []
    for product in products:
        local = download_file(product, RAW_DIR, token)
        frame = process_file(local, BBOX)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        log.error("No valid pixels extracted. Check QA threshold and bounding box.")
        return

    # Concatenate all orbits into a single point cloud for the day
    df = pd.concat(frames, ignore_index=True)

    # Drop duplicate pixels where orbit swaths overlap
    df = df.drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True)

    # --- Save ---
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(DATA_DIR) / f"ozone_{DATE}.parquet"
    df.to_parquet(out_path, index=False)

    # --- Summary ---
    log.info(f"Saved {len(df):,} pixels → {out_path}")
    print(f"\n=== Summary ===")
    print(f"  Pixels:          {len(df):,}")
    print(f"  Ozone mean (DU): {df['ozone_du'].mean():.1f}")
    print(f"  Ozone std  (DU): {df['ozone_du'].std():.1f}")
    print(f"  Ozone range:     {df['ozone_du'].min():.1f} – {df['ozone_du'].max():.1f}")
    print(f"\nTo load in your GP notebook:")
    print(f"  import pandas as pd, numpy as np")
    print(f"  df = pd.read_parquet('{out_path}')")
    print(f"  s  = df[['lon', 'lat']].values   # [N, 2] — spatial locations")
    print(f"  y  = df['ozone_du'].values        # [N]    — observations")

if __name__ == "__main__":
    run_pipeline()