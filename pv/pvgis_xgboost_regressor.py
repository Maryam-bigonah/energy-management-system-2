#!/usr/bin/env python3
"""
Train XGBoost to predict PVGIS PV power (P) using *Open-Meteo* features, then predict P for 2025.

Key change vs your previous script: we no longer train on PVGIS POA components as inputs.
Instead, we train on Open-Meteo (2018–2023) -> PVGIS P (2018–2023), then apply to Open-Meteo 2025.
This fixes the feature/source mismatch in your earlier pipeline. :contentReference[oaicite:0]{index=0}

Example:
  python pvgis_openmeteo_xgboost_regressor.py \
      --pvgis_csv "/mnt/data/Pvgis-127panel,2018_2023.csv" \
      --meteo_csv "/mnt/data/open-meteo-2018-2025.csv" \
      --out_csv "pv_prediction_2025.csv" \
      --predict_year 2025
"""

import argparse
import datetime
import math
import os
import re
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# Helpers: robust CSV readers
# -----------------------------
def find_header_line(filepath: str, startswith: str) -> int:
    """Return 0-based line index where header starts (line begins with `startswith`)."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.strip().startswith(startswith):
                return i
    raise ValueError(f"Header starting with '{startswith}' not found in {filepath}")


def parse_pvgis_metadata(filepath: str) -> dict:
    """Extract lat/lon/elev/tilt/azimuth from PVGIS file header (first ~40 lines)."""
    meta = {
        "latitude": None,
        "longitude": None,
        "elevation_m": None,
        "tilt_deg": None,
        "azimuth_pvgis_deg": None,
    }
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(60):
            line = f.readline()
            if not line:
                break
            s = line.strip()

            if s.startswith("Latitude"):
                meta["latitude"] = float(s.split("\t")[-1])
            elif s.startswith("Longitude"):
                meta["longitude"] = float(s.split("\t")[-1])
            elif s.startswith("Elevation"):
                meta["elevation_m"] = float(s.split("\t")[-1])

            # PVGIS time-series exports often include these "Slope/Azimuth" lines
            elif s.startswith("Slope"):
                m = re.search(r"Slope:\s*([0-9\.\-]+)", s)
                if m:
                    meta["tilt_deg"] = float(m.group(1))
            elif s.startswith("Azimuth"):
                m = re.search(r"Azimuth:\s*([0-9\.\-]+)", s)
                if m:
                    meta["azimuth_pvgis_deg"] = float(m.group(1))

    return meta


def parse_openmeteo_metadata(filepath: str) -> dict:
    """
    Open-Meteo CSV begins with two lines:
      latitude,longitude,elevation,utc_offset_seconds,timezone,timezone_abbreviation
      45.xx,7.xx,253.0,3600,Europe/Berlin,GMT+1
    """
    meta = {
        "latitude": None,
        "longitude": None,
        "elevation_m": None,
        "utc_offset_seconds": None,
        "timezone": None,
        "timezone_abbreviation": None,
    }
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
        values = f.readline()
        if not header or not values:
            return meta
        h = [x.strip() for x in header.strip().split(",")]
        v = [x.strip() for x in values.strip().split(",")]
        if len(h) != len(v):
            return meta
        d = dict(zip(h, v))

    def _f(key, cast):
        if key in d and d[key] != "":
            try:
                return cast(d[key])
            except Exception:
                return None
        return None

    meta["latitude"] = _f("latitude", float)
    meta["longitude"] = _f("longitude", float)
    meta["elevation_m"] = _f("elevation", float)
    meta["utc_offset_seconds"] = _f("utc_offset_seconds", int)
    meta["timezone"] = d.get("timezone")
    meta["timezone_abbreviation"] = d.get("timezone_abbreviation")
    return meta


def read_pvgis_timeseries(filepath: str) -> pd.DataFrame:
    header_line = find_header_line(filepath, "time,")
    df = pd.read_csv(filepath, skiprows=header_line)

    # PVGIS sometimes appends legend/footer rows -> coerce then drop NaT
    df["dt"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M", errors="coerce")
    df = df.dropna(subset=["dt"]).copy()

    # Coerce numeric columns
    for c in df.columns:
        if c not in ("time", "dt"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def read_open_meteo(filepath: str) -> pd.DataFrame:
    header_line = find_header_line(filepath, "time,")
    df = pd.read_csv(filepath, skiprows=header_line)
    df["dt_end"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["dt_end"]).copy()

    # Coerce numeric columns
    for c in df.columns:
        if c not in ("time", "dt_end"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# -----------------------------------------
# Metrics
# -----------------------------------------
def eval_metrics(y_true, y_pred, label="") -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    denom = float(np.mean(y_true[y_true > 0])) if np.any(y_true > 0) else float(np.mean(y_true))
    nmae = float(mae / denom) if denom and denom > 0 else float("nan")
    nrmse = float(rmse / denom) if denom and denom > 0 else float("nan")

    print(f"[{label}] RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.6f}  nMAE={nmae:.4f}  nRMSE={nrmse:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2, "nmae": nmae, "nrmse": nrmse}


# -----------------------------------------
# Solar position (NOAA fractional-year)
# -----------------------------------------
def solar_position_noaa(times_local: pd.DatetimeIndex, lat_deg: float, lon_deg: float, tz_hours: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (solar_zenith_deg, solar_azimuth_deg_from_north, solar_elevation_deg).
    Azimuth is clockwise from North.
    times_local should be in the *same clock time* as tz_hours.
    """
    times = pd.DatetimeIndex(times_local)

    doy = times.dayofyear.values
    dec_hour = times.hour.values + times.minute.values / 60 + times.second.values / 3600

    gamma = 2 * np.pi / 365 * (doy - 1 + (dec_hour - 12) / 24)

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )

    time_offset = eqtime + 4 * lon_deg - 60 * tz_hours
    true_solar_time = (dec_hour * 60 + time_offset) % 1440

    hour_angle_deg = true_solar_time / 4 - 180
    ha = np.deg2rad(hour_angle_deg)

    lat = np.deg2rad(lat_deg)

    cos_zenith = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    zenith = np.arccos(cos_zenith)

    elevation = np.pi / 2 - zenith

    az = np.arctan2(np.sin(ha), (np.cos(ha) * np.sin(lat) - np.tan(decl) * np.cos(lat)))
    az_deg = (np.rad2deg(az) + 180) % 360

    return np.rad2deg(zenith), az_deg, np.rad2deg(elevation)


# -----------------------------------------
# Column detection (robust to units)
# -----------------------------------------
def pick_col(df: pd.DataFrame, contains: str) -> str:
    contains = contains.lower()
    for c in df.columns:
        if contains in c.lower():
            return c
    raise ValueError(f"Missing column containing '{contains}'. Available columns: {list(df.columns)}")


# -----------------------------------------
# Time-shift estimation (optional)
# -----------------------------------------
def estimate_best_shift_minutes(
    pv: pd.DataFrame,
    om: pd.DataFrame,
    gti_col: str,
    tolerance_minutes: int = 30,
    candidate_range: Tuple[int, int] = (-120, 120),
    step: int = 5,
) -> int:
    """
    Finds a minute shift to apply to Open-Meteo dt_end so it best aligns with PVGIS dt.

    We merge_asof PVGIS(dt) with OpenMeteo(dt_end + shift) and maximize corr(P, GTI) on daylight.
    Tie-breaker: smaller mean abs time difference.
    """
    pv = pv[["dt", "P", "H_sun"]].dropna(subset=["dt", "P"]).sort_values("dt").copy()
    om = om[["dt_end", gti_col]].dropna(subset=["dt_end", gti_col]).sort_values("dt_end").copy()

    # Focus on daylight to avoid all-night zeros dominating
    pv_day = pv[pv["H_sun"].fillna(0) > 0].copy()
    if len(pv_day) < 1000:
        pv_day = pv.copy()  # fallback

    best = None  # (corr, mean_abs_dt_seconds, shift)
    tol = pd.Timedelta(minutes=tolerance_minutes)

    for shift in range(candidate_range[0], candidate_range[1] + 1, step):
        om_tmp = om.copy()
        om_tmp["dt"] = om_tmp["dt_end"] + pd.Timedelta(minutes=shift)
        om_tmp = om_tmp.sort_values("dt")

        merged = pd.merge_asof(
            pv_day[["dt", "P"]],
            om_tmp[["dt", gti_col]],
            on="dt",
            tolerance=tol,
            direction="nearest",
        ).dropna(subset=[gti_col])

        if len(merged) < 500:
            continue

        corr = np.corrcoef(merged["P"].to_numpy(), merged[gti_col].to_numpy())[0, 1]
        if np.isnan(corr):
            continue

        # Time diff is 0 because we merge on same 'dt' key; approximate by nearest match count at exact minutes:
        # Use minute-of-hour mismatch as proxy:
        # (If shift yields exact minute alignment, it's usually better.)
        mean_abs_dt_seconds = 0.0  # keep simple

        cand = (float(corr), float(mean_abs_dt_seconds), int(shift))
        if best is None:
            best = cand
        else:
            # Max corr, then min time diff
            if cand[0] > best[0] + 1e-6:
                best = cand
            elif abs(cand[0] - best[0]) <= 1e-6 and cand[1] < best[1]:
                best = cand

    if best is None:
        print("WARN: Could not estimate shift; using 0 minutes.")
        return 0

    print(f"Auto shift selected: {best[2]} minutes (corr={best[0]:.4f})")
    return best[2]


# -----------------------------------------
# Feature engineering
# -----------------------------------------
def add_time_features(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    out = df.copy()
    dt = pd.DatetimeIndex(out[dt_col])

    doy = dt.dayofyear.astype(float)
    hod = (dt.hour + dt.minute / 60.0).astype(float)

    out["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    out["sin_hod"] = np.sin(2 * np.pi * hod / 24.0)
    out["cos_hod"] = np.cos(2 * np.pi * hod / 24.0)
    out["month"] = dt.month.astype(int)

    return out


def build_feature_frame(
    merged: pd.DataFrame,
    tz_hours: float,
    lat: float,
    lon: float,
    use_solar_pos: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Builds the final ML frame + returns feature column names.
    `merged` must contain 'dt' plus Open-Meteo columns (GTI/GHI/DNI/DHI/temp/wind/clouds).
    """
    df = merged.copy()

    # Detect columns (robust to units)
    col_temp = pick_col(df, "temperature_2m")
    col_wind = pick_col(df, "wind_speed_10m")
    col_gti = pick_col(df, "global_tilted_irradiance")
    col_ghi = pick_col(df, "shortwave_radiation")
    col_dhi = pick_col(df, "diffuse_radiation")
    col_dni = pick_col(df, "direct_normal_irradiance")
    col_cc = pick_col(df, "cloud_cover (%)") if any("cloud_cover (%)" in c.lower() for c in df.columns) else pick_col(df, "cloud_cover")
    col_cc_low = pick_col(df, "cloud_cover_low")
    col_cc_mid = pick_col(df, "cloud_cover_mid")
    col_cc_high = pick_col(df, "cloud_cover_high")

    # Basic features
    df["T2m_om"] = df[col_temp]
    df["WS10m_om"] = df[col_wind]  # already m/s in your file
    df["GTI"] = df[col_gti]
    df["GHI"] = df[col_ghi]
    df["DHI"] = df[col_dhi]
    df["DNI"] = df[col_dni]
    df["CC"] = df[col_cc]
    df["CC_low"] = df[col_cc_low]
    df["CC_mid"] = df[col_cc_mid]
    df["CC_high"] = df[col_cc_high]

    # Time features
    df = add_time_features(df, "dt")

    # Solar position features (optional but useful)
    if use_solar_pos:
        zen, az, el = solar_position_noaa(pd.DatetimeIndex(df["dt"]), lat_deg=lat, lon_deg=lon, tz_hours=tz_hours)
        df["solar_elev_deg"] = el
        df["cos_zenith"] = np.cos(np.deg2rad(zen)).clip(0, 1)

    # Interaction features that often help trees
    df["GTI_x_cosz"] = df["GTI"] * df.get("cos_zenith", 1.0)
    df["GHI_x_cosz"] = df["GHI"] * df.get("cos_zenith", 1.0)

    feature_cols = [
        "GTI", "GHI", "DHI", "DNI",
        "T2m_om", "WS10m_om",
        "CC", "CC_low", "CC_mid", "CC_high",
        "sin_doy", "cos_doy", "sin_hod", "cos_hod",
        "month",
        "GTI_x_cosz", "GHI_x_cosz",
    ]
    if use_solar_pos:
        feature_cols += ["solar_elev_deg", "cos_zenith"]

    return df, feature_cols


# -----------------------------------------
# Main pipeline
# -----------------------------------------
def main(
    pvgis_csv: str,
    meteo_csv: str,
    out_csv: str,
    predict_year: int = 2025,
    time_shift_minutes: int = -50,
    auto_time_shift: bool = False,
    merge_tolerance_minutes: int = 30,
    use_solar_pos: bool = True,
):
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(out_csv)
    final_out_csv = f"{base}_{timestamp_str}{ext}"

    pv_meta = parse_pvgis_metadata(pvgis_csv)
    om_meta = parse_openmeteo_metadata(meteo_csv)

    if pv_meta["latitude"] is None or pv_meta["longitude"] is None:
        raise ValueError(f"Could not parse PVGIS lat/lon: {pv_meta}")

    tz_hours = 0.0
    if om_meta.get("utc_offset_seconds") is not None:
        tz_hours = float(om_meta["utc_offset_seconds"]) / 3600.0

    # Warn if coordinates differ a lot
    if om_meta.get("latitude") is not None and om_meta.get("longitude") is not None:
        dist_km = haversine_km(pv_meta["latitude"], pv_meta["longitude"], om_meta["latitude"], om_meta["longitude"])
        if dist_km > 1.0:
            print(f"WARN: Open-Meteo point is ~{dist_km:.2f} km from PVGIS point. "
                  f"For best accuracy, use the same coordinates in both.")

    # Load data
    df_pv = read_pvgis_timeseries(pvgis_csv)
    df_om = read_open_meteo(meteo_csv)

    # Basic sanity
    needed_pv = ["dt", "P"]
    df_pv = df_pv.dropna(subset=needed_pv).copy()
    df_pv["year"] = df_pv["dt"].dt.year

    # Determine GTI column name for shift estimation
    gti_col_om = pick_col(df_om, "global_tilted_irradiance")

    # Estimate time shift automatically if requested
    if auto_time_shift:
        # PVGIS file often has H_sun; if missing, create a placeholder (auto-shift becomes less reliable)
        if "H_sun" not in df_pv.columns:
            df_pv["H_sun"] = 0.0
        time_shift_minutes = estimate_best_shift_minutes(
            pv=df_pv, om=df_om, gti_col=gti_col_om, tolerance_minutes=merge_tolerance_minutes
        )

    print(f"Using Open-Meteo time shift: {time_shift_minutes} minutes "
          f"(OpenMeteo dt = dt_end + shift; then merged to PVGIS dt).")

    # Apply shift and create merge key
    df_om = df_om.copy()
    df_om["dt"] = df_om["dt_end"] + pd.Timedelta(minutes=time_shift_minutes)
    df_om = df_om.sort_values("dt")

    df_pv = df_pv.sort_values("dt")

    # Merge Open-Meteo features onto PVGIS targets (2018–2023)
    tol = pd.Timedelta(minutes=merge_tolerance_minutes)
    merged = pd.merge_asof(
        df_pv,
        df_om,
        on="dt",
        tolerance=tol,
        direction="nearest",
        suffixes=("", "_om"),
    )

    # Keep only rows where Open-Meteo matched
    merged = merged.dropna(subset=[gti_col_om]).copy()

    # Build features from merged
    ml_df, feature_cols = build_feature_frame(
        merged=merged,
        tz_hours=tz_hours,
        lat=pv_meta["latitude"],
        lon=pv_meta["longitude"],
        use_solar_pos=use_solar_pos,
    )

    # Clean NaNs for training
    ml_df = ml_df.dropna(subset=feature_cols + ["P"]).copy()

    # Train/Val/Test split (same as your original idea)
    ml_df["year"] = ml_df["dt"].dt.year
    train = ml_df[ml_df["year"].between(2018, 2021)]
    val = ml_df[ml_df["year"] == 2022]
    test = ml_df[ml_df["year"] == 2023]

    print(f"Rows total after merge+clean: {len(ml_df)} | Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    X_train, y_train = train[feature_cols], train["P"]
    X_val, y_val = val[feature_cols], val["P"]
    X_test, y_test = test[feature_cols], test["P"]

    # Model
    model = XGBRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=4,
        random_state=42,
        early_stopping_rounds=100,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    best_iter = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)
    print(f"Best iteration: {best_iter} | Best val score: {best_score}")

    # Evaluate
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    eval_metrics(y_val, pred_val, "VAL@2022 (all hours)")
    eval_metrics(y_test, pred_test, "TEST@2023 (all hours)")

    # Daylight-only evaluation (use PVGIS H_sun if available; else use solar elevation feature)
    if "H_sun" in val.columns:
        val_day = val["H_sun"].fillna(0).to_numpy() > 0
    else:
        val_day = val.get("solar_elev_deg", pd.Series(0, index=val.index)).to_numpy() > 0

    if "H_sun" in test.columns:
        test_day = test["H_sun"].fillna(0).to_numpy() > 0
    else:
        test_day = test.get("solar_elev_deg", pd.Series(0, index=test.index)).to_numpy() > 0

    if np.any(val_day):
        eval_metrics(y_val[val_day], pred_val[val_day], "VAL@2022 (daylight)")
    if np.any(test_day):
        eval_metrics(y_test[test_day], pred_test[test_day], "TEST@2023 (daylight)")

    # -----------------------------------------
    # Predict for desired year from Open-Meteo
    # -----------------------------------------
    df_pred = df_om[df_om["dt_end"].dt.year == predict_year].copy()
    if len(df_pred) == 0:
        raise ValueError(f"No Open-Meteo rows found for predict_year={predict_year}. "
                         f"Available years: {sorted(df_om['dt_end'].dt.year.unique().tolist())}")

    # Build features on Open-Meteo-only frame
    # Note: we reuse the same feature builder by passing a "merged-like" df with 'dt' set.
    pred_frame = df_pred.copy()
    pred_frame["year"] = pred_frame["dt"].dt.year  # dt is already shifted key used for feature time
    pred_ml, _ = build_feature_frame(
        merged=pred_frame,
        tz_hours=tz_hours,
        lat=pv_meta["latitude"],
        lon=pv_meta["longitude"],
        use_solar_pos=use_solar_pos,
    )
    pred_ml = pred_ml.dropna(subset=feature_cols).copy()
    pred_ml["P_pred"] = pd.Series(model.predict(pred_ml[feature_cols]), index=pred_ml.index).clip(lower=0)

    # Save output
    out = pd.DataFrame({
        "dt_end": pred_ml["dt_end"],   # original Open-Meteo stamp
        "dt_aligned": pred_ml["dt"],   # shifted timestamp aligned to PVGIS
        "P_pred": pred_ml["P_pred"],
        "GTI": pred_ml["GTI"],
        "GHI": pred_ml["GHI"],
        "DHI": pred_ml["DHI"],
        "DNI": pred_ml["DNI"],
        "T2m": pred_ml["T2m_om"],
        "WS10m": pred_ml["WS10m_om"],
        "CC": pred_ml["CC"],
        "CC_low": pred_ml["CC_low"],
        "CC_mid": pred_ml["CC_mid"],
        "CC_high": pred_ml["CC_high"],
    })
    if use_solar_pos:
        out["solar_elev_deg"] = pred_ml["solar_elev_deg"]

    out.to_csv(final_out_csv, index=False)
    print(f"Saved predictions to: {final_out_csv}")

    # -----------------------------------------
    # Plots (optional but handy)
    # -----------------------------------------
    import matplotlib.pyplot as plt

    # 2023 daily max: true vs predicted (on test set)
    test_plot = test[["dt", "P"]].copy()
    test_plot["P_pred"] = pred_test
    test_plot["date"] = test_plot["dt"].dt.date
    daily_2023 = test_plot.groupby("date", as_index=False)[["P", "P_pred"]].max()
    daily_2023["date"] = pd.to_datetime(daily_2023["date"])

    plt.figure(figsize=(14, 6))
    plt.plot(daily_2023["date"], daily_2023["P"], label="PVGIS 2023 (true)", linewidth=1, alpha=0.8)
    plt.plot(daily_2023["date"], daily_2023["P_pred"], label="Model 2023 (pred)", linewidth=1, alpha=0.8)
    plt.xlabel("Date")
    plt.ylabel("Daily Max PV Power (W)")
    plt.title("2023 Daily Max: PVGIS (true) vs Model (Open-Meteo features)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot1 = f"{base}_{timestamp_str}_daily_max_2023_true_vs_pred.png"
    plt.savefig(plot1)
    plt.close()
    print(f"Saved plot: {plot1}")

    # 2023 vs 2025 daily max comparison (predicted 2025)
    out2 = out.copy()
    # Filter by dt_end year to avoid inclusion of previous year due to negative time shift
    out2["dt_end_ts"] = pd.to_datetime(out2["dt_end"])
    out2 = out2[out2["dt_end_ts"].dt.year == predict_year]
    
    # Use dt_end for grouping to stay within the target year
    out2["date"] = out2["dt_end_ts"].dt.date
    daily_2025 = out2.groupby("date", as_index=False)["P_pred"].max()
    daily_2025["date"] = pd.to_datetime(daily_2025["date"])
    daily_2025["doy"] = daily_2025["date"].dt.dayofyear
    daily_2025 = daily_2025.sort_values("doy")

    daily_2023b = daily_2023.copy()
    daily_2023b["doy"] = daily_2023b["date"].dt.dayofyear

    plt.figure(figsize=(14, 6))
    plt.plot(daily_2023b["doy"], daily_2023b["P"], label="PVGIS 2023 (true)", linewidth=1, alpha=0.8)
    plt.plot(daily_2025["doy"], daily_2025["P_pred"], label=f"{predict_year} (pred)", linewidth=1, alpha=0.8)
    plt.xlabel("Day of Year")
    plt.ylabel("Daily Max PV Power (W)")
    plt.title(f"Daily Max PV Power: 2023 (true) vs {predict_year} (pred)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot2 = f"{base}_{timestamp_str}_daily_max_2023_vs_{predict_year}.png"
    plt.savefig(plot2)
    plt.close()
    print(f"Saved plot: {plot2}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pvgis_csv", required=True, help="PVGIS timeseries CSV (2018-2023)")
    ap.add_argument("--meteo_csv", required=True, help="Open-Meteo CSV (2018-2025)")
    ap.add_argument("--out_csv", default="pv_prediction_2025.csv", help="Output CSV path")
    ap.add_argument("--predict_year", type=int, default=2025, help="Year to predict (default: 2025)")
    ap.add_argument("--time_shift_minutes", type=int, default=-50,
                    help="Shift applied to Open-Meteo timestamps: dt = dt_end + shift (default: -50)")
    ap.add_argument("--auto_time_shift", action="store_true",
                    help="Estimate best timestamp shift automatically (overrides --time_shift_minutes)")
    ap.add_argument("--merge_tolerance_minutes", type=int, default=30,
                    help="merge_asof tolerance in minutes (default: 30)")
    ap.add_argument("--no_solar_pos", action="store_true",
                    help="Disable solar position features (faster, slightly less accurate)")
    args = ap.parse_args()

    main(
        pvgis_csv=args.pvgis_csv,
        meteo_csv=args.meteo_csv,
        out_csv=args.out_csv,
        predict_year=args.predict_year,
        time_shift_minutes=args.time_shift_minutes,
        auto_time_shift=args.auto_time_shift,
        merge_tolerance_minutes=args.merge_tolerance_minutes,
        use_solar_pos=not args.no_solar_pos,
    )
