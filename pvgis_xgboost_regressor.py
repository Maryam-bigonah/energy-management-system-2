#!/usr/bin/env python3
import re
import argparse
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
    """Extract lat/lon/tilt/azimuth from PVGIS file header."""
    meta = {"latitude": None, "longitude": None, "elevation_m": None, "tilt_deg": None, "azimuth_pvgis_deg": None}
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(30):
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
            elif s.startswith("Slope"):
                m = re.search(r"Slope:\s*([0-9\.\-]+)", s)
                if m:
                    meta["tilt_deg"] = float(m.group(1))
            elif s.startswith("Azimuth"):
                m = re.search(r"Azimuth:\s*([0-9\.\-]+)", s)
                if m:
                    meta["azimuth_pvgis_deg"] = float(m.group(1))
    return meta


def read_pvgis_timeseries(filepath: str) -> pd.DataFrame:
    header_line = find_header_line(filepath, "time,")
    df = pd.read_csv(filepath, skiprows=header_line)
    df["dt"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M", errors="coerce")
    return df


def read_open_meteo(filepath: str) -> pd.DataFrame:
    header_line = find_header_line(filepath, "time,")
    df = pd.read_csv(filepath, skiprows=header_line)
    df["dt_end"] = pd.to_datetime(df["time"], errors="coerce")
    return df


# -----------------------------------------
# Metrics printing (full + daylight)
# -----------------------------------------
def eval_metrics(y_true, y_pred, label=""):
    # Convert to numpy first and force to float64
    y_true_np = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred_np = np.asarray(y_pred, dtype=np.float64).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
    mae = mean_absolute_error(y_true_np, y_pred_np)
    r2 = r2_score(y_true_np, y_pred_np)

    denom = np.mean(y_true_np[y_true_np > 0]) if np.any(y_true_np > 0) else np.mean(y_true_np)
    nmae = mae / denom if denom and denom > 0 else np.nan
    nrmse = rmse / denom if denom and denom > 0 else np.nan

    print(f"[{label}] RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.6f}  nMAE={nmae:.4f}  nRMSE={nrmse:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2, "nmae": nmae, "nrmse": nrmse}


# -----------------------------------------
# Solar position (NOAA fractional-year)
# -----------------------------------------
def solar_position_noaa(times_utc: pd.DatetimeIndex, lat_deg: float, lon_deg: float, tz_hours: float = 0.0):
    """
    Returns (solar_zenith_deg, solar_azimuth_deg_from_north, solar_elevation_deg).
    Azimuth is clockwise from North.
    """
    times = pd.DatetimeIndex(times_utc)

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
# POA components (isotropic + ground)
# -----------------------------------------
def poa_components_isotropic(
    dni, dhi, ghi,
    solar_zenith_deg, solar_azimuth_deg,
    surface_tilt_deg, surface_azimuth_deg,
    albedo=0.2
):
    """
    Returns Gb(i), Gd(i), Gr(i) in W/m².
    surface_azimuth_deg is clockwise from North.
    """
    theta_z = np.deg2rad(solar_zenith_deg)
    beta = np.deg2rad(surface_tilt_deg)
    gamma_s = np.deg2rad(solar_azimuth_deg)
    gamma_p = np.deg2rad(surface_azimuth_deg)

    cos_aoi = np.cos(theta_z) * np.cos(beta) + np.sin(theta_z) * np.sin(beta) * np.cos(gamma_s - gamma_p)
    cos_aoi = np.maximum(0.0, cos_aoi)

    gb = np.asarray(dni) * cos_aoi
    gd = np.asarray(dhi) * (1 + np.cos(beta)) / 2
    gr = np.asarray(ghi) * albedo * (1 - np.cos(beta)) / 2
    return gb, gd, gr


# -----------------------------------------
# Main pipeline
# -----------------------------------------
def main(pvgis_csv: str, meteo_csv: str, out_csv: str, albedo: float = 0.2):
    meta = parse_pvgis_metadata(pvgis_csv)
    if any(meta[k] is None for k in ["latitude", "longitude", "tilt_deg", "azimuth_pvgis_deg"]):
        raise ValueError(f"Could not parse PVGIS metadata correctly: {meta}")

    # PVGIS azimuth convention: 0=south, east negative, west positive.
    # Convert to clockwise-from-North:
    surface_azimuth_deg = 180 + meta["azimuth_pvgis_deg"]

    # ---------------- Train/Val/Test on PVGIS ----------------
    df_pv = read_pvgis_timeseries(pvgis_csv).copy()

    needed = ["dt", "P", "Gb(i)", "Gd(i)", "Gr(i)", "H_sun", "T2m", "WS10m"]
    df_pv = df_pv.dropna(subset=needed)

    features = ["Gb(i)", "Gd(i)", "Gr(i)", "H_sun", "T2m", "WS10m"]
    target = "P"

    df_pv["year"] = df_pv["dt"].dt.year

    train = df_pv[df_pv["year"].between(2018, 2021)]
    val   = df_pv[df_pv["year"] == 2022]
    test  = df_pv[df_pv["year"] == 2023]

    print(f"Rows total: {len(df_pv)} | Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print("Any NaNs in features?", df_pv[features].isna().any().to_dict())

    X_train, y_train = train[features], train[target]
    X_val, y_val     = val[features], val[target]
    X_test, y_test   = test[features], test[target]

    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=4,
        random_state=42,
        early_stopping_rounds=100,  # XGBoost 3.x: goes in constructor
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    best_iter = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)
    print(f"Best iteration: {best_iter} | Best val score: {best_score}")

    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    # All-hours metrics
    eval_metrics(y_val, pred_val, "PVGIS VAL@2022 (all hours)")
    eval_metrics(y_test, pred_test, "PVGIS TEST@2023 (all hours)")

    # Daylight-only metrics (recommended)
    val_day_mask = val["H_sun"].values > 0
    test_day_mask = test["H_sun"].values > 0

    if np.any(val_day_mask):
        eval_metrics(y_val[val_day_mask], pred_val[val_day_mask], "PVGIS VAL@2022 (daylight)")
    else:
        print("[PVGIS VAL@2022 (daylight)] No daylight samples found (check H_sun).")

    if np.any(test_day_mask):
        eval_metrics(y_test[test_day_mask], pred_test[test_day_mask], "PVGIS TEST@2023 (daylight)")
    else:
        print("[PVGIS TEST@2023 (daylight)] No daylight samples found (check H_sun).")

    # ---------------- Build 2025 features from Open-Meteo ----------------
    df_w = read_open_meteo(meteo_csv).copy()
    df_w = df_w.dropna(subset=["dt_end"])

    # Open-Meteo radiation is preceding-hour mean -> midpoint time for solar position
    df_w["dt"] = df_w["dt_end"] - pd.Timedelta(minutes=30)

    col_t = "temperature_2m (°C)"
    col_dni = "direct_normal_irradiance (W/m²)"
    col_dhi = "diffuse_radiation (W/m²)"
    col_ghi = "shortwave_radiation (W/m²)"
    col_wind_kmh = "wind_speed_10m (km/h)"

    for c in [col_t, col_dni, col_dhi, col_ghi, col_wind_kmh]:
        if c not in df_w.columns:
            raise ValueError(f"Missing column in Open-Meteo file: {c}")

    zen, az, el = solar_position_noaa(
        pd.DatetimeIndex(df_w["dt"]),
        lat_deg=meta["latitude"],
        lon_deg=meta["longitude"],
        tz_hours=0.0,  # Open-Meteo sample shows GMT
    )

    gb, gd, gr = poa_components_isotropic(
        dni=df_w[col_dni].to_numpy(),
        dhi=df_w[col_dhi].to_numpy(),
        ghi=df_w[col_ghi].to_numpy(),
        solar_zenith_deg=zen,
        solar_azimuth_deg=az,
        surface_tilt_deg=meta["tilt_deg"],
        surface_azimuth_deg=surface_azimuth_deg,
        albedo=albedo,
    )

    out = pd.DataFrame({
        "time": df_w["dt"],  # midpoint timestamps
        "Gb(i)": gb,
        "Gd(i)": gd,
        "Gr(i)": gr,
        "H_sun": np.maximum(0, el),
        "T2m": df_w[col_t],
        "wind_speed_10m_kmh": df_w[col_wind_kmh],
        "wind_speed_10m_mph": df_w[col_wind_kmh] * 0.621371,
        "WS10m": df_w[col_wind_kmh] / 3.6,  # m/s to match PVGIS WS10m
    })

    out["P_pred"] = pd.Series(model.predict(out[features]), index=out.index).clip(lower=0)

    out.to_csv(out_csv, index=False)
    print(f"Saved predictions to: {out_csv}")

    # --- Plotting ---
    import matplotlib.pyplot as plt

    # Build aligned, numeric plot frame
    test_plot = test[["dt", "H_sun"]].copy()

    test_plot["P_true"] = pd.to_numeric(y_test, errors="coerce")
    test_plot["P_pred"] = pd.to_numeric(pd.Series(pred_test, index=test.index), errors="coerce")

    test_plot = test_plot.dropna(subset=["P_true", "P_pred"])

    print("DEBUG P_true dtype:", test_plot["P_true"].dtype,
          "min/max:", float(test_plot["P_true"].min()), float(test_plot["P_true"].max()))
    print("DEBUG P_pred dtype:", test_plot["P_pred"].dtype,
          "min/max:", float(test_plot["P_pred"].min()), float(test_plot["P_pred"].max()))
    print("DEBUG sample:\n", test_plot.head(5))

    # 1) Time series (daily max for full year - 365 points)
    test_daily = test_plot.copy()
    test_daily["date"] = test_daily["dt"].dt.date
    daily_max = test_daily.groupby("date").agg({
        "P_true": "max",
        "P_pred": "max"
    }).reset_index()
    daily_max["date"] = pd.to_datetime(daily_max["date"])
    
    plt.figure(figsize=(14, 6))
    plt.plot(daily_max["date"], daily_max["P_true"], label="True", linewidth=1, alpha=0.7, marker="o", markersize=2)
    plt.plot(daily_max["date"], daily_max["P_pred"], label="Predicted", linewidth=1, alpha=0.7, marker="o", markersize=2)
    plt.xlabel("Date")
    plt.ylabel("Daily Max PV Power (W)")
    plt.title("Test 2023: True vs Predicted (daily max)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2) Scatter (all hours)
    plt.figure()
    plt.scatter(test_plot["P_true"], test_plot["P_pred"], s=3, alpha=0.25)
    mx = float(max(test_plot["P_true"].max(), test_plot["P_pred"].max()))
    plt.plot([0, mx], [0, mx])
    plt.xlabel("True P")
    plt.ylabel("Predicted P")
    plt.title("Test 2023: Predicted vs True (all hours)")
    plt.tight_layout()
    plt.show()

    # 3) Scatter (daylight only)
    day = test_plot[test_plot["H_sun"] > 0]
    plt.figure()
    plt.scatter(day["P_true"], day["P_pred"], s=3, alpha=0.25)
    mx = float(max(day["P_true"].max(), day["P_pred"].max()))
    plt.plot([0, mx], [0, mx])
    plt.xlabel("True P (daylight)")
    plt.ylabel("Predicted P (daylight)")
    plt.title("Test 2023: Predicted vs True (daylight only)")
    plt.tight_layout()
    plt.show()

    # --- FIXED: 2023 vs 2025 PV Power (daily max) ---

    # 2023 daily max
    daily_2023 = (
        test_plot.assign(date=test_plot["dt"].dt.date)
        .groupby("date", as_index=False)["P_true"].max()
    )
    daily_2023["date"] = pd.to_datetime(daily_2023["date"])
    daily_2023["doy"] = daily_2023["date"].dt.dayofyear

    # 2025 daily max
    daily_2025 = (
        out.assign(date=out["time"].dt.date)
        .groupby("date", as_index=False)["P_pred"].max()
    )
    daily_2025["date"] = pd.to_datetime(daily_2025["date"])
    daily_2025["doy"] = daily_2025["date"].dt.dayofyear

    # Ensure one value per DOY (safety) + sort
    daily_2023 = daily_2023.groupby("doy", as_index=False)["P_true"].max().sort_values("doy")
    daily_2025 = daily_2025.groupby("doy", as_index=False)["P_pred"].max().sort_values("doy")

    # Debug checks (do this once)
    print("2023 DOY unique:", daily_2023["doy"].is_unique, "min/max:", daily_2023["doy"].min(), daily_2023["doy"].max())
    print("2025 DOY unique:", daily_2025["doy"].is_unique, "min/max:", daily_2025["doy"].min(), daily_2025["doy"].max())

    plt.figure(figsize=(14, 6))
    plt.plot(daily_2023["doy"], daily_2023["P_true"], label="2023", linewidth=1, alpha=0.8)
    plt.plot(daily_2025["doy"], daily_2025["P_pred"], label="2025", linewidth=1, alpha=0.8)
    plt.xlabel("Day of Year")
    plt.ylabel("Daily Max PV Power (W)")
    plt.title("PV Power: 2023 vs 2025 (daily max)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pvgis_csv", required=True, help="PVGIS 2018-2023 timeseries CSV")
    ap.add_argument("--meteo_csv", required=True, help="Open-Meteo 2025 CSV")
    ap.add_argument("--out_csv", default="pv_prediction_2025.csv", help="Output CSV path")
    ap.add_argument("--albedo", type=float, default=0.2, help="Ground albedo (e.g., 0.2 typical)")
    args = ap.parse_args()

    main(args.pvgis_csv, args.meteo_csv, args.out_csv, albedo=args.albedo)
