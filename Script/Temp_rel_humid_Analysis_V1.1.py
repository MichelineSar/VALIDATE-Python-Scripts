import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

############################
# Colors
############################

GOLDENROD = "#D6B30F"
SLATEGREY = "#345C81"
BLACK = "black"

############################
# Paths
############################

BASE_DIR = Path(r"U:/Home/pyvuli54/1_Projekte/5_VALIDATE/4_Data/Data_Analysis")

############################
# Load ground truth data
############################

df_groundTruth = pd.read_csv(
    BASE_DIR / "Klimakammer_1" / "Rohdaten_KK_V1_V2_Sensirion.csv",
    sep=";",
    usecols=["Local (GMT +01:00)", "temperature", "humidity"]
)

df_groundTruth["Local (GMT +01:00)"] = pd.to_datetime(
    df_groundTruth["Local (GMT +01:00)"],
    errors="coerce"
).dt.tz_localize(None)

############################
# Define time windows
############################

start_K1 = pd.to_datetime("2025-03-20 16:55:00")
end_K1   = pd.to_datetime("2025-03-21 16:56:00")

start_K2 = pd.to_datetime("2025-03-21 17:43:00")
end_K2   = pd.to_datetime("2025-03-22 17:44:00")

df_groundTruth_K1 = df_groundTruth[
    (df_groundTruth["Local (GMT +01:00)"] >= start_K1) &
    (df_groundTruth["Local (GMT +01:00)"] <= end_K1)
]

df_groundTruth_K2 = df_groundTruth[
    (df_groundTruth["Local (GMT +01:00)"] >= start_K2) &
    (df_groundTruth["Local (GMT +01:00)"] <= end_K2)
]

############################
# Load sensor data
############################

def load_chamber(folder) -> pd.DataFrame:
    folder = Path(folder)

    files = [
        f for f in folder.glob("*.csv")
        if f.stem.split("_")[0].isdigit()
    ]

    print(f"📂 Found {len(files)} files in {folder}:")
    for f in files:
        print(f"   - {f.name}")

    frames = []

    for f in files:
        try:
            print(f"➡️ Reading {f.name} ...")

            with open(f, "r", encoding="utf-8") as fh:
                header_line = fh.readline()
                sep = ";" if ";" in header_line and "," not in header_line else ","

            df = pd.read_csv(
                f,
                sep=sep,
                usecols=["serial_number", "temperature", "humidity", "sensortime"],
                dtype={"serial_number": "string"},
                low_memory=False
            )

            df["source_file"] = f.name

            df["sensortime"] = pd.to_datetime(
                df["sensortime"].astype("string"),
                utc=True,
                errors="coerce"
            )

            df["sensortime"] = (
                df["sensortime"]
                .dt.tz_convert("Etc/GMT-1")
                .dt.tz_localize(None)
            )

            df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
            df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")

            print(f"   ✅ Loaded {len(df)} rows from {f.name}")
            frames.append(df)

        except Exception as e:
            print(f"   ⚠️ Skipping {f.name}: {e}")

    if not frames:
        print(f"⚠️ No valid files found in {folder}")
        return pd.DataFrame(
            columns=[
                "serial_number",
                "temperature",
                "humidity",
                "sensortime",
                "source_file",
                "chamber"
            ]
        )

    df = pd.concat(frames, ignore_index=True)
    df["chamber"] = folder.name

    print(f"📊 Combined {len(df)} rows from {len(frames)} files in {folder}")

    return df


df_K1 = load_chamber(BASE_DIR / "Klimakammer_1")
df_K2 = load_chamber(BASE_DIR / "Klimakammer_2")

df_K1_filtered = df_K1[
    (df_K1["sensortime"] >= start_K1) &
    (df_K1["sensortime"] <= end_K1)
]

df_K2_filtered = df_K2[
    (df_K2["sensortime"] >= start_K2) &
    (df_K2["sensortime"] <= end_K2)
]

############################
# Plotting
############################

def plot_temp_humidity_comparison(
    df_ground,
    df_sensor,
    time_col_ground,
    time_col_sensor,
    title
):

    df_ground = df_ground.copy()
    df_sensor = df_sensor.copy()

    df_ground[time_col_ground] = pd.to_datetime(df_ground[time_col_ground], errors="coerce")
    df_sensor[time_col_sensor] = pd.to_datetime(df_sensor[time_col_sensor], errors="coerce")

    df_ground["temperature"] = pd.to_numeric(df_ground["temperature"], errors="coerce")
    df_ground["humidity"] = pd.to_numeric(df_ground["humidity"], errors="coerce")
    df_sensor["temperature"] = pd.to_numeric(df_sensor["temperature"], errors="coerce")
    df_sensor["humidity"] = pd.to_numeric(df_sensor["humidity"], errors="coerce")

    df_ground = df_ground.dropna(subset=[time_col_ground, "temperature", "humidity"])
    df_sensor = df_sensor.dropna(subset=[time_col_sensor, "temperature", "humidity"])

    if df_ground.empty or df_sensor.empty:
        print(f"⚠️ No valid data for {title}")
        return

    # -------------------------
    # 30 min stabilization cutoff
    # -------------------------
    t0 = min(df_ground[time_col_ground].min(), df_sensor[time_col_sensor].min())
    cutoff = t0 + pd.Timedelta(minutes=30)

    df_ground = df_ground[df_ground[time_col_ground] >= cutoff]
    df_sensor = df_sensor[df_sensor[time_col_sensor] >= cutoff]

    if df_ground.empty or df_sensor.empty:
        print(f"⚠️ No data after cutoff for {title}")
        return

    # -------------------------
    # 1-minute binning
    # -------------------------
    df_sensor["time_group"] = df_sensor[time_col_sensor].dt.floor("1min")

    sensor_summary = (
        df_sensor
        .groupby("time_group")
        .agg(
            temp_mean=("temperature", "mean"),
            temp_sd=("temperature", "std"),
            temp_n=("temperature", "count"),
            humid_mean=("humidity", "mean"),
            humid_sd=("humidity", "std"),
            humid_n=("humidity", "count"),
        )
        .reset_index()
    )

    sensor_summary["temp_ci"] = 1.96 * sensor_summary["temp_sd"] / np.sqrt(sensor_summary["temp_n"])
    sensor_summary["humid_ci"] = 1.96 * sensor_summary["humid_sd"] / np.sqrt(sensor_summary["humid_n"])

    sensor_summary[["temp_ci", "humid_ci"]] = sensor_summary[["temp_ci", "humid_ci"]].fillna(0)

    df_ground = df_ground.sort_values(by=time_col_ground)
    sensor_summary = sensor_summary.sort_values(by="time_group")

    ground_hours = (df_ground[time_col_ground] - cutoff).dt.total_seconds() / 3600
    sensor_hours = (sensor_summary["time_group"] - cutoff).dt.total_seconds() / 3600

    # -------------------------
    # STYLE (Nature-like)
    # -------------------------
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "figure.dpi": 300,
    })

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8.5, 6.5),
        sharex=True
    )

    fig.subplots_adjust(hspace=0.25)


    # -------------------------
    # Temperature
    # -------------------------
    ax1.plot(
        ground_hours,
        df_ground["temperature"],
        color="black",
        linewidth=0.5,
        alpha=0.75,
        label="Reference logger"
    )

    ax1.fill_between(
        sensor_hours,
        sensor_summary["temp_mean"] - sensor_summary["temp_ci"],
        sensor_summary["temp_mean"] + sensor_summary["temp_ci"],
        color=GOLDENROD,
        alpha=0.35,
        linewidth=0,
        label="95% CI"
    )

    ax1.plot(
        sensor_hours,
        sensor_summary["temp_mean"],
        color=GOLDENROD,
        linewidth=0.5,
        label="BEAM sensors mean"
    )

    ax1.set_ylabel("Temperature (°C)")

    ax1.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0
    )

    ax1.text(0.01, 0.95, "a", transform=ax1.transAxes, fontweight="bold", va="top")

    # -------------------------
    # Humidity
    # -------------------------
    ax2.plot(
        ground_hours,
        df_ground["humidity"],
        color="black",
        linewidth=0.5,
        alpha=0.75,
        label="Reference logger"
    )

    ax2.fill_between(
        sensor_hours,
        sensor_summary["humid_mean"] - sensor_summary["humid_ci"],
        sensor_summary["humid_mean"] + sensor_summary["humid_ci"],
        color=SLATEGREY,
        alpha=0.35,
        linewidth=0,
        label="95% CI"
    )

    ax2.plot(
        sensor_hours,
        sensor_summary["humid_mean"],
        color=SLATEGREY,
        linewidth=0.5,
        label="BEAM sensors mean"
    )

    ax2.set_xlabel("Time after stabilization phase (hours)")
    ax2.set_ylabel("Relative humidity (%)")

    ax2.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0
    )

    ax2.text(0.01, 0.95, "b", transform=ax2.transAxes, fontweight="bold", va="top")

    # -------------------------
    # Final layout
    # -------------------------
    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    plt.show()


plot_temp_humidity_comparison(
    df_groundTruth_K1,
    df_K1_filtered,
    time_col_ground="Local (GMT +01:00)",
    time_col_sensor="sensortime",
    title="Condition 1 (25°C and 40% RH)"
)

plot_temp_humidity_comparison(
    df_groundTruth_K2,
    df_K2_filtered,
    time_col_ground="Local (GMT +01:00)",
    time_col_sensor="sensortime",
    title="Condition 2 (40°C and 75% RH)"
)

############################
# Statistics
############################

def summarize_chamber_overall(df_sensor):
    summary = {
        "temp_mean": df_sensor["temperature"].mean(),
        "temp_median": df_sensor["temperature"].median(),
        "temp_std": df_sensor["temperature"].std(),
        "temp_n": df_sensor["temperature"].count(),
        "humid_mean": df_sensor["humidity"].mean(),
        "humid_median": df_sensor["humidity"].median(),
        "humid_std": df_sensor["humidity"].std(),
        "humid_n": df_sensor["humidity"].count(),
    }

    return pd.Series(summary)


def apply_cutoff(df, time_col):
    t0 = df[time_col].min()
    cutoff = t0 + pd.Timedelta(minutes=30)
    return df[df[time_col] >= cutoff]

df_K1_stats = apply_cutoff(df_K1_filtered, "sensortime")
df_K2_stats = apply_cutoff(df_K2_filtered, "sensortime")

############################
# Statistics after 30-min stabilization phase
############################

def remove_first_30_minutes(df_sensor, time_col):
    df_sensor = df_sensor.copy()
    df_sensor[time_col] = pd.to_datetime(df_sensor[time_col], errors="coerce")

    df_sensor = df_sensor.dropna(subset=[time_col])

    cutoff = df_sensor[time_col].min() + pd.Timedelta(minutes=30)

    return df_sensor[df_sensor[time_col] >= cutoff]


df_K1_stats = remove_first_30_minutes(df_K1_filtered, "sensortime")
df_K2_stats = remove_first_30_minutes(df_K2_filtered, "sensortime")


summary_K1 = summarize_chamber_overall(df_K1_stats)
summary_K2 = summarize_chamber_overall(df_K2_stats)

def add_ci(summary):
    summary["temp_ci"] = 1.96 * summary["temp_std"] / np.sqrt(summary["temp_n"])
    summary["humid_ci"] = 1.96 * summary["humid_std"] / np.sqrt(summary["humid_n"])
    return summary

summary_K1 = add_ci(summary_K1)
summary_K2 = add_ci(summary_K2)

print("\nSummary Statistics for Klimakammer 1 after 30-min stabilization:")
print(summary_K1)

print("\nSummary Statistics for Klimakammer 2 after 30-min stabilization:")
print(summary_K2)

"""
Sensitivity analysis for the stabilization lag-time cutoff.

This snippet is meant to be appended to your existing script. It reuses
the already-loaded dataframes:
    df_groundTruth_K1, df_groundTruth_K2   (reference logger, full window)
    df_K1_filtered, df_K2_filtered         (BEAM sensors, full window, BEFORE
                                             the 30-min stabilization cutoff)
and the color constants GOLDENROD, SLATEGREY, BLACK already defined above.

It answers two things for the reviewer:
1) A quantitative sensitivity analysis (printed table + CSV export): how do
   bias/RMSE change if the lag-time cutoff is 0, 15, 30, or 60 minutes?
2) A visual justification: a plot of the reference logger's own
   temperature and humidity traces for Condition 1, to show empirically
   when the chamber physically reaches its setpoint (independent of any
   statistical cutoff).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LAG_TIMES = [0, 15, 30, 60]  # minutes


############################
# 1. Align reference logger and BEAM sensors on 1-min bins
############################

def merge_ground_and_sensor(df_ground, df_sensor, time_col_ground, time_col_sensor,
                             tolerance="30s", label=""):
    df_ground = df_ground.copy()
    df_sensor = df_sensor.copy()

    df_ground[time_col_ground] = pd.to_datetime(df_ground[time_col_ground], errors="coerce")
    df_sensor[time_col_sensor] = pd.to_datetime(df_sensor[time_col_sensor], errors="coerce")

    df_ground["temperature"] = pd.to_numeric(df_ground["temperature"], errors="coerce")
    df_ground["humidity"] = pd.to_numeric(df_ground["humidity"], errors="coerce")
    df_sensor["temperature"] = pd.to_numeric(df_sensor["temperature"], errors="coerce")
    df_sensor["humidity"] = pd.to_numeric(df_sensor["humidity"], errors="coerce")

    df_ground = df_ground.dropna(subset=[time_col_ground, "temperature", "humidity"]).sort_values(time_col_ground)
    df_sensor = df_sensor.dropna(subset=[time_col_sensor, "temperature", "humidity"])

    # Bin sensor readings to 1-min means first (several BEAM sensors per minute)
    df_sensor["time_group"] = df_sensor[time_col_sensor].dt.floor("1min")
    sensor_binned = (
        df_sensor.groupby("time_group")
        .agg(sensor_temp=("temperature", "mean"), sensor_humid=("humidity", "mean"))
        .reset_index()
        .sort_values("time_group")
    )

    ground_renamed = df_ground[[time_col_ground, "temperature", "humidity"]].rename(
        columns={time_col_ground: "time_group", "temperature": "ref_temp", "humidity": "ref_humid"}
    )

    # Diagnostic: check whether the two time ranges actually overlap at all
    print(f"   [{label}] sensor range:    {sensor_binned['time_group'].min()} → {sensor_binned['time_group'].max()}")
    print(f"   [{label}] reference range: {ground_renamed['time_group'].min()} → {ground_renamed['time_group'].max()}")

    # merge_asof requires both frames sorted by the merge key and no NaT
    merged = pd.merge_asof(
        sensor_binned,
        ground_renamed,
        on="time_group",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )

    merged = merged.dropna(subset=["ref_temp", "ref_humid"]).reset_index(drop=True)

    print(f"   [{label}] matched rows within {tolerance} tolerance: {len(merged)}")

    return merged


############################
# 2. Compute bias/RMSE/MAE for one lag-time cutoff
############################

def compute_metrics_for_lag(merged, lag_minutes):
    t0 = merged["time_group"].min()
    cutoff = t0 + pd.Timedelta(minutes=lag_minutes)
    sub = merged[merged["time_group"] >= cutoff]

    if sub.empty:
        return None

    temp_resid = sub["sensor_temp"] - sub["ref_temp"]
    humid_resid = sub["sensor_humid"] - sub["ref_humid"]

    return {
        "lag_minutes": lag_minutes,
        "n": len(sub),
        "temp_bias": temp_resid.mean(),
        "temp_rmse": np.sqrt((temp_resid ** 2).mean()),
        "temp_mae": temp_resid.abs().mean(),
        "temp_sd": temp_resid.std(),
        "humid_bias": humid_resid.mean(),
        "humid_rmse": np.sqrt((humid_resid ** 2).mean()),
        "humid_mae": humid_resid.abs().mean(),
        "humid_sd": humid_resid.std(),
    }


############################
# 3. Run the sensitivity analysis across LAG_TIMES for one condition
############################

def run_sensitivity_analysis(df_ground, df_sensor, time_col_ground, time_col_sensor, condition_label,
                              tolerance="30s"):
    merged = merge_ground_and_sensor(
        df_ground, df_sensor, time_col_ground, time_col_sensor,
        tolerance=tolerance, label=condition_label
    )

    if merged.empty:
        print(f"⚠️ {condition_label}: no matched timestamps at all within {tolerance} tolerance. "
              f"Check the printed time ranges above — if they don't overlap, the logger and "
              f"BEAM sensor clocks are likely offset by more than the tolerance. "
              f"Try increasing `tolerance` (e.g. '2min') to check.")
        return pd.DataFrame()

    results = []
    for lag in LAG_TIMES:
        res = compute_metrics_for_lag(merged, lag)
        if res is not None:
            res["condition"] = condition_label
            results.append(res)
        else:
            print(f"⚠️ No data left for {condition_label} at lag={lag} min")

    return pd.DataFrame(results)


sens_K1 = run_sensitivity_analysis(
    df_groundTruth_K1, df_K1_filtered,
    "Local (GMT +01:00)", "sensortime",
    "Condition 1 (25°C, 40% RH)"
)

sens_K2 = run_sensitivity_analysis(
    df_groundTruth_K2, df_K2_filtered,
    "Local (GMT +01:00)", "sensortime",
    "Condition 2 (40°C, 75% RH)"
)

sens_all = pd.concat([sens_K1, sens_K2], ignore_index=True)

if sens_all.empty:
    print("\n⚠️ sens_all is empty — no matched timestamps for either condition. "
          "See the [Condition ...] range/matched-rows diagnostics printed above; "
          "this points to a clock offset between the reference logger and the BEAM sensors "
          "larger than the merge tolerance.")
else:
    print("\nSensitivity analysis across lag-time cutoffs:")
    print(
        sens_all[
            ["condition", "lag_minutes", "n", "temp_bias", "temp_rmse", "humid_bias", "humid_rmse"]
        ].round(3).to_string(index=False)
    )

    # Optional: export as a table for the manuscript / reviewer response
    try:
        sens_all.round(3).to_csv(
            BASE_DIR / "sensitivity_analysis_lag_time.csv", sep=";", index=False
        )
    except PermissionError:
        print("⚠️ Could not write sensitivity_analysis_lag_time.csv (file open elsewhere or no "
              "write permission) — continuing without export.")


############################
# 4. Stabilization curve: when does the chamber itself reach setpoint?
############################

def plot_stabilization_curve(df_ground, time_col_ground, title, xlim_minutes=120):
    df = df_ground.copy()
    df[time_col_ground] = pd.to_datetime(df[time_col_ground], errors="coerce")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
    df = df.dropna(subset=[time_col_ground, "temperature", "humidity"]).sort_values(time_col_ground)

    t0 = df[time_col_ground].min()
    minutes = (df[time_col_ground] - t0).dt.total_seconds() / 60

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 300,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    fig.subplots_adjust(hspace=0.25)

    # -------------------------
    # Temperature
    # -------------------------
    ax1.plot(minutes, df["temperature"], color=BLACK, linewidth=0.7)
    for lag in LAG_TIMES:
        if lag > 0:
            ax1.axvline(lag, color=GOLDENROD, linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_xlim(0, xlim_minutes)
    ax1.set_ylabel("Temperature (°C)")
    ax1.text(0.01, 0.95, "a", transform=ax1.transAxes, fontweight="bold", va="top")

    # -------------------------
    # Humidity
    # -------------------------
    ax2.plot(minutes, df["humidity"], color=BLACK, linewidth=0.7)
    for lag in LAG_TIMES:
        if lag > 0:
            ax2.axvline(lag, color=SLATEGREY, linestyle="--", linewidth=0.8, alpha=0.7)
            ax2.text(lag, ax2.get_ylim()[1], f"{lag} min", rotation=90,
                     va="top", ha="right", fontsize=7, color=SLATEGREY)
    ax2.set_xlim(0, xlim_minutes)
    ax2.set_xlabel("Time since chamber start (minutes)")
    ax2.set_ylabel("Relative humidity (%)")
    ax2.text(0.01, 0.95, "b", transform=ax2.transAxes, fontweight="bold", va="top")

    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.show()


plot_stabilization_curve(df_groundTruth_K1, "Local (GMT +01:00)", "Condition 1 (25°C, 40% RH) — chamber stabilization")
plot_stabilization_curve(df_groundTruth_K2, "Local (GMT +01:00)", "Condition 2 (40°C, 75% RH) — chamber stabilization")