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