import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


###########################
# Einstellungen
###########################

FILE_PATH = r"\example_data.csv"

GRACE_PERIODS = [1, 2, 5, 10, 30]
COLOR = "#D68B0F"


###########################
# Daten laden und vorbereiten
###########################

df = pd.read_csv(
    FILE_PATH,
    sep=";",
    usecols=["P_Code", "Diary_Time", "BEAM_Time"]
)

df["Diary_Time"] = pd.to_datetime(
    df["Diary_Time"],
    format="%d.%m.%Y %H:%M",
    errors="coerce"
)

df["BEAM_Time"] = pd.to_datetime(
    df["BEAM_Time"],
    format="%d.%m.%Y %H:%M",
    errors="coerce"
)

df["Diary_Hour"] = df["Diary_Time"].dt.time
df["BEAM_Hour"] = df["BEAM_Time"].dt.time

df_clean = df.dropna(subset=["Diary_Time", "BEAM_Time"], how="all")

print("\nFirst rows:")
print(df_clean[["P_Code", "Diary_Time", "BEAM_Time", "Diary_Hour", "BEAM_Hour"]].head(10))


###########################
# Hilfsfunktionen
###########################

def proportion_ci(successes, total, alpha=0.05):
    if total == 0:
        return np.nan, np.nan

    ci_low, ci_high = sm.stats.proportion_confint(
        successes,
        total,
        alpha=alpha,
        method="wilson"
    )

    return ci_low, ci_high


def classify_events(group, grace_minutes):
    group = group.sort_values("Diary_Time").reset_index(drop=True)

    grace = pd.Timedelta(minutes=grace_minutes)

    # ---- 1. True Positives ----
    group["TP"] = (
        group["Diary_Time"].notna()
        & group["BEAM_Time"].notna()
        & ((group["BEAM_Time"] - group["Diary_Time"]).abs() <= grace)
    )

    # ---- 2. False Negatives ----
    # Diary vorhanden, aber kein BEAM innerhalb der Grace Period
    group["FN"] = (
        group["Diary_Time"].notna()
        & ~group["TP"]
    )

    # ---- 3. Diary intervals definieren ----
    diary_times = (
        group["Diary_Time"]
        .dropna()
        .sort_values()
        .reset_index(drop=True)
    )

    intervals = []

    for i in range(len(diary_times) - 1):
        start = diary_times.iloc[i] + grace
        end = diary_times.iloc[i + 1] - grace

        if start < end:
            intervals.append((start, end))

    # ---- 4. False Positives ----
    # BEAM liegt zwischen Diary-Ereignissen, aber außerhalb der Grace Period
    def in_intervals(beam_time):
        if pd.isna(beam_time):
            return False

        return any(start < beam_time < end for start, end in intervals)

    group["FP"] = group["BEAM_Time"].apply(in_intervals)

    # ---- 5. True Negatives ----
    # Analog zu deiner bisherigen Logik:
    # Kein False Positive
    group["TN"] = ~group["FP"]

    return group


def calculate_metrics(group):
    TP = group["TP"].sum()
    TN = group["TN"].sum()
    FP = group["FP"].sum()
    FN = group["FN"].sum()

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan

    sens_ci = proportion_ci(TP, TP + FN)
    spec_ci = proportion_ci(TN, TN + FP)

    return pd.Series({
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Sensitivity": sensitivity,
        "Sensitivity_CI_Low": sens_ci[0],
        "Sensitivity_CI_High": sens_ci[1],
        "Specificity": specificity,
        "Specificity_CI_Low": spec_ci[0],
        "Specificity_CI_High": spec_ci[1]
    })


def mean_ci(values):
    values = pd.Series(values).dropna()
    n = len(values)

    if n == 0:
        return np.nan, np.nan, np.nan

    mean = values.mean()

    if n == 1:
        return mean, np.nan, np.nan

    se = values.std(ddof=1) / np.sqrt(n)

    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se

    return mean, ci_low, ci_high


###########################
# Sensitivity Analysis
###########################

summary_results = []
per_patient_all = []

for grace in GRACE_PERIODS:

    print(f"\nRunning analysis for grace period: {grace} minutes")

    df_grouped = (
        df_clean
        .groupby("P_Code", group_keys=False)
        .apply(lambda g: classify_events(g, grace))
    )

    metrics_per_patient = (
        df_grouped
        .groupby("P_Code")
        .apply(calculate_metrics)
        .reset_index()
    )

    metrics_per_patient["Grace_Period"] = grace
    per_patient_all.append(metrics_per_patient)

    sens_mean, sens_ci_low, sens_ci_high = mean_ci(metrics_per_patient["Sensitivity"])
    spec_mean, spec_ci_low, spec_ci_high = mean_ci(metrics_per_patient["Specificity"])

    summary_results.append({
        "Grace_Period": grace,
        "Sensitivity_Mean": sens_mean,
        "Sensitivity_CI_Low": sens_ci_low,
        "Sensitivity_CI_High": sens_ci_high,
        "Specificity_Mean": spec_mean,
        "Specificity_CI_Low": spec_ci_low,
        "Specificity_CI_High": spec_ci_high
    })


summary_results = pd.DataFrame(summary_results)
per_patient_all = pd.concat(per_patient_all, ignore_index=True)


###########################
# Ergebnisse anzeigen
###########################

print("\nSummary results:")
print(summary_results)

print("\nPer-patient results:")
print(per_patient_all)


###########################
# Plot aus berechneten summary_results (optimiert)
###########################

from matplotlib.lines import Line2D

COLOR = "#D68B0F"

summary_results = summary_results.sort_values("Grace_Period").reset_index(drop=True)

sens_yerr = [
    summary_results["Sensitivity_Mean"] - summary_results["Sensitivity_CI_Low"],
    summary_results["Sensitivity_CI_High"] - summary_results["Sensitivity_Mean"]
]

spec_yerr = [
    summary_results["Specificity_Mean"] - summary_results["Specificity_CI_Low"],
    summary_results["Specificity_CI_High"] - summary_results["Specificity_Mean"]
]

# Etwas breiter, damit Legende Platz hat
fig, ax = plt.subplots(figsize=(8.5, 5))

# Specificity
ax.errorbar(
    summary_results["Grace_Period"],
    summary_results["Specificity_Mean"],
    yerr=spec_yerr,
    fmt="o",
    linestyle="-",
    linewidth=1.8,
    color=COLOR,
    ecolor=COLOR,
    elinewidth=1.5,
    capsize=6,
    capthick=1.5,
    markersize=9,
    label="_nolegend_"
)

# Sensitivity
ax.errorbar(
    summary_results["Grace_Period"],
    summary_results["Sensitivity_Mean"],
    yerr=sens_yerr,
    fmt="^",
    linestyle="-",
    linewidth=1.8,
    color=COLOR,
    ecolor=COLOR,
    elinewidth=1.5,
    capsize=6,
    capthick=1.5,
    markersize=9,
    label="_nolegend_"
)

ax.set_xlabel("Grace period (lag time; minutes)", fontsize=14)
ax.set_ylabel("Sensitivity / Specificity", fontsize=14)

ax.set_xticks(summary_results["Grace_Period"])
ax.set_ylim(0.60, 1.02)

ax.tick_params(axis="both", labelsize=12)
ax.grid(True, alpha=0.3)

# Custom legend: nur Linie + Marker, keine CI
legend_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="-",
        color=COLOR,
        markersize=9,
        linewidth=1.8,
        label="Specificity\n(mean ± 95% CI)"   # 🔥 Zeilenumbruch
    ),
    Line2D(
        [0], [0],
        marker="^",
        linestyle="-",
        color=COLOR,
        markersize=9,
        linewidth=1.8,
        label="Sensitivity\n(mean ± 95% CI)"   # 🔥 Zeilenumbruch
    )
]

ax.legend(
    handles=legend_handles,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
    fontsize=12
)

# Genug Platz rechts für Legende
fig.subplots_adjust(right=0.72)

plt.show()