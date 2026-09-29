import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

output_dir = r'u:\Home\pyvuli54\1_Projekte\5_VALIDATE\4_Data\Data_Analysis'

# ---- 1. Einlesen ----
df = pd.read_csv(r'u:\Home\pyvuli54\1_Projekte\5_VALIDATE\4_Data\Data_Analysis\Data_clean_R.csv', sep=';', usecols=['P_Code', 'Diary_Time', 'BEAM_Time'])

df["Diary_Time"] = pd.to_datetime(df["Diary_Time"], format="%d.%m.%Y %H:%M", errors="coerce")
df["BEAM_Time"] = pd.to_datetime(df["BEAM_Time"], format="%d.%m.%Y %H:%M", errors="coerce")

# ---- 2. Klassifikation pro Proband:in ----
def classify_events(df_p, tolerance="5min"):
    tol = pd.Timedelta(tolerance)
    df_p = df_p.sort_values(["Diary_Time", "BEAM_Time"], na_position="last").reset_index(drop=True)
    df_p = df_p.dropna(subset=["Diary_Time", "BEAM_Time"], how="all")

## Event-level Klassifikation
    has_diary = df_p["Diary_Time"].notna()
    has_beam  = df_p["BEAM_Time"].notna()
    diff = (df_p["BEAM_Time"] - df_p["Diary_Time"]).abs()

    df_p["TP"] = has_diary & has_beam & (diff <= tol)
    df_p["FN"] = has_diary & ~df_p["TP"]
    df_p["FP"] = ~has_diary & has_beam

## Intervall-Level-Klassifikation (TN)
    planned = df_p.loc[has_diary, "Diary_Time"].sort_values().reset_index(drop=True)
    fp_times = df_p.loc[df_p["FP"], "BEAM_Time"]

    interval_records = []
    for i in range(len(planned) - 1):
        start, end = planned[i] + tol, planned[i + 1] - tol
        if start >= end: ## Bspw. T1 = 8:00 + 5 mins, T2 = 8:03 - 5 mins => start >= end, dann skip
            continue
        fp_in_interval = fp_times[(fp_times >= start) & (fp_times <= end)] ## überprüft, ob kein FP im Intervall liegt  
        n_fp = len(fp_in_interval)
        interval_records.append({
            "interval_start": start,
            "interval_end": end,
            "n_fp_raw": n_fp,   # nur zur TN-Bestimmung (Intervall-Level), NICHT für FP-Metriken verwenden
            "TN": n_fp == 0,    # Intervall gilt als TN nur wenn KEIN FP enthalten ist
        })

    interval_df = pd.DataFrame(interval_records)
    return df_p, interval_df


# ---- 3. Metriken + Wilson-CI (pro Proband:in) ----
def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return pd.NA, pd.NA
    z = stats.norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return max(0, center - margin), min(1, center + margin)


def compute_metrics(TP, FN, FP, TN):
    sens = TP / (TP + FN) if (TP + FN) > 0 else pd.NA
    spec = TN / (TN + FP) if (TN + FP) > 0 else pd.NA
    ppv  = TP / (TP + FP) if (TP + FP) > 0 else pd.NA
    npv  = TN / (TN + FN) if (TN + FN) > 0 else pd.NA
    return sens, spec, ppv, npv


results = []
for p_code, df_p in df.groupby("P_Code"):
    events_df, interval_df = classify_events(df_p)

    TP = events_df["TP"].sum()
    FN = events_df["FN"].sum()
    TN = interval_df["TN"].sum()
    FP = events_df["FP"].sum()  

    sens, spec, ppv, npv = compute_metrics(TP, FN, FP, TN)
    sens_lo, sens_hi = wilson_ci(TP, TP + FN)
    spec_lo, spec_hi = wilson_ci(TN, TN + FP)
    ppv_lo, ppv_hi   = wilson_ci(TP, TP + FP)
    npv_lo, npv_hi   = wilson_ci(TN, TN + FN)

    results.append({
        "P_Code": p_code,
        "TP": TP, "FN": FN, "FP": FP, "TN": TN,
        "Sensitivity": sens, "Sens_CI_low": sens_lo, "Sens_CI_high": sens_hi,
        "Specificity": spec, "Spec_CI_low": spec_lo, "Spec_CI_high": spec_hi,
        "PPV": ppv, "PPV_CI_low": ppv_lo, "PPV_CI_high": ppv_hi,
        "NPV": npv, "NPV_CI_low": npv_lo, "NPV_CI_high": npv_hi,
    })

per_participant = pd.DataFrame(results)

# ---- 4. Overall = Mittelwert + Bootstrap-CI ----
def bootstrap_ci_mean(values, n_boot=2000, alpha=0.05, seed=42):
    values = pd.Series(values).dropna().to_numpy()
    n = len(values)
    if n < 2:
        return pd.NA, pd.NA
    rng = np.random.default_rng(seed)
    res = stats.bootstrap(
        (values,),
        statistic=np.mean,
        n_resamples=n_boot,
        confidence_level=1 - alpha,
        method="BCa",
        random_state=rng
    )
    return res.confidence_interval.low, res.confidence_interval.high


def overall_row_bootstrap(metric_col, n_boot=2000, seed=42):
    lo, hi = bootstrap_ci_mean(per_participant[metric_col], n_boot=n_boot, seed=seed)
    return per_participant[metric_col].mean(), lo, hi

sens_m, sens_lo, sens_hi = overall_row_bootstrap("Sensitivity")
spec_m, spec_lo, spec_hi = overall_row_bootstrap("Specificity")
ppv_m, ppv_lo, ppv_hi = overall_row_bootstrap("PPV")
npv_m, npv_lo, npv_hi = overall_row_bootstrap("NPV")

overall = pd.DataFrame([{
    "P_Code": "OVERALL",
    "TP": per_participant["TP"].sum(),
    "FN": per_participant["FN"].sum(),
    "FP": per_participant["FP"].sum(),
    "TN": per_participant["TN"].sum(),
    "Sensitivity": sens_m, "Sens_CI_low": sens_lo, "Sens_CI_high": sens_hi,
    "Specificity": spec_m, "Spec_CI_low": spec_lo, "Spec_CI_high": spec_hi,
    "PPV": ppv_m, "PPV_CI_low": ppv_lo, "PPV_CI_high": ppv_hi,
    "NPV": npv_m, "NPV_CI_low": npv_lo, "NPV_CI_high": npv_hi,
}])

summary = pd.concat([per_participant, overall], ignore_index=True)
print(summary.to_string(index=False))
csv_path = os.path.join(output_dir, "validation_metrics.csv")
summary.to_csv(csv_path, sep=";", index=False, decimal=",")
print(f"CSV-Datei gespeichert: {csv_path}")



# ---- 5. Forest Plots ----
def forest_plot(ax, data, metric, ci_low_col, ci_high_col, title, panel_label=None):
    plot_data = data.copy()
    plot_data = plot_data.dropna(subset=[metric, ci_low_col, ci_high_col])
    y_pos = np.arange(len(plot_data))[::-1]

    is_overall = plot_data["P_Code"].str.startswith("OVERALL")
    colors = ["#4D343E" if o else "#9897AC" for o in is_overall]

    err_low = np.clip((plot_data[metric] - plot_data[ci_low_col]).astype(float), 0, None)
    err_high = np.clip((plot_data[ci_high_col] - plot_data[metric]).astype(float), 0, None)

    ax.errorbar(
        plot_data[metric], y_pos,
        xerr=[err_low, err_high],
        fmt="o", color="black", ecolor="gray", elinewidth=1, capsize=3, markersize=0
    )
    ax.scatter(plot_data[metric], y_pos, c=colors, s=60, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_data["P_Code"], fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.axvline(plot_data.loc[is_overall, metric].values[0], color="#4D343E", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel(f"{metric} (proportion, 95% CI)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="x", alpha=0.3)

    if panel_label is not None:
        ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes,
                 fontsize=14, fontweight="bold", va="top", ha="right")


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

forest_plot(axes[0, 0], summary, "Sensitivity", "Sens_CI_low", "Sens_CI_high",
            "Sensitivity", panel_label="A")
forest_plot(axes[0, 1], summary, "Specificity", "Spec_CI_low", "Spec_CI_high",
            "Specificity", panel_label="B")
forest_plot(axes[1, 0], summary, "PPV", "PPV_CI_low", "PPV_CI_high",
            "Positive Predictive Value (PPV)", panel_label="C")
forest_plot(axes[1, 1], summary, "NPV", "NPV_CI_low", "NPV_CI_high",
            "Negative Predictive Value (NPV)", panel_label="D")

plt.tight_layout()
output_dir = r'u:\Home\pyvuli54\1_Projekte\5_VALIDATE\4_Data\Data_Analysis'
plt.savefig(os.path.join(output_dir, "forest_plots.png"), dpi=300, bbox_inches="tight")
plt.show()

# ---- 6. Sensitivitätsanalyse (Toleranzintervalle: 1, 2, 3, 5, 10, 30 min) ----
tolerances = [1, 2, 3, 5, 10, 30]
sensitivity_results = []

for tol_min in tolerances:
    tol_str = f"{tol_min}min"
    
    # 6.1 Metriken pro Proband:in berechnen für aktuelle Toleranz
    p_results = []
    for p_code, df_p in df.groupby("P_Code"):
        events_df, interval_df = classify_events(df_p, tolerance=tol_str)

        TP = events_df["TP"].sum()
        FN = events_df["FN"].sum()
        TN = interval_df["TN"].sum()
        FP = events_df["FP"].sum()

        sens, spec, ppv, npv = compute_metrics(TP, FN, FP, TN)
        p_results.append({
            "P_Code": p_code,
            "Sensitivity": sens,
            "Specificity": spec,
            "PPV": ppv,
            "NPV": npv
        })

    df_p_results = pd.DataFrame(p_results)

    # 6.2 Overall-Mittelwerte & BCa-Bootstrap CIs berechnen
    for metric in ["Sensitivity", "Specificity", "PPV", "NPV"]:
        mean_val = df_p_results[metric].mean()
        lo, hi = bootstrap_ci_mean(df_p_results[metric], n_boot=2000, seed=42)
        
        sensitivity_results.append({
            "Tolerance_min": tol_min,
            "Metric": metric,
            "Mean": mean_val,
            "CI_low": lo,
            "CI_high": hi
        })

sens_df = pd.DataFrame(sensitivity_results)

# Tabelle speichern
sens_csv_path = os.path.join(output_dir, "sensitivity_analysis_metrics.csv")
sens_df.to_csv(sens_csv_path, sep=";", index=False, decimal=",")
print(f"Sensitivitätsanalyse CSV gespeichert: {sens_csv_path}")


# ---- 7. Plot der Sensitivitätsanalyse ----
fig, ax = plt.subplots(figsize=(10, 6))

metrics_config = [
    ("Sensitivity", "#4D343E", "o", "-"),
    ("Specificity", "#8F8AA3", "s", "-"),
    ("PPV", "#807076", "^", "-"),
    ("NPV", "#B38195", "d", "-")
]

for metric, color, marker, linestyle in metrics_config:
    sub = sens_df[sens_df["Metric"] == metric].sort_values("Tolerance_min")
    
    x = sub["Tolerance_min"].values
    y = sub["Mean"].values
    y_lo = sub["CI_low"].values
    y_hi = sub["CI_high"].values

    err_low = np.clip(y - y_lo, 0, None)
    err_high = np.clip(y_hi - y, 0, None)

    # Verläufe mit Fehlerbalken zeichnen
    ax.errorbar(
        x, y, yerr=[err_low, err_high],
        label=metric, color=color, fmt=marker + linestyle,
        capsize=4, elinewidth=1.2, markersize=7, linewidth=1.8
    )

ax.set_xscale("log")  # Logarithmische X-Achse für bessere Lesbarkeit (1 bis 30 min)
ax.set_xticks(tolerances)
ax.set_xticklabels([f"{t}" for t in tolerances])

ax.set_ylim(0.5, 1.02)
ax.set_xlabel("Grace Period (Minutes)", fontsize=11, fontweight="bold")
ax.set_ylabel("Overall Performance Metric (BCa 95% CI)", fontsize=11, fontweight="bold")
ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.legend(title="Metric", loc="lower right", frameon=True)

plt.tight_layout()
sens_plot_path = os.path.join(output_dir, "sensitivity_analysis_plot.png")
plt.savefig(sens_plot_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Sensitivitätsanalyse Plot gespeichert: {sens_plot_path}")