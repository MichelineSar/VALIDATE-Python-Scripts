import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
import seaborn as sns




###########################
## Load and preprocess the data
###########################

# Read only the specified columns from the CSV file
df = pd.read_csv(r'example_data.csv')

# Convert to datetime (day first because your format is dd.mm.yyyy)
df["Diary_Time"] = pd.to_datetime(df["Diary_Time"], format="%d.%m.%Y %H:%M", errors="coerce")
df["BEAM_Time"] = pd.to_datetime(df["BEAM_Time"], format="%d.%m.%Y %H:%M", errors="coerce")

# Split into separate columns
df["Diary_Hour"] = df["Diary_Time"].dt.time

df["BEAM_Hour"] = df["BEAM_Time"].dt.time

# Print the first 10 rows
print(df[["P_Code", "Diary_Time", "BEAM_Time", "Diary_Hour", "BEAM_Hour"]].head(10))


# ###########################
# ## Data Inspection
# ###########################

# # Compute absolute time difference in minutes
# df["Time_Diff"] = (df["Diary_Time"] - df["BEAM_Time"]).abs() / np.timedelta64(1, "m")

# # Create true positive column: True if ≤ 5 minutes, False otherwise
# df["TP"] = df["Time_Diff"] <= 5

# # Create true negative column: True if both times are NaT, False otherwise
# df["TN"] = df["Diary_Hour"].isna() & df["BEAM_Hour"].isna()

# # Create false positive column: True if Diary_Hour is NaT and BEAM_Hour is not NaT
# df["FP"] = df["Diary_Hour"].isna() & ~df["BEAM_Hour"].isna()

# # Create false negative column: True if BEAM_Hour is NaT and Diary_Hour is not NaT
# df["FN"] = df["BEAM_Hour"].isna() & ~df["Diary_Hour"].isna()

# print(df.head(10))

# # Count rows where difference is > 6 minutes
# count_over_6 = (df["Time_Diff"] > 1).sum()

# TP, TN, FP, FN = df["TP"].sum(), df["TN"].sum(), df["FP"].sum(), df["FN"].sum()


# print("\n Number of rows with time difference > 6 minutes:", count_over_6)
# print("True Positives (TP):", TP)
# print("True Negatives (TN):", TN)
# print("False Positives (FP):", FP)
# print("False Negatives (FN):", FN)

# print("\n**************************\n")

###########################
## Data Inspection
###########################

def classify_events(df):
    df = df.sort_values("Diary_Time").reset_index(drop=True)

    # ---- 1. True Positives ----
    df["TP"] = (
        df["BEAM_Time"].notna() &
        ((df["BEAM_Time"] - df["Diary_Time"]).abs() <= pd.Timedelta("5min"))
    )

    # ---- 2. False Negatives ----
    df["FN"] = ~df["TP"]  # no beam within ±5min

    # ---- 3. Define diary intervals ----
    intervals = []
    for i in range(len(df) - 1):
        start = df.loc[i, "Diary_Time"] + pd.Timedelta("5min")
        end   = df.loc[i+1, "Diary_Time"] - pd.Timedelta("5min")
        intervals.append((start, end))

    # ---- 4. Check each BEAM for FP ----
    def in_intervals(beam_time, intervals):
        if pd.isna(beam_time):
            return False
        return any(start < beam_time < end for start, end in intervals)

    df["FP"] = df["BEAM_Time"].apply(lambda x: in_intervals(x, intervals))

    # ---- 5. True Negatives ----
    # TN if no FP in that row (and no TP/FN applies)
    df["TN"] = (~df["FP"])

    return df


# --- Apply per participant ---
df_clean = df.dropna(subset=["Diary_Time", "BEAM_Time"], how="all")


df_grouped = df_clean.groupby("P_Code").apply(classify_events, include_groups=False)

# --- Counts per participant ---
counts_per_participant = df_grouped.groupby("P_Code")[["TP", "TN", "FP", "FN"]].sum()

# --- Global totals ---
TP = df_grouped["TP"].sum()
TN = df_grouped["TN"].sum()
FP = df_grouped["FP"].sum()
FN = df_grouped["FN"].sum()

# --- Counts per participant + total diary entries (including NaT) ---
counts_per_participant = (
    df_grouped.groupby("P_Code")
    .agg({
        "TP": "sum",
        "TN": "sum",
        "FP": "sum",
        "FN": "sum"
    })
)


# --- Global totals ---
TP_total = df_grouped["TP"].sum()
TN_total = df_grouped["TN"].sum()
FP_total = df_grouped["FP"].sum()
FN_total = df_grouped["FN"].sum()
total_diary_entries = len(df_grouped)   # counts all rows in the cleaned df

print("Totals:")
print("TP:", TP_total, "TN:", TN_total, "FP:", FP_total, "FN:", FN_total)
print("Total diary entries:", total_diary_entries)


#########################
## Analysis
##########################

# Avoid ZeroDivisionError with .where or simple checks
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan
specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan
ppv = TP / (TP + FP) if (TP + FP) > 0 else np.nan
npv = TN / (TN + FN) if (TN + FN) > 0 else np.nan


# Function to calculate 95% confidence intervals for proportions
def proportion_ci(successes, total, alpha=0.05):
    if total == 0:
        return (np.nan, np.nan)
    ci_low, ci_upp = sm.stats.proportion_confint(successes, total, alpha=alpha, method="wilson")
    return ci_low, ci_upp

sens_ci = proportion_ci(TP, TP + FN)
spec_ci = proportion_ci(TN, TN + FP)
ppv_ci  = proportion_ci(TP, TP + FP)
npv_ci  = proportion_ci(TN, TN + FN)

print(f"Sensitivity: {sensitivity:.2f} (95% CI {sens_ci[0]:.2f}–{sens_ci[1]:.2f})")
print(f"Specificity: {specificity:.2f} (95% CI {spec_ci[0]:.2f}–{spec_ci[1]:.2f})")
print(f"PPV:         {ppv:.2f} (95% CI {ppv_ci[0]:.2f}–{ppv_ci[1]:.2f})")
print(f"NPV:         {npv:.2f} (95% CI {npv_ci[0]:.2f}–{npv_ci[1]:.2f})")

print("\n**************************\n")

###########################
# Analysis per Patient
###########################

def calculate_metrics(group):
    TP = group["TP"].sum()
    TN = group["TN"].sum()
    FP = group["FP"].sum()
    FN = group["FN"].sum()

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    ppv = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    npv = TN / (TN + FN) if (TN + FN) > 0 else np.nan

     # confidence intervals
    sens_ci = proportion_ci(TP, TP + FN)
    spec_ci = proportion_ci(TN, TN + FP)
    ppv_ci  = proportion_ci(TP, TP + FP)
    npv_ci  = proportion_ci(TN, TN + FN)

    return pd.Series({
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Sensitivity": sensitivity,
        "Sensitivity_CI": f"[{sens_ci[0]:.2f}, {sens_ci[1]:.2f}]",
        "Specificity": specificity,
        "Specificity_CI": f"[{spec_ci[0]:.2f}, {spec_ci[1]:.2f}]",
        "PPV": ppv,
        "PPV_CI": f"[{ppv_ci[0]:.2f}, {ppv_ci[1]:.2f}]",
        "NPV": npv,
        "NPV_CI": f"[{npv_ci[0]:.2f}, {npv_ci[1]:.2f}]"
    })

metrics_per_id = df_grouped.groupby("P_Code").apply(calculate_metrics, include_groups=False).reset_index()
print(metrics_per_id)

print("\n**************************\n")

###########################
# Bland-Altman Plot
###########################

# 1) S1 and S2 per participant
counts = (
    df_clean.groupby("P_Code")
    .agg(
        S1_diary=("Diary_Time", lambda s: s.notna().sum()),
        S2_beam =("BEAM_Time",  lambda s: s.notna().sum())
    )
    .reset_index()
)

# 2) Bland–Altman transforms
counts["Mean"] = (counts["S1_diary"] + counts["S2_beam"]) / 2.0
counts["Diff"] = counts["S1_diary"] - counts["S2_beam"]

# 3) BA stats across participants
bias = counts["Diff"].mean()
sd   = counts["Diff"].std(ddof=1)
loa_upper = bias + 1.96 * sd
loa_lower = bias - 1.96 * sd

# 4) Plot
plt.figure(figsize=(8,6))
plt.scatter(counts["Mean"], counts["Diff"])

# annotate each point with P_Code
for _, r in counts.iterrows():
    plt.annotate(r["P_Code"], (r["Mean"], r["Diff"]), xytext=(4,4), textcoords="offset points")

plt.axhline(bias,       linestyle="-",  label=f"Bias = {bias:.2f}")
plt.axhline(loa_upper,  linestyle="--", label=f"+1.96 SD = {loa_upper:.2f}")
plt.axhline(loa_lower,  linestyle="--", label=f"-1.96 SD = {loa_lower:.2f}")

plt.xlabel("Mean count ( (S1 + S2) / 2 )")
plt.ylabel("Difference ( S1 − S2 )")
plt.title("Bland–Altman: Diary vs BEAM counts per participant")
plt.legend()
plt.tight_layout()
plt.show()

# Optional: view the table used to plot
print(counts.sort_values("P_Code"))