"""A/B Testing: Comparing Maximum Bidding and Average Bidding.

This analysis follows the four tasks in the provided case study.
Run from the repository folder: python AB_Testing_Analysis.py
"""
from pathlib import Path
import pandas as pd
from scipy import stats

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:.2f}")

# BUSINESS PROBLEM
# An e-commerce company ran a one-month A/B test comparing Facebook's
# existing Maximum Bidding strategy with the new Average Bidding strategy.
# Purchase is the primary success metric.

# DATASET
# Impression: number of ad impressions
# Click: number of ad clicks
# Purchase: number of purchases attributed to clicked ads
# Earning: revenue generated from purchases
# Control Group: Maximum Bidding; Test Group: Average Bidding.

# TASK 1: DATA PREPARATION AND EXPLORATORY ANALYSIS
# Step 1: Load the control and test groups from separate Excel sheets.
data_path = Path(__file__).resolve().parent / "ab_testing.xlsx"
df_control = pd.read_excel(data_path, sheet_name="Control Group")
df_test = pd.read_excel(data_path, sheet_name="Test Group")

print("\nControl group — first five rows:")
print(df_control.head())
print("\nTest group — first five rows:")
print(df_test.head())

# Step 2: Examine data types, missing values, and descriptive statistics.
print("\nControl group information:")
df_control.info()
print("\nTest group information:")
df_test.info()
print("\nMissing values — control:")
print(df_control.isna().sum())
print("\nMissing values — test:")
print(df_test.isna().sum())
print("\nDescriptive statistics — control:")
print(df_control.describe().T)
print("\nDescriptive statistics — test:")
print(df_test.describe().T)

# Step 3: Label groups and concatenate them vertically.
df_control["Group"] = "Control"
df_test["Group"] = "Test"
df = pd.concat([df_control, df_test], ignore_index=True)
print("\nCombined data — first and last five rows:")
print(df.head())
print(df.tail())
print("Combined shape:", df.shape)

# TASK 2: DEFINE THE A/B TEST HYPOTHESES
# Step 1: H0: The population mean Purchase is equal in the two groups.
#         H1: The population mean Purchase differs between the two groups.
# Step 2: Compare observed mean Purchase values.
means = df.groupby("Group")["Purchase"].mean()
print("\nMean Purchase by group:")
print(means)
relative_difference = (means["Test"] / means["Control"] - 1) * 100
print(f"Observed relative difference (Test vs Control): {relative_difference:.2f}%")
# A higher sample mean alone does not establish statistical significance.

# TASK 3: PERFORM HYPOTHESIS TESTING
alpha = 0.05
# Step 1a: Shapiro-Wilk normality tests, performed separately for each group.
# H0: Data are consistent with a normal distribution.
# H1: Data are not consistent with a normal distribution.
control_w, control_p = stats.shapiro(df_control["Purchase"])
test_w, test_p = stats.shapiro(df_test["Purchase"])
print(f"\nShapiro-Wilk — Control: W={control_w:.4f}, p={control_p:.4f}")
print(f"Shapiro-Wilk — Test:    W={test_w:.4f}, p={test_p:.4f}")
# Both p-values exceed 0.05, so normality is not rejected for either group.
# Failure to reject H0 does not prove that the data are normally distributed.

# Step 1b: Levene's test for equality of variances.
# H0: The population variances are equal.
# H1: The population variances differ.
levene_stat, levene_p = stats.levene(
    df_control["Purchase"], df_test["Purchase"]
)
print(f"Levene's test: statistic={levene_stat:.4f}, p={levene_p:.4f}")
# The p-value exceeds 0.05, so equality of variances is not rejected.

# Step 2: Select the independent-samples Student's t-test.
# Neither normality nor equal variances was rejected at alpha = 0.05.
# This case study assumes the observations in the two groups are independent.
# If equal variances were not plausible, Welch's t-test would be preferable.

# Step 3: Compare Purchase means with a two-sided Student's t-test.
t_stat, t_p = stats.ttest_ind(
    df_control["Purchase"], df_test["Purchase"], equal_var=True
)
print(f"Student's t-test: t={t_stat:.4f}, p={t_p:.4f}")
if t_p < alpha:
    print("Decision: Reject H0; evidence of a difference in mean Purchase.")
else:
    print("Decision: Do not reject H0; no statistically significant difference detected.")

# TASK 4: INTERPRETATION AND BUSINESS RECOMMENDATION
# Step 1: The Student's t-test was selected because Shapiro-Wilk did not
# reject normality in either group and Levene's test did not reject equal
# variances. The test compares mean Purchase between independent groups.
#
# Step 2: The Average Bidding group has a higher observed Purchase mean
# (about 582.11 vs 550.89, approximately +5.67%), but the two-sided t-test
# p-value is approximately 0.3493, above the 0.05 significance threshold.
# We therefore cannot conclude that the population mean Purchase differs.
# This does NOT prove the strategies are equivalent.
# Recommendation: Do not switch strategies based on this test alone.
# Plan a follow-up experiment with an appropriate sample-size/power analysis,
# and evaluate ad costs, revenue, and business impact before deciding.
