# %% [markdown]
# # AI-Powered Credit Risk Intelligence Platform - Exploratory Data Analysis
#
# **Dataset analysed:** `application_train.csv` (Home Credit Default Risk),
# plus supplementary tables (`bureau.csv`, `previous_application.csv`,
# `installments_payments.csv`) in the final section.

# %% [markdown]
# ## 1. Introduction & Business Understanding
#
# **Problem Statement:** Financial institutions face the challenge of
# identifying applicants who may fail to repay their loans. Incorrect
# credit decisions can lead to financial losses, while overly strict
# lending decisions may reject potentially reliable customers.
#
# **Objective:** I'm exploring historical loan application data to
# identify patterns associated with loan default, understand the quality
# and characteristics of the dataset, and generate insights that support
# building a credit risk prediction model.
#
# **Target variable:**
# - `TARGET = 0` -> the applicant did not experience payment difficulties
# - `TARGET = 1` -> the applicant experienced payment difficulties/default
#
# The model I build later will learn from these historical applicants to
# predict a probability of default for new applicants.

# %% [markdown]
# ## 2. Import Libraries
#
# I'm importing what I need for data manipulation, numerical operations,
# visualization, and statistics.

# %%
import os
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)

SCREENSHOT_DIR = Path("../documents/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 3. Load the Dataset
#
# `application_train.csv` contains historical loan applications and the
# `TARGET` label - this is the core table for everything in this notebook.

# %%
DATA_PATH = "../data/application_train.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# %% [markdown]
# ## 4. Dataset Overview
#
# I want to understand: number of applicants, number of features, dataset
# structure, data types, and what a few sample rows actually look like.

# %%
print("Dataset Shape:")
print(df.shape)

print("\nFirst Five Rows:")
df.head()

# %%
print("Random Sample:")
df.sample(5, random_state=42)

# %% [markdown]
# ### Data Types

# %%
print(df.dtypes.value_counts())

print("\nDetailed Dataset Information:")
df.info(verbose=False)

# %% [markdown]
# **My takeaway:** the dataset has {shape} rows and {cols} columns, mixing
# numerical variables (financial amounts, day-based durations) with
# categorical variables (demographics, employment, housing). I'll fill in
# the exact counts once I run this against the real file. This mix is why
# I categorize features by business meaning in Section 7, rather than
# treating all 122+ columns as one undifferentiated block.

# %% [markdown]
# ## 5. Target Variable Analysis ⭐
#
# Understanding: number of defaults, number of non-defaults, default
# percentage, and class imbalance.

# %%
target_counts = df["TARGET"].value_counts()
target_percentage = df["TARGET"].value_counts(normalize=True) * 100

print("Target Counts:")
print(target_counts)

print("\nTarget Percentage:")
print(target_percentage)

# %%
plt.figure(figsize=(8, 6))

ax = sns.countplot(data=df, x="TARGET")

plt.title("Loan Default Distribution")
plt.xlabel("Target")
plt.ylabel("Number of Applicants")
plt.xticks([0, 1], ["No Default (0)", "Default (1)"])

total = len(df)
for p in ax.patches:
    percentage = p.get_height() / total * 100
    ax.annotate(
        f"{percentage:.2f}%",
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center", va="bottom"
    )

plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "01_target_distribution.png", dpi=120)
plt.show()

# %% [markdown]
# **My takeaway - Class Imbalance:** if roughly 8% of applicants default,
# the dataset is imbalanced - default cases represent a significantly
# smaller share than non-default cases. This means accuracy alone won't be
# an appropriate model evaluation metric later; I'll need ROC-AUC and
# PR-AUC, and I'll need to handle the imbalance explicitly when training
# (e.g. class weighting) rather than training on raw class proportions.

# %% [markdown]
# ## 6. Data Quality Analysis ⭐
#
# I'm checking missing values, duplicates, and data types here - this
# directly decides my cleaning/imputation strategy in the next phase.

# %% [markdown]
# ### 6.1 Missing Value Analysis

# %%
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": missing_values.values,
    "Missing Percentage": missing_percentage.values
})

missing_df = missing_df.sort_values("Missing Percentage", ascending=False)

missing_df.head(20)

# %% [markdown]
# ### 6.2 Categorize Missing Values
#
# Raw percentages are harder to reason about than buckets, so I'm grouping
# every column into a missingness tier.

# %%
def missing_category(value):
    if value == 0:
        return "No Missing Values"
    elif value <= 10:
        return "Low Missingness (0-10%)"
    elif value <= 40:
        return "Moderate Missingness (10-40%)"
    elif value <= 60:
        return "High Missingness (40-60%)"
    else:
        return "Very High Missingness (>60%)"


missing_df["Missing Category"] = missing_df["Missing Percentage"].apply(missing_category)

missing_df.head(30)

# %%
top_missing = missing_df[missing_df["Missing Percentage"] > 0].head(20)

plt.figure(figsize=(12, 8))
sns.barplot(data=top_missing, x="Missing Percentage", y="Column")
plt.title("Top 20 Features with Missing Values")
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "02_missing_values.png", dpi=120)
plt.show()

# %% [markdown]
# ### 6.3 Duplicate Analysis

# %%
duplicates = df.duplicated().sum()
print("Number of Duplicate Rows:", duplicates)

# %% [markdown]
# ### 6.4 Missing Value Summary

# %%
missing_summary = (
    missing_df
    .groupby("Missing Category")
    .size()
    .reset_index(name="Number of Columns")
)

missing_summary

# %% [markdown]
# **My takeaway:** I'm noting the duplicate row count above - if it's 0,
# no dedup step is needed in preprocessing. For missing values, columns in
# the "Very High Missingness" tier (mostly apartment/building detail
# fields, e.g. `COMMONAREA_AVG`) are candidates to drop entirely rather
# than impute, since imputing >60% of a column mostly manufactures signal
# that isn't really there. Columns in "Low/Moderate" tiers I'll impute
# (median for numeric, a dedicated "Unknown" category for categorical)
# rather than drop, so I don't lose rare default cases along with them.

# %% [markdown]
# ## 7. Feature Understanding & Categorization
#
# Instead of looking at 122+ columns randomly, I'm grouping them into
# business-meaningful categories.

# %%
feature_categories = {
    "Demographics": [
        "CODE_GENDER", "DAYS_BIRTH", "CNT_CHILDREN",
        "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE"
    ],
    "Financial": [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"
    ],
    "Employment": [
        "NAME_INCOME_TYPE", "NAME_OCCUPATION_TYPE", "DAYS_EMPLOYED"
    ],
    "Family": [
        "CNT_FAM_MEMBERS", "NAME_FAMILY_STATUS", "CNT_CHILDREN"
    ],
    "Assets": [
        "FLAG_OWN_CAR", "FLAG_OWN_REALTY"
    ],
    "Housing": [
        "NAME_HOUSING_TYPE", "REGION_POPULATION_RELATIVE"
    ],
    "Credit Bureau": [
        "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"
    ],
    "Other": [
        "FLAG_MOBIL", "FLAG_EMAIL", "WEEKDAY_APPR_PROCESS_START"
    ],
}

for category, columns in feature_categories.items():
    print(f"\n{category}")
    for column in columns:
        if column in df.columns:
            print(" -", column)

# %% [markdown]
# **My takeaway:** "Credit Bureau" here refers to `AMT_REQ_CREDIT_BUREAU_*`
# columns that already live inside `application_train.csv` - these count
# how many times a credit bureau was checked about this applicant recently
# (hour/day/week/month/quarter/year). This is different from the separate
# `bureau.csv` table (the applicant's actual credit history at other
# lenders), which I analyse separately in Section 21, since it needs a
# join rather than living in this table directly.

# %% [markdown]
# ## 8. Data Anomaly Analysis ⭐
#
# `DAYS_EMPLOYED` contains a known placeholder value that needs flagging
# before it reaches the model as a fake numeric outlier.

# %%
df["DAYS_EMPLOYED"].describe()

# %%
df["DAYS_EMPLOYED"].value_counts().head()

# %% [markdown]
# **My takeaway:** the employment duration variable contains an unusually
# large value of 365243 days (~1000 years) - not a realistic employment
# duration. This is a placeholder Home Credit used for applicants who
# aren't currently employed (e.g. pensioners). I'll treat this as an
# anomaly and convert it to NaN in Section 9 rather than leaving it as a
# fake numeric value.

# %% [markdown]
# ## 9. Feature Engineering for EDA ⭐⭐⭐
#
# I'm creating business-meaningful features here to make the rest of the
# analysis more interpretable - these are candidates I'll formalize
# properly in the Phase 3 feature engineering pipeline.

# %% [markdown]
# ### 9.1 Age
# The dataset stores age as negative days, so I convert to positive years.

# %%
df["AGE_YEARS"] = df["DAYS_BIRTH"].abs() / 365

# %% [markdown]
# ### 9.2 Employment Years
# I replace the 365243 anomaly with NaN first, then convert to years.

# %%
df["DAYS_EMPLOYED_CLEAN"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
df["EMPLOYMENT_YEARS"] = df["DAYS_EMPLOYED_CLEAN"].abs() / 365

# %% [markdown]
# ### 9.3 Loan-to-Income Ratio ⭐
# `AMT_CREDIT / AMT_INCOME_TOTAL` - how large is the requested loan
# relative to the applicant's income. A higher ratio may indicate the
# applicant is requesting a loan that's large relative to what they earn.

# %%
df["LOAN_TO_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]

# %% [markdown]
# ### 9.4 Annuity-to-Income Ratio ⭐
# `AMT_ANNUITY / AMT_INCOME_TOTAL` - regular payment relative to income.
# This approximates the applicant's repayment burden.

# %%
df["ANNUITY_TO_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]

# %% [markdown]
# ## 10. Univariate Analysis
#
# Studying individual features on their own, before comparing anything
# against the target.

# %% [markdown]
# ### 10.1 Age Distribution

# %%
plt.figure(figsize=(10, 6))
sns.histplot(df["AGE_YEARS"], bins=40, kde=True)
plt.title("Applicant Age Distribution")
plt.xlabel("Age (Years)")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "03_age_distribution.png", dpi=120)
plt.show()

# %% [markdown]
# ### 10.2 Income Distribution
# Income contains extreme values, so I use a log transform just for
# visualization - this doesn't change the underlying data, only how it's
# plotted.

# %%
plt.figure(figsize=(10, 6))
sns.histplot(np.log1p(df["AMT_INCOME_TOTAL"]), bins=50, kde=True)
plt.title("Income Distribution (Log Scale)")
plt.xlabel("Log(Income)")
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "04_income_distribution.png", dpi=120)
plt.show()

# %% [markdown]
# ### 10.3 Loan Amount Distribution

# %%
plt.figure(figsize=(10, 6))
sns.histplot(np.log1p(df["AMT_CREDIT"]), bins=50, kde=True)
plt.title("Loan Amount Distribution (Log Scale)")
plt.xlabel("Log(Loan Amount)")
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "05_loan_amount_distribution.png", dpi=120)
plt.show()

# %% [markdown]
# ### 10.4 Employment Duration Distribution

# %%
plt.figure(figsize=(10, 6))
sns.histplot(df["EMPLOYMENT_YEARS"].dropna(), bins=50, kde=True)
plt.title("Employment Duration Distribution")
plt.xlabel("Years Employed")
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "06_employment_distribution.png", dpi=120)
plt.show()

# %% [markdown]
# **My takeaway:** I'm looking at each distribution's shape here - income
# and loan amount are both right-skewed (most applicants cluster at lower
# values with a long tail of high earners/large loans), which is why I
# log-transformed them just to make the histogram readable. Age looks
# fairly evenly spread across working adults. I'll note the actual skew
# and any surprising spikes once I run this against the real file.

# %% [markdown]
# ## 11. Outlier Analysis ⭐
#
# Visualizing income, loan amount, and annuity for extreme values.

# %%
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["AMT_INCOME_TOTAL"])
plt.title("Income Outlier Analysis")
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "07_income_outliers.png", dpi=120)
plt.show()

# %%
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["AMT_CREDIT"])
plt.title("Loan Amount Outlier Analysis")
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "08_loan_outliers.png", dpi=120)
plt.show()

# %%
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["AMT_ANNUITY"])
plt.title("Annuity Outlier Analysis")
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "09_annuity_outliers.png", dpi=120)
plt.show()

# %% [markdown]
# **My takeaway:** I expect a handful of very high-income/high-loan
# outliers given how skewed these distributions are. I won't necessarily
# remove these rows - a genuinely wealthy applicant isn't a data error -
# but I'll cap or log-transform these features during modeling so a small
# number of extreme values don't dominate the model's learned splits.

# %% [markdown]
# ## 12. Categorical Feature Distribution
#
# Understanding the population itself before comparing anything against
# default - e.g. what does the applicant base actually look like.

# %%
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="NAME_INCOME_TYPE")
plt.xticks(rotation=30)
plt.title("Distribution of Applicant Income Types")
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "10_income_type_distribution.png", dpi=120)
plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.countplot(data=df, x="NAME_EDUCATION_TYPE", ax=axes[0])
axes[0].set_title("Education Level")
axes[0].tick_params(axis="x", rotation=45)

sns.countplot(data=df, x="NAME_FAMILY_STATUS", ax=axes[1])
axes[1].set_title("Family Status")
axes[1].tick_params(axis="x", rotation=45)

sns.countplot(data=df, x="CODE_GENDER", ax=axes[2])
axes[2].set_title("Gender")

plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "11_categorical_distributions.png", dpi=120)
plt.show()

# %% [markdown]
# **My takeaway:** this tells me the shape of the applicant population
# itself, which matters for interpreting Section 13 correctly - if one
# category has very few applicants, its default rate can swing wildly on
# small counts and shouldn't be over-trusted the same way a large category
# would be.

# %% [markdown]
# ## 13. Bivariate Risk Analysis ⭐⭐⭐
#
# The most important section - comparing features against `TARGET` to see
# which ones actually relate to default.

# %% [markdown]
# ### 13.1 Default Rate by Age

# %%
df["AGE_GROUP"] = pd.cut(
    df["AGE_YEARS"],
    bins=[0, 25, 35, 45, 55, 65, 100],
    labels=["Below 25", "25-35", "35-45", "45-55", "55-65", "65+"]
)

age_default = df.groupby("AGE_GROUP", observed=True)["TARGET"].mean() * 100
print(age_default)

age_default.plot(kind="bar", figsize=(10, 6), color="#6A4C93")
plt.title("Default Rate by Age Group")
plt.ylabel("Default Rate (%)")
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "12_default_by_age.png", dpi=120)
plt.show()

# %% [markdown]
# ### 13.2 Default Rate by Income

# %%
df["INCOME_GROUP"] = pd.qcut(df["AMT_INCOME_TOTAL"], q=5, duplicates="drop")

income_default = df.groupby("INCOME_GROUP", observed=True)["TARGET"].mean() * 100
print(income_default)

income_default.plot(kind="bar", figsize=(12, 6), color="#F18F01")
plt.title("Default Rate by Income Group")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "13_default_by_income.png", dpi=120)
plt.show()

# %% [markdown]
# ### 13.3 Default Rate by Loan Amount

# %%
df["LOAN_GROUP"] = pd.qcut(df["AMT_CREDIT"], q=5, duplicates="drop")

loan_default = df.groupby("LOAN_GROUP", observed=True)["TARGET"].mean() * 100
print(loan_default)

loan_default.plot(kind="bar", figsize=(12, 6), color="#E63946")
plt.title("Default Rate by Loan Amount Group")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "14_default_by_loan_amount.png", dpi=120)
plt.show()

# %% [markdown]
# ### 13.4 Default Rate by Education

# %%
education_default = df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean() * 100
education_default = education_default.sort_values(ascending=False)
print(education_default)

# %% [markdown]
# ### 13.5 Default Rate by Income Type

# %%
income_type_default = df.groupby("NAME_INCOME_TYPE")["TARGET"].mean() * 100
income_type_default = income_type_default.sort_values(ascending=False)
print(income_type_default)

# %% [markdown]
# ### 13.6 Default Rate by Employment Duration

# %%
df["EMPLOYMENT_GROUP"] = pd.cut(
    df["EMPLOYMENT_YEARS"],
    bins=[-1, 1, 3, 5, 10, 20, 100],
    labels=["< 1 Year", "1-3 Years", "3-5 Years", "5-10 Years", "10-20 Years", "20+ Years"]
)

employment_default = df.groupby("EMPLOYMENT_GROUP", observed=True)["TARGET"].mean() * 100
print(employment_default)

employment_default.plot(kind="bar", figsize=(10, 6), color="#457B9D")
plt.title("Default Rate by Employment Duration")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "15_default_by_employment.png", dpi=120)
plt.show()

# %% [markdown]
# ### 13.7 Default Rate by Gender

# %%
gender_default = df.groupby("CODE_GENDER")["TARGET"].mean() * 100
print(gender_default)

# %% [markdown]
# ### 13.8 Default Rate by Family Status

# %%
family_default = df.groupby("NAME_FAMILY_STATUS")["TARGET"].mean().sort_values(ascending=False) * 100
print(family_default)

# %% [markdown]
# **My takeaway:** I'll fill in the specific highest/lowest-risk group for
# each of the above once I run this against the real data (see Section
# 18). Structurally, this section is exactly the evidence base for which
# features go into the model and which decision rules become defensible
# business logic later in Phase 5.

# %% [markdown]
# ## 14. Financial Risk Analysis ⭐⭐⭐
#
# Directly relevant to this project - checking whether the two engineered
# ratios from Section 9 actually relate to default.

# %% [markdown]
# ### Default Rate by Loan-to-Income Ratio

# %%
ratio_df = df.dropna(subset=["LOAN_TO_INCOME_RATIO"]).copy()
ratio_df["RATIO_GROUP"] = pd.qcut(ratio_df["LOAN_TO_INCOME_RATIO"], q=5, duplicates="drop")

ratio_default = ratio_df.groupby("RATIO_GROUP", observed=True)["TARGET"].mean() * 100
print(ratio_default)

ratio_default.plot(kind="bar", figsize=(10, 6), color="#D62839")
plt.title("Default Rate by Loan-to-Income Ratio")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "16_default_by_loan_to_income.png", dpi=120)
plt.show()

# %% [markdown]
# ### Default Rate by Annuity-to-Income Ratio

# %%
annuity_df = df.dropna(subset=["ANNUITY_TO_INCOME_RATIO"]).copy()
annuity_df["ANNUITY_RATIO_GROUP"] = pd.qcut(
    annuity_df["ANNUITY_TO_INCOME_RATIO"], q=5, duplicates="drop"
)

annuity_default = annuity_df.groupby("ANNUITY_RATIO_GROUP", observed=True)["TARGET"].mean() * 100
print(annuity_default)

annuity_default.plot(kind="bar", figsize=(10, 6), color="#2E86AB")
plt.title("Default Rate by Annuity-to-Income Ratio")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "17_default_by_annuity_to_income.png", dpi=120)
plt.show()

# %% [markdown]
# **My takeaway:** I'm using these two engineered ratios rather than the
# raw amounts because a raw loan amount alone doesn't tell you whether
# that's "a lot" for that applicant - the ratio does. If the pattern holds
# (higher ratio -> higher default rate), both become strong candidate
# features and strong candidates for the business rule engine in Phase 5.

# %% [markdown]
# ## 15. Asset Analysis

# %% [markdown]
# ### Car Ownership

# %%
car_default = df.groupby("FLAG_OWN_CAR")["TARGET"].mean() * 100
print(car_default)

# %% [markdown]
# ### Property Ownership

# %%
property_default = df.groupby("FLAG_OWN_REALTY")["TARGET"].mean() * 100
print(property_default)

# %% [markdown]
# **My takeaway:** owning assets (car/property) often acts as a proxy for
# financial stability. I'll note whether owners default less often once I
# run this - if the gap is small, I won't overweight these as features
# despite the intuitive story.

# %% [markdown]
# ## 16. Correlation Analysis ⭐
#
# Only numeric features can go into a Pearson correlation directly.

# %%
numeric_df = df.select_dtypes(include=["int64", "float64"])

target_correlation = numeric_df.corr()["TARGET"].sort_values()
print(target_correlation)

# %%
top_features = pd.concat([
    target_correlation.head(10),
    target_correlation.tail(10)
]).drop_duplicates()

plt.figure(figsize=(10, 8))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Features Most Correlated with Default")
plt.xlabel("Correlation with TARGET")
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "18_correlation_analysis.png", dpi=120)
plt.show()

# %% [markdown]
# ⚠️ **Important documentation note:** I'm not writing "correlation proves
# this feature causes default." Correlation indicates an association
# between variables but does not establish causation. I'll phrase my
# findings in Section 18 accordingly.

# %% [markdown]
# ## 17. Feature Availability Analysis ⭐⭐⭐
#
# I want to be explicit about *when* each piece of information is actually
# available, so the model only ever uses data that would genuinely exist
# at the moment a real credit decision gets made.
#
# **Available at loan application time:** age, income, employment,
# education, requested loan amount, housing, assets.
#
# **Available from credit history:** credit bureau information, previous
# applications, installment history, credit card history.
#
# **Why this matters:** using information that wouldn't actually have
# existed at approval time would make the model look better than it really
# is during testing, but fail in production. I'll keep this distinction in
# mind explicitly during Phase 3 feature engineering.

# %% [markdown]
# ## 18. EDA Key Findings ⭐⭐⭐
#
# **I'm filling these in only after running the notebook against the real
# data - not before.** The blanks below get replaced with the actual
# numbers from the cells above once I run this end to end.
#
# **Finding 1 - Class Imbalance:** Approximately ___% of applicants
# experienced payment difficulties, while ___% did not. This confirms
# significant class imbalance and requires appropriate model evaluation
# techniques (ROC-AUC, PR-AUC) during the ML stage rather than accuracy.
#
# **Finding 2 - Missing Values:** ___ columns show high (>40%) missingness,
# mostly in the ___ category. These require a defined strategy (drop vs.
# impute) before model training - see Section 6.
#
# **Finding 3 - Duplicate Rows:** ___ duplicate rows were found.
#
# **Finding 4 - Age Risk Pattern:** The ___ age group showed the highest
# observed default rate of ___%.
#
# **Finding 5 - Income Pattern:** Applicants in the ___ income group
# showed the highest observed default rate of ___%.
#
# **Finding 6 - Financial Burden:** Applicants with a loan-to-income ratio
# in the ___ band showed a default rate of ___%, compared to ___% for the
# lowest band.
#
# **Finding 7 - Employment:** Default rates varied across employment
# duration groups - the ___ group showed the highest rate at ___%,
# suggesting employment characteristics carry real predictive signal.
#
# **Finding 8 - Data Anomaly:** `DAYS_EMPLOYED` contains an anomalous
# value of 365243 days, affecting ___ rows (___%). This requires
# preprocessing before machine learning (see Section 8).

# %% [markdown]
# ## 19. Business Implications ⭐⭐⭐
#
# Connecting the EDA back to the actual banking problem NeoStats described.
#
# **Financial Risk:** loan-to-income and annuity-to-income ratios can
# provide useful indicators of repayment burden and should be surfaced as
# both model features and business rules.
#
# **Customer Segmentation:** age, employment characteristics, and income
# category can be used to understand differences in observed default
# behaviour across customer segments - useful for the business rule engine
# and for explaining individual predictions to a non-technical reviewer.
#
# **Data Quality:** missing values and the `DAYS_EMPLOYED` anomaly need to
# be addressed in preprocessing before the model is trained, or they will
# quietly distort what the model learns.

# %% [markdown]
# ## 20. Next Steps
#
# 1. Data cleaning
# 2. Missing value treatment
# 3. Outlier handling
# 4. Categorical encoding
# 5. Feature engineering (including features from bureau/previous
#    application/installments tables - see Section 21 below)
# 6. Train/test/validation split
# 7. Baseline machine learning model
# 8. Model evaluation (ROC-AUC, PR-AUC)
# 9. Model explainability (SHAP)
# 10. Talk-to-data chatbot + business rule derivation

# %% [markdown]
# ## 21. Bonus: Supplementary Table Analysis (Credit History & Repayment Behaviour)
#
# Everything above uses only `application_train.csv`. But the assignment
# specifically asks me to analyse "credit history" and "repayment
# behaviour" - and the real signal for both lives in three *other* tables
# that only connect to `application_train` through a join, not as columns
# already sitting in this file. I'm covering them here so this notebook
# genuinely covers all five areas NeoStats listed (demographics, financial,
# credit history, repayment behaviour, data quality), not just the four
# that live inside the main table alone.

# %%
import sys
sys.path.append(os.path.join(os.getcwd(), ".."))
from src.data.loader import load_all_tables

tables = load_all_tables(data_dir="../data")

# %% [markdown]
# ### 21.1 Credit History - bureau.csv
# The applicant's credit history at *other* lenders, not just Home Credit.

# %%
bureau = tables.get("bureau")
if bureau is not None:
    print(f"bureau.csv: {bureau.shape[0]:,} records across "
          f"{bureau['SK_ID_CURR'].nunique():,} unique applicants")
    print("\nCredit status breakdown:")
    print(bureau["CREDIT_ACTIVE"].value_counts())

    bureau_count = bureau.groupby("SK_ID_CURR").size().rename("bureau_credit_count")
    df_bureau = df.merge(bureau_count, on="SK_ID_CURR", how="left")
    df_bureau["bureau_credit_count"] = df_bureau["bureau_credit_count"].fillna(0)
    df_bureau["has_bureau_history"] = df_bureau["bureau_credit_count"] > 0

    rate = df_bureau.groupby("has_bureau_history")["TARGET"].mean() * 100
    print("\nDefault rate by presence of external bureau history:")
    print(rate)
else:
    print("bureau.csv not found in data/ - skipping.")

# %% [markdown]
# ### 21.2 Previous Applications - previous_application.csv
# The applicant's past loan applications with Home Credit itself.

# %%
prev = tables.get("previous_application")
if prev is not None:
    refused = prev[prev["NAME_CONTRACT_STATUS"] == "Refused"]
    refusal_counts = refused.groupby("SK_ID_CURR").size().rename("refusal_count")

    df_prev = df.merge(refusal_counts, on="SK_ID_CURR", how="left")
    df_prev["refusal_count"] = df_prev["refusal_count"].fillna(0)
    df_prev["has_prior_refusal"] = df_prev["refusal_count"] > 0

    rate = df_prev.groupby("has_prior_refusal")["TARGET"].mean() * 100
    print("Default rate by prior refusal history:")
    print(rate)

    rate.plot(kind="bar", figsize=(5, 4), color=["#457B9D", "#E63946"])
    plt.xticks([0, 1], ["No Prior Refusal", "Has Prior Refusal"], rotation=0)
    plt.title("Default Rate by Prior Refusal History")
    plt.ylabel("Default Rate (%)")
    plt.tight_layout()
    plt.savefig(SCREENSHOT_DIR / "19_default_by_prior_refusal.png", dpi=120)
    plt.show()
else:
    print("previous_application.csv not found in data/ - skipping.")

# %% [markdown]
# ### 21.3 Repayment Behaviour - installments_payments.csv
# Compares what applicants were supposed to pay vs. what they actually
# paid and when - the most direct historical repayment-discipline signal
# in the whole dataset.

# %%
installments = tables.get("installments")
if installments is not None:
    installments["DAYS_LATE"] = installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
    avg_late = installments.groupby("SK_ID_CURR")["DAYS_LATE"].mean().rename("avg_days_late")

    df_late = df.merge(avg_late, on="SK_ID_CURR", how="left")
    df_late["LATE_BAND"] = pd.cut(
        df_late["avg_days_late"],
        bins=[-1000, 0, 5, 15, 1000],
        labels=["Never late / early", "1-5 days late", "6-15 days late", "15+ days late"]
    )
    rate = df_late.groupby("LATE_BAND", observed=True)["TARGET"].mean() * 100
    print("Default rate by average historical payment lateness:")
    print(rate)

    rate.plot(kind="bar", figsize=(6, 4), color="#D62839")
    plt.title("Default Rate by Average Historical Payment Lateness")
    plt.ylabel("Default Rate (%)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(SCREENSHOT_DIR / "20_default_by_payment_lateness.png", dpi=120)
    plt.show()
else:
    print("installments_payments.csv not found in data/ - skipping.")

# %% [markdown]
# **My takeaway (Section 21):** I expect both prior refusals and historical
# payment lateness to show a clear, meaningful gap in default rate. If
# they do, this is my strongest justification for the multi-table feature
# engineering work in Phase 3 - it shows these supplementary tables carry
# real signal application_train.csv alone doesn't have, not just extra
# complexity for its own sake.
