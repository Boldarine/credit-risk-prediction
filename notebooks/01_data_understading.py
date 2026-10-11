# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load raw dataset
df = pd.read_excel("../data/raw/credit_card_clients.xls",header=1)
# %%
df.shape
# %%
df.info()
# %%
df.isnull().sum()
# %%
df.head()
# %%
df.describe().T
# %%
df["SEX"].value_counts().sort_index()
# %%
df["EDUCATION"].value_counts().sort_index()
# %%
df["MARRIAGE"].value_counts().sort_index()
# %%
df["EDUCATION"].isin([0, 5, 6]).sum()
df[df["EDUCATION"].isin([0, 5, 6])][["ID", "EDUCATION", "AGE", "MARRIAGE"]]
# %%
df[df["EDUCATION"].isin([0, 5, 6])]["default payment next month"].value_counts(normalize=True)
# %%
df["AGE"].agg(["min", "max"])
# %%
df[["LIMIT_BAL",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]].describe().T
# %%
(df[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]] < 0).sum()
# %%
plt.figure(figsize=(10, 6))

plt.hist(
    df["AGE"],
    bins=20,
    alpha=0.5,
    label="Age"
)

plt.xlabel("AGE")
plt.ylabel("Number of Clients")
plt.title("Distribution by AGE Clients")
plt.legend()
plt.show()
# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Sex
df["SEX"].value_counts().sort_index().plot(
    kind="bar",
    ax=axes[0, 0]
)
axes[0, 0].set_title("Sex distribution")
axes[0, 0].set_xlabel("Sex")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].set_xticklabels(["Male", "Female"], rotation=0)
axes[0, 0].bar_label(axes[0, 0].containers[0])

# Education
df["EDUCATION"].value_counts().sort_index().plot(
    kind="bar",
    ax=axes[0, 1]
)
axes[0, 1].set_title("Education distribution")
axes[0, 1].set_xlabel("Education")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].bar_label(axes[0, 1].containers[0])

# Marriage
df["MARRIAGE"].value_counts().sort_index().plot(
    kind="bar",
    ax=axes[1, 0]
)
axes[1, 0].set_title("Marriage distribution")
axes[1, 0].set_xlabel("Marriage")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].bar_label(axes[1, 0].containers[0])

# Target
df["default payment next month"].value_counts().sort_index().plot(
    kind="bar",
    ax=axes[1, 1]
)
axes[1, 1].set_title("Default distribution")
axes[1, 1].set_xlabel("Default")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].set_xticklabels(["No", "Yes"], rotation=0)
axes[1, 1].bar_label(axes[1, 1].containers[0])

plt.tight_layout()
plt.show()
# %%
bill_cols = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6"
]

bill_long = df[bill_cols].melt(
    var_name="Month",
    value_name="Bill Amount"
)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Create boxplot
sns.boxplot(
    data=bill_long,
    x="Month",
    y="Bill Amount",
    ax=ax
)

# Customize labels
ax.set_title("Bill Amount Distribution by Month")
ax.set_xlabel("Month")
ax.set_ylabel("Bill Amount")

# Replace technical column names with month names
ax.set_xticklabels(["Sep", "Aug", "Jul", "Jun", "May", "Apr"])

plt.tight_layout()
plt.show()
# %%
df[bill_cols].describe()
# %%
pay_cols = [
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"
]

pay_long = df[pay_cols].melt(
    var_name="Month",
    value_name="Pay Amount"
)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Create boxplot
sns.boxplot(
    data=pay_long,
    x="Month",
    y="Pay Amount",
    ax=ax
)

# Customize labels
ax.set_title("Pay Amount Distribution by Month")
ax.set_xlabel("Month")
ax.set_ylabel("Pay Amount")

# Replace technical column names with month names
ax.set_xticklabels(["Sep", "Aug", "Jul", "Jun", "May", "Apr"])

plt.tight_layout()
plt.show()
# %%
df[pay_cols].describe()
# %%
