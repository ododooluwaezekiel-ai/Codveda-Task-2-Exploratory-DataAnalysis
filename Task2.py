import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# TASK 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

# 1. LOAD THE DATASET
df = pd.read_csv("1) iris.csv")

print("========== FIRST 5 ROWS ==========")
print(df.head())


# ==========================================
# 2. DATASET INFORMATION
# ==========================================

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())


# ==========================================
# 3. SELECT NUMERICAL COLUMNS
# ==========================================

numeric_columns = df.select_dtypes(include="number").columns


# ==========================================
# 4. SUMMARY STATISTICS
# ==========================================

print("\n========== MEAN ==========")
print(df[numeric_columns].mean())

print("\n========== MEDIAN ==========")
print(df[numeric_columns].median())

print("\n========== MODE ==========")
print(df[numeric_columns].mode().iloc[0])

print("\n========== STANDARD DEVIATION ==========")
print(df[numeric_columns].std())

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df[numeric_columns].describe())


# ==========================================
# 5. HISTOGRAMS
# ==========================================

df[numeric_columns].hist(
    figsize=(10, 8),
    bins=10,
    edgecolor="black"
)

plt.suptitle("Distribution of Iris Measurements")
plt.tight_layout()
plt.show()


# ==========================================
# 6. BOXPLOT
# ==========================================

plt.figure(figsize=(10, 6))

sns.boxplot(data=df[numeric_columns])

plt.title("Boxplot of Iris Measurements")
plt.xlabel("Features")
plt.ylabel("Measurement (cm)")
plt.show()


# ==========================================
# 7. SCATTER PLOT
# SEPAL LENGTH VS SEPAL WIDTH
# ==========================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="sepal_length",
    y="sepal_width",
    hue="species"
)

plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()


# ==========================================
# 8. SCATTER PLOT
# PETAL LENGTH VS PETAL WIDTH
# ==========================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="petal_length",
    y="petal_width",
    hue="species"
)

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()


# ==========================================
# 9. SCATTER PLOT
# SEPAL LENGTH VS PETAL LENGTH
# ==========================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="sepal_length",
    y="petal_length",
    hue="species"
)

plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()


# ==========================================
# 10. CORRELATION MATRIX
# ==========================================

correlation = df[numeric_columns].corr()

print("\n========== CORRELATION MATRIX ==========")
print(correlation)


# ==========================================
# 11. CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix of Iris Features")
plt.show()


# ==========================================
# 12. FINAL INSIGHTS
# ==========================================

print("\n========== EDA INSIGHTS ==========")

print("1. The dataset contains 150 rows and 5 columns.")

print("2. There are no missing values in the dataset.")

print("3. The dataset contains 3 duplicate rows.")

print("4. Petal length and petal width have the strongest positive correlation.")

print("5. The scatter plots show that the Iris species can be distinguished more clearly using petal measurements.")

print("6. Histograms show the distribution of the numerical features.")

print("7. Boxplots help identify the spread and possible outliers in the measurements.")