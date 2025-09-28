import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("data.csv")

print("✅ Dataset loaded successfully!")

# First 5 rows
print("\n🔹 First 5 rows of dataset:")
print(data.head())

# Dataset info
print("\n🔹 Dataset Information:")
print(data.info())

# Summary statistics
print("\n🔹 Summary Statistics:")
print(data.describe())

# Missing values
print("\n🔹 Missing Values:")
print(data.isnull().sum())

# Correlation heatmap (only numeric columns)
print("\n🔹 Generating correlation heatmap...")
plt.figure(figsize=(8, 6))
sns.heatmap(data.select_dtypes(include='number').corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Distribution plots
print("\n🔹 Generating distribution plots...")
data.hist(figsize=(10, 8), bins=10, edgecolor="black")
plt.suptitle("Feature Distributions")
plt.show()
