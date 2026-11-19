
import pandas as pd

# Load the raw dataset
df = pd.read_csv("data/raw/breast_cancer.csv")

print("DATA VALIDATION REPORT")

# 1. Check dataset shape
print("\nDataset Shape:")
print(df.shape)

# 2. Check missing values
print("\nMissing Values:")
print(df.isnull().sum().sum())

# 3. Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# 4. Check data types
print("\nData Types:")
print(df.dtypes)

# 5. Check target values
print("\nTarget Distribution:")
print(df["target"].value_counts())

# 6. Check invalid target values
print("\nInvalid Target Values:")
print(df[~df["target"].isin([0, 1])].shape[0])

# 7. Statistical summary
print("\nStatistical Summary:")
print(df.describe())

print("\nData validation completed!")
