import pandas as pd

# Replace with your CSV filename
df = pd.read_csv("activity_dataset.csv")
print(df["99"].value_counts())

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nClass Distribution:")
print(df.iloc[:, -1].value_counts())