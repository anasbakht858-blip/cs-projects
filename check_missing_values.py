import pandas as pd

df = pd.read_csv("activity_dataset.csv")

print("Missing values:")
print(df.isnull().sum().sum())