import pandas as pd

# Column names as defined by NASA dataset
columns = ["engine_id", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

# Load dataset
data = pd.read_csv("data/train_FD001.txt", sep=" ", header=None)
data.columns = columns

# Remove empty columns
data.dropna(axis=1, inplace=True)

print(data.head())