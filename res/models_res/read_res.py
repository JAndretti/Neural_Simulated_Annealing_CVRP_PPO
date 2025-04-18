import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

CSV_FILE = "res_model_init_temp_benchmark"
PATH = "./res/models_res/"
CSV_PATH = os.path.join(PATH, CSV_FILE + ".csv")

# Load the CSV file
df = pd.read_csv(CSV_PATH)

# Display the first few rows of the DataFrame ordered by smallest final_cost
df_sorted = df.sort_values("final_cost", ascending=True)
print("Models sorted by final_cost:")
print(df_sorted.drop(columns=["initial_cost"]).head(50))

# Select the parameter columns (excluding model, initial_cost, final_cost)
param_cols = [
    col for col in df.columns if col not in ["model", "initial_cost", "final_cost"]
]

# Encode categorical columns if needed
X = df[param_cols].copy()
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

y = df["final_cost"]

# Linear regression to estimate the influence of each parameter
reg = LinearRegression()
reg.fit(X, y)
importances = pd.Series(reg.coef_, index=param_cols).abs().sort_values(ascending=False)

print("\n Parameters that most influence final cost:")
print(importances)

output_filename = "./res/models_res/" + CSV_FILE + "_sorted.txt"

with open(output_filename, "w") as f:
    f.write("Models sorted by final_cost:\n")
    f.write(df_sorted.drop(columns=["initial_cost"]).head(50).to_string(index=False))
    f.write("\n\nParameters that most influence final cost:\n")
    f.write(importances.to_string())
