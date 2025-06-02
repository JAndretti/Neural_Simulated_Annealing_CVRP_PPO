import pandas as pd
import os

CSV_FILE = "res_model_invalid_step"  # Name of the CSV file without extension
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

output_filename = "./res/models_res/" + CSV_FILE + "_sorted.txt"

with open(output_filename, "w") as f:
    f.write("Models sorted by final_cost:\n")
    f.write(df_sorted.drop(columns=["initial_cost"]).head(50).to_string(index=False))
