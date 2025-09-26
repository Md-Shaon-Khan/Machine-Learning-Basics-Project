import pandas as pd
import numpy as np

# Reproducibility
np.random.seed(42)

# Define possible Areas with factors
areas = {"Dhaka": 5000, "Ctg": 3000, "Rangpur": 1000, "Khulna": 7000}

# Number of rows
n = 50

# Generate synthetic dataset
data = {
    "Marketing Spend": np.round(np.random.uniform(100000, 200000, n), 2),
    "Administration": np.round(np.random.uniform(80000, 160000, n), 2),
    "Transport": np.round(np.random.uniform(300000, 500000, n), 2),
    "Area": np.random.choice(list(areas.keys()), n),
}

# Calculate Profit as a function of features
data["Profit"] = (
    0.5 * data["Marketing Spend"]
    + 0.3 * data["Administration"]
    + 0.2 * data["Transport"]
    + [areas[a] for a in data["Area"]]
    + np.random.normal(0, 5000, n)  # add noise
)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save dataset as CSV
df.to_csv("linear_regression_project_dataset.csv", index=False)

print(df.head())
