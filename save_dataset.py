# save_dataset.py
import pandas as pd
from sklearn.datasets import load_iris
import os

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Save to CSV
os.makedirs("data", exist_ok=True)
df.to_csv("data/iris.csv", index=False)