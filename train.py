# train.py
import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

# Save the model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/logistic_regression.pkl")

# Save the accuracy to a metrics file
os.makedirs("metrics", exist_ok=True)
with open("metrics/accuracy.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}")