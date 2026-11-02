import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)
import joblib

# Load dataset
df = pd.read_csv("activity_dataset.csv")

# Split features and label
X = df.drop("label", axis=1)
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=None
)

# Train model
print("Training model...")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("\n==============================")
print("MODEL ACCURACY")
print("==============================")
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification Report
print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "activity_model.pkl")

print("\n==============================")
print("Model saved successfully!")
print("==============================")