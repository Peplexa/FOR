import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score, classification_report
import time

# Function to load dataset
def load_dataset(feature_path, label_path):
    features = np.load(feature_path)
    labels = np.load(label_path)
    return features, labels

# Load the test dataset
X_test, y_test = load_dataset('./npy3/testing_features.npy', './npy3/testing_labels.npy')

# Load the trained model
loaded_rf_classifier = load('./random_forest_model.joblib')

# Start timing
start_time = time.time()

# Make predictions for the entire dataset
y_pred = loaded_rf_classifier.predict(X_test)

# End timing
end_time = time.time()
total_time = end_time - start_time

# Calculate average time per 1000 files
num_files = len(X_test)
average_time_per_1000 = (total_time / num_files) * 1000
print(f"Average evaluation time per 1000 files: {average_time_per_1000:.4f} seconds")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
