import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load

# Function to load dataset from .npy files
def load_dataset(feature_path, label_path):
    features = np.load(feature_path)
    labels = np.load(label_path)
    return features, labels

# Load training and test datasets
X_train, y_train = load_dataset('./npy3/training_features.npy', './npy3/training_labels.npy')
X_test, y_test = load_dataset('./npy3/testing_features.npy', './npy3/testing_labels.npy')

# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42) #42 is the seed I've seen the most in examples so I went with that.
rf_classifier.fit(X_train, y_train)

# Predict on the test set and evaluate the classifier
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

# Save the trained model to disk
model_filename = './random_forest_model.joblib'
dump(rf_classifier, model_filename)
print(f"Model saved to {model_filename}")

# Example of how to load and use the model
loaded_rf_classifier = load(model_filename)
# Use the loaded model to make predictions on the same test set
example_predictions = loaded_rf_classifier.predict(X_test)
# Evaluate the loaded model's predictions
example_accuracy = accuracy_score(y_test, example_predictions)
print(f"Accuracy of loaded model: {example_accuracy}")
