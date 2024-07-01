import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import time

class TDNN(nn.Module):
    def __init__(self, input_size, output_size=2):
        super(TDNN, self).__init__()
        self.layer1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.layer2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.layer3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Function to load the dataset
def load_dataset(features_path, labels_path):
    features = np.load(features_path)
    labels = np.load(labels_path)
    return features, labels

# Load the test dataset
X_test, y_test = load_dataset('npy3/testing_features.npy', 'npy3/testing_labels.npy')
test_tensor = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())
test_loader = DataLoader(test_tensor, batch_size=16, shuffle=False)

# Load the trained model
model = TDNN(input_size=212)  # Ensure input_size matches your feature dimension
model.load_state_dict(torch.load('best_tdnn_model.pth'))
model.eval()

# Criterion
criterion = nn.CrossEntropyLoss()

# Evaluate model function
def evaluate_model(model, test_loader, criterion):
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    start_time = time.time()

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    end_time = time.time()
    elapsed_time = end_time - start_time

    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    print(f'Test Loss: {test_loss}, Accuracy: {accuracy}%')

    # Detailed evaluation metrics
    print("Detailed classification report:")
    print(classification_report(all_labels, all_preds, digits=4))

    evaluations_per_second = (elapsed_time / total)
    print(f'Time for 1000 evaluations  {evaluations_per_second*1000} ')

# Call evaluate_model
evaluate_model(model, test_loader, criterion)
