import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import time

class CRNN(nn.Module):
    def __init__(self, input_size, output_size=2, hidden_dim=128, n_layers=2):
        super(CRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.lstm = nn.LSTM(32 * (input_size // 4), hidden_dim, n_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Compute correct feature size dynamically
        feature_size = x.size(2) * x.size(1)  # length * channels
        x = x.view(-1, feature_size)  # Flatten for input to LSTM
        
        # Reshape for LSTM: from (batch_size, feature_size) to (batch_size, seq_len, feature_size)
        x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Get the output of the last time step
        
        x = self.dropout(x)
        x = self.fc(x)
        return x

def load_dataset(features_path, labels_path):
    features = np.load(features_path)
    labels = np.load(labels_path)
    return features, labels

# Load datasets
X_test, y_test = load_dataset('npy3/testing_features.npy', 'npy3/testing_labels.npy')

# Preparing the data loader
batch_size = 16
test_tensor = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())
test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

# Initialize and load the trained model
input_size = X_test.shape[1]  # Update input_size if necessary
model = CRNN(input_size)
model.load_state_dict(torch.load('best_crnn3_model.pth'))
model.eval()

criterion = nn.CrossEntropyLoss()

def evaluate_model(model, test_loader, criterion):
    start_time = time.time()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

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

    test_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f'Test Loss: {test_loss}, Accuracy: {accuracy}%')
    print("Detailed classification report:")
    print(classification_report(all_labels, all_preds, digits=4))

    evaluations_per_second = (elapsed_time / total)
    print(f'Time for 1000 evaluations  {evaluations_per_second*1000} ')

# Evaluate the model
evaluate_model(model, test_loader, criterion)
