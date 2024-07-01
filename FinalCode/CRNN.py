import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
        
        # Compute the correct feature size dynamically
        feature_size = x.size(2) * x.size(1)  # length * channels
        x = x.view(-1, feature_size)  # Reshape for LSTM, removing the need for the non-existing fourth dimension
        
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
X_train, y_train = load_dataset('npy3/training_features.npy', 'npy3/training_labels.npy')
X_val, y_val = load_dataset('npy3/validation_features.npy', 'npy3/validation_labels.npy')
X_test, y_test = load_dataset('npy3/testing_features.npy', 'npy3/testing_labels.npy')

# Preparing the data loaders
batch_size = 16
train_tensor = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
val_tensor = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long())
test_tensor = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())

train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

# Initialize model
input_size = X_train.shape[1]
model = CRNN(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, patience=3):
    best_val_loss = np.Inf
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total

        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Accuracy: {accuracy}%')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_crnn3_model.pth')
            patience_counter = 0
            print("Validation loss decreased, saving model...")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print("Stopping early due to lack of improvement.")
                break

    model.load_state_dict(torch.load('best_crnn3_model.pth'))
    print("Training complete. Model with lowest validation loss restored.")

#This evaluation mode is mainly for testing, the in depth evals come from the specific py file CRNNEVAL.py
def evaluate_model(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f'Test Loss: {test_loss}, Accuracy: {accuracy}%')




# Example training execution
if __name__ == "__main__":
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20, patience=5)
    # Load the best model
    model.load_state_dict(torch.load('best_crnn3_model.pth'))

    # Evaluate the model with the test data
    evaluate_model(model, test_loader, criterion)