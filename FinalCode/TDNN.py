import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def load_dataset(features_path, labels_path):
    features = np.load(features_path)
    labels = np.load(labels_path)
    return features, labels

# Assuming the same data loading paths as in the CRNN, if not change your path
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
        x = x.unsqueeze(1)  # Add channel dimension for Conv1d
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Initialize TDNN model
model = TDNN(input_size=212)  # Ensure input_size matches your feature dimension, this follows the extractorBest.py features (212 in total)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Training function
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
            torch.save(model.state_dict(), 'best_tdnn_model.pth')
            patience_counter = 0
            print("Validation loss decreased, saving model...")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print("Stopping early due to lack of improvement.")
                break

    model.load_state_dict(torch.load('best_tdnn_model.pth'))
    print("Training complete. Model with lowest validation loss restored.")

# Evaluation function
def evaluate_model(model, test_loader, criterion):
    model.eval()
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

# Main execution
if __name__ == "__main__":
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=40, patience=5)
    evaluate_model(model, test_loader, criterion)
