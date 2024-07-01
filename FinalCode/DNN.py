import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import mode

# Set a base seed for reproducibility this is just an initial demonstration, will be varied later
np.random.seed(666)
torch.manual_seed(666)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(666)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to load datasets
def load_dataset(features_path, labels_path):
    features = np.load(f'npy3/{features_path}')
    labels = np.load(f'npy3/{labels_path}')
    return features, labels

# Load datasets
X_train, y_train = load_dataset('training_features.npy', 'training_labels.npy')
X_val, y_val = load_dataset('validation_features.npy', 'validation_labels.npy')
X_test, y_test = load_dataset('testing_features.npy', 'testing_labels.npy')

# Neural network architecture
class AudioFeatureNet(nn.Module):
    def __init__(self, input_size, output_size=2):
        super(AudioFeatureNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x)

input_size = X_train.shape[1]
model = AudioFeatureNet(input_size)

# Convert datasets to PyTorch tensors and create data loaders
batch_size = 16
train_tensor = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
val_tensor = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long())
test_tensor = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())

train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

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
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
            print("Validation loss decreased, saving model...")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print("Stopping early due to lack of improvement.")
                break

    model.load_state_dict(torch.load('best_model.pth'))
    print("Training complete. Model with lowest validation loss restored.")

# Evaluate function remains unchanged

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Train and Save Multiple Models, due to high variance in initial seeds. 
def train_and_save_models(num_models=5, epochs=20):
    for i in range(num_models):
        # Set a different seed for each model to ensure different initial weights
        seed = 666 + i * 10
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        model = AudioFeatureNet(input_size)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
        
        print(f"Training model {i+1}")
        train_model(model, train_loader, val_loader, criterion, optimizer, epochs=epochs)
        
        # Save each model with a unique name
        torch.save(model.state_dict(), f'best_model_{i+1}.pth')

# Majority Vote Function
def majority_vote(predictions):
    predictions_stack = torch.stack(predictions)
    majority_vote, _ = mode(predictions_stack.cpu().numpy(), axis=0)
    return torch.tensor(majority_vote).squeeze()

# Evaluate Models with Majority Voting
def evaluate_with_majority_vote(num_models=3):
    model_predictions = []
    for i in range(num_models):
        model = AudioFeatureNet(input_size)
        model.load_state_dict(torch.load(f'best_model_{i+1}.pth'))
        model.eval()
        
        predictions = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.append(predicted)
                
        model_predictions.append(torch.cat(predictions))

    final_predictions = majority_vote(model_predictions)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            total += labels.size(0)
            correct += (final_predictions[:labels.size(0)] == labels).sum().item()
            final_predictions = final_predictions[labels.size(0):]
    
    print(f'Final Test Accuracy after Majority Vote: {100 * correct / total}%')

# Main execution
if __name__ == "__main__":
    train_and_save_models(num_models=3, epochs=20)
    evaluate_with_majority_vote(num_models=3)
