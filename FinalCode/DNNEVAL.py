import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import mode
from sklearn.metrics import classification_report
import time
#This file is similar to CRNNEVAL.py.
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

def load_dataset(features_path, labels_path):
    features = np.load(f'npy3/{features_path}')
    labels = np.load(f'npy3/{labels_path}')
    return features, labels

X_test, y_test = load_dataset('testing_features.npy', 'testing_labels.npy')
test_tensor = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())
test_loader = DataLoader(test_tensor, batch_size=16, shuffle=False)

input_size = X_test.shape[1]

def majority_vote(predictions):
    predictions_stack = torch.stack(predictions)
    majority_vote, _ = mode(predictions_stack.cpu().numpy(), axis=0)
    return torch.tensor(majority_vote).squeeze()

def evaluate_with_majority_vote(num_models=3):
    start_time = time.time()
    model_predictions = []
    for i in range(num_models):
        model = AudioFeatureNet(input_size)
        model.load_state_dict(torch.load(f'best_model_{i+1}.pth'))
        model.eval()
        
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        model_predictions.append(torch.tensor(all_preds))

    final_predictions = majority_vote(model_predictions)
    correct = (final_predictions == torch.tensor(all_labels)).sum().item()
    total = len(all_labels)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    
    accuracy = 100 * correct / total
    print(f'Final Test Accuracy after Majority Vote: {accuracy}%')
    print("Detailed classification report:")
    print(classification_report(all_labels, final_predictions.numpy(), digits=4))
    evaluations_per_second = (elapsed_time / total)
    print(f'Time for 1000 evaluations  {evaluations_per_second*1000} ')

if __name__ == "__main__":
    evaluate_with_majority_vote(num_models=3)
