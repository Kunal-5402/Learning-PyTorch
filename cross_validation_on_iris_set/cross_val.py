from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

class iris_dataset(Dataset):
    def __init__(self, features, labels):
        super().__init__()
        self.feature = features
        self.labels = labels
        self.f_transform = lambda x: torch.tensor(x, dtype=torch.float32)
        self.l_transform = lambda x: torch.tensor(x, dtype=torch.long)

    def  __getitem__(self, index):
        return self.f_transform(self.feature[index]), self.l_transform(self.labels[index])
    
    def __len__(self):
        return len(self.feature)
    
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network_layers = nn.Sequential(
            nn.Linear(in_features=4, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=3)
        )
    
    def forward(self, x):
        return self.network_layers(x)


def train(model, train_loader, val_loader, loss_fn, optimizer, num_epochs):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for features, labels in train_loader:
            output = model(features)
            loss = loss_fn(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = loss_fn(outputs, labels)
                running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    print("Training complete.")
    return train_losses, val_losses, model

def test(test_loader, model):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad(): 
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the test set: {accuracy:.2f}%")

X,y = load_iris(return_X_y=True)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, stratify=y)
folds = KFold(n_splits=3, shuffle=True)

for train_index, val_index in folds.split(X,y):
    x_train, y_train = X[train_index], y[train_index]
    x_val, y_val = X[val_index], y[val_index]
    
    train_set = iris_dataset(features=x_train, labels=y_train)
    val_set = iris_dataset(features=x_val, labels=y_val)

    train_loader = DataLoader(train_set, batch_size=4)
    val_loader = DataLoader(val_set, batch_size=4)

    model = ANN()
    optimizer = optim.Adam(params=model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    train(model, train_loader, val_loader, loss_fn, optimizer, 20)

train_set = iris_dataset(features=train_x, labels=train_y)
val_set = iris_dataset(features=test_x, labels=test_y)

train_loader = DataLoader(train_set, batch_size=4)
val_loader = DataLoader(val_set, batch_size=4)

model = ANN()
optimizer = optim.Adam(params=model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
    
t_loss, v_loss, model = train(model, train_loader, val_loader, loss_fn, optimizer, 20)
test(val_loader, model)