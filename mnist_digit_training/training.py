from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn, optim
from torch.functional import F
from torchview import draw_graph
from torch.utils.data import Dataset, DataLoader

import plotly.graph_objects as go

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} now!")

class MNIST_dataset(Dataset):
    def  __init__(self, images, labels):
        super().__init__()
        self.images = images
        self.labels = labels
        self.image_tranform = lambda x: torch.tensor(x, dtype=torch.float32)
        self.label_tranform = lambda x: torch.tensor(x, dtype=torch.long)

    def __getitem__(self, index):
        return self.image_tranform(images[index].reshape(-1)), self.label_tranform(labels[index])
    
    def __len__(self):
        return len(images)


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        input_neuron = 64
        hidded_1_neurons = 32
        hidded_2_neurons = 32
        hidded_3_neurons = 64
        output_neurons = 10

        self.network_layers = nn.Sequential(
            nn.Linear(in_features=input_neuron, out_features=hidded_1_neurons),
            nn.ReLU(),
            nn.Linear(in_features=hidded_1_neurons, out_features=hidded_2_neurons),
            nn.ReLU(),
            nn.Linear(in_features=hidded_2_neurons, out_features=hidded_3_neurons),
            nn.ReLU(),
            nn.Linear(in_features=hidded_3_neurons, out_features=output_neurons)
        )
    def forward(self, x):
        return self.network_layers(x)

data = load_digits()
print(data.keys())
images, labels = data['images'], data['target']


random_indices = np.random.choice(len(images), 9, replace=False)
plt.figure(figsize=(10, 10))

for i, idx in enumerate(random_indices):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[idx], cmap='gray')
    plt.title(f"Label: {labels[idx]}")

# plt.tight_layout()
plt.savefig("digit_data.png")

train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=0.2, random_state=123, stratify=labels)

train_set = MNIST_dataset(images=train_x, labels=train_y)
test_set = MNIST_dataset(images=test_x, labels=test_y)

train_loader = DataLoader(train_set, batch_size=64)
test_loader = DataLoader(test_set, batch_size=64)

model = ANN()

model_graph = draw_graph(model, input_size=[64], expand_nested=True, save_graph=True, filename="NN")

optimizer = optim.Adam(params=model.parameters(),lr=0.01)
loss_fn = nn.CrossEntropyLoss()

def train(train_loader, test_loader, model, loss_fn, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    model.to(device)
    for epoch in range(num_epochs):
        model.train() #Set model to train mode
        running_train_loss = 0.0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            
            optimizer.zero_grad()   # Clear gradients from previous step
            loss.backward()         # Compute gradients
            optimizer.step()        # Update weights
            
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = loss_fn(outputs, labels)
                running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    print("Training complete.")
    return train_losses, val_losses, model

def test(test_loader, model):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No need to track gradients during evaluation
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)  # Get the class with highest probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the test set: {accuracy:.2f}%")

def save_loss_curve( train_loss, val_loss, num_epochs):
    # Create a line plot for training and validation losses
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(range(1, num_epochs + 1)), y=train_loss,
                            mode='lines+markers', name='Train Loss'))

    fig.add_trace(go.Scatter(x=list(range(1, num_epochs + 1)), y=val_loss,
                            mode='lines+markers', name='Validation Loss'))
    # Update layout
    fig.update_layout(
        title='Training and Validation Loss',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        template='plotly_dark',
        legend=dict(x=0, y=1, traceorder='normal')
    )

    fig.write_image("loss_plot.png", width=1024, height=512)


num_epochs = 30
train_loss, val_loss, model = train(train_loader, test_loader, model, loss_fn, optimizer, num_epochs)
save_loss_curve( train_loss, val_loss, num_epochs)
test(test_loader, model)

"""
This is the recommended way to say trained model by PyTorch, saving only weights
- torch.save(model.state_dict(), 'iris_model.pth')

To load the model, we will have to instantiate the model again with same network class, then load the trained weights in the 
model architecture for evaluations.

- model = ANN(4,10,3)
- model.load_state_dict(torch.load(f='iris_model.pth'))
"""