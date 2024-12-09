from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torch.functional import F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import plotly.graph_objects as go

class Iris_dataset(Dataset):
    def  __init__(self, in_features, labels):
        super().__init__()
        self.in_features = in_features
        self.labels = labels
        self.transform_in_features = lambda x: torch.tensor(x, dtype=torch.float32)
        self.transform_labels = lambda x: torch.tensor(x, dtype=torch.long)
    
    def __getitem__(self, index):
        return self.transform_in_features(self.in_features[index]), self.transform_labels(self.labels[index])
    
    def __len__(self):
        return len(self.in_features)

class ANN(nn.Module):
    def __init__(self, input_neuron, hidden_neurons, output_neuron):
        super().__init__()
        self.input_layer = nn.Linear(in_features=input_neuron,out_features=hidden_neurons)
        self.hidden_layer_1 = nn.Linear(in_features=hidden_neurons,out_features=output_neuron)
    
    def forward(self, x):
        output_from_input_layer = self.input_layer(x)
        output_from_hidden_layer = F.relu(output_from_input_layer)
        logits = self.hidden_layer_1(output_from_hidden_layer)
        return logits

def load_and_split_data():
    in_features, labels = load_iris(return_X_y=True)

    scaler = StandardScaler()
    in_features = scaler.fit_transform(in_features)

    train_x, test_x, train_y, test_y = train_test_split(in_features, 
                                                        labels, 
                                                        test_size=0.2, 
                                                        random_state=123, 
                                                        stratify=labels)

    print(f"Size of training set: in_features {train_x.shape}, labels {train_y.shape}")
    print(f"Size of testing set: in_features {test_x.shape}, labels {test_y.shape}")

    return train_x, test_x, train_y, test_y

def generate_loader():

    train_x, test_x, train_y, test_y = load_and_split_data()

    train_dataset = Iris_dataset(in_features=train_x, labels=train_y)
    test_dataset = Iris_dataset(in_features=test_x, labels=test_y)

    train_loader = DataLoader(dataset=train_dataset, batch_size=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=4)

    return train_loader, test_loader

def train(train_loader, test_loader, model, loss_fn, optimizer, num_epochs):
    train_losses = []
    val_losses = []

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

if __name__ == "__main__":

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} now!")
    
    model = ANN(4,10,3).to(device)
    print(model)

    for parameter in model.parameters():
        print(parameter)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    train_loader, test_loader = generate_loader()
    num_epochs = 50
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