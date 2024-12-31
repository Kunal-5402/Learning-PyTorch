from sklearn.metrics import (accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             classification_report, 
                             confusion_matrix)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from typing import List, Callable
from dataclasses import dataclass, field

import warnings
warnings.filterwarnings('ignore')

class pollution_data(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.transform_x = lambda x: torch.tensor(x, dtype=torch.float32)
        self.transform_y = lambda y: torch.tensor(y, dtype=torch.long)

    def __getitem__(self, index):
        return self.transform_x(self.x[index]), self.transform_y(self.y[index])

    def __len__(self):
        return len(self.y)

@dataclass
class trainer():
    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    loss_fn: Callable
    optimizer: torch.optim.Optimizer
    num_epochs: int

    def train(self):

        for epoch in range(self.num_epochs):

            #setting model to train mode
            self.model.train()
            running_train_epoch_loss = 0.0

            for inputs, output in self.train_loader:

                y_pred = self.model(inputs)
                loss = self.loss_fn(y_pred, output)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_train_epoch_loss += loss.item()

            average_train_epoch_loss = running_train_epoch_loss/len(self.train_loader)

            #setting the model to eval to study val loss
            self.model.eval()
            running_validation_epoch_loss = 0.0

            with torch.no_grad():
                for inputs, output in self.val_loader:

                    y_pred = self.model(inputs)
                    loss = self.loss_fn(y_pred, output)

                    running_validation_epoch_loss += loss.item()
                
            average_validation_epoch_loss = running_validation_epoch_loss/len(self.val_loader)

            print(f"Epoch[{epoch}/{self.num_epochs}],\t Train Loss: {average_train_epoch_loss:.4f},\t Validation Loss: {average_validation_epoch_loss:.4f}")

class teacher(nn.Module):
    def __init__(self, num_classes, inputs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features = inputs, out_features = 128),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = 128),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = num_classes),
        )
    
    def forward(self, x):
        return self.network(x)

class student(nn.Module):
    def __init__(self, num_classes, inputs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features = inputs, out_features = 64),
            nn.ReLU(),
            nn.Linear(in_features = 64, out_features = num_classes)
        )
    def forward(self, x):
        return self.network(x)

def prediction(model, test_loader):
        model.eval()

        y = []      
        y_pred = [] 

        with torch.no_grad():
            for inputs, outputs in test_loader:

                predictions = model(inputs)
                predictions = torch.argmax(predictions, dim=1)

                y.extend(outputs.tolist())            
                y_pred.extend(predictions.tolist())

        return y, y_pred

def compute_metrics(y, y_pred):
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average="weighted"),
            "recall": recall_score(y, y_pred, average="weighted"),
            "f1_score": f1_score(y, y_pred, average="weighted"),
            "classification_report": classification_report(y, y_pred),
            "confusion_matrix": confusion_matrix(y, y_pred),
        }

        return metrics

def distillation_loss(teacher_logits, student_logits, true_labels, alpha = 0.5, temprature = 3.0):

    soft_teacher_prob = nn.functional.softmax(teacher_logits / temprature, dim=1)
    soft_student_prob = nn.functional.log_softmax(student_logits / temprature, dim=1)
    
    soft_loss = nn.functional.kl_div(soft_student_prob, soft_teacher_prob, reduction="batchmean") * (temprature ** 2)
    hard_loss = nn.functional.cross_entropy(student_logits, true_labels)

    KD_loss = alpha * soft_loss + (1 - alpha) * hard_loss

    return KD_loss

def KD_train(teacher_model, student_model, train_loader, num_epochs):

    optimizer = optim.Adam(student_model.parameters())

    for epoch in range(num_epochs):
        running_loss = 0.0
        teacher_model.eval()
        student_model.train()

        for x, y in train_loader:
        
            with torch.no_grad():
                teacher_logits = teacher_model(x)
            student_logits = student_model(x)
          
            optimizer.zero_grad()
            loss = distillation_loss(teacher_logits, student_logits, y)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

def main():
    df = pd.read_csv("pollution_dataset.csv")

    print(f"The data contains Nan values: {df.isna().values.any()}")
    print(f"The shape of the data : {df.shape}")
    print(f"\nDistribution of labels: \n{df['Air Quality'].value_counts()}")


    one_hot_encoder = LabelEncoder()
    normalizer = StandardScaler()

    x = normalizer.fit_transform(df.drop("Air Quality",axis=1))
    y = one_hot_encoder.fit_transform(df['Air Quality'])

    train_x, test_x, train_y, test_y = train_test_split(x, 
                                                        y, 
                                                        test_size=0.2, 
                                                        stratify=y, 
                                                        random_state=123, 
                                                        shuffle=True)

    train_data = pollution_data(train_x, train_y)
    test_data = pollution_data(test_x, test_y)

    train_loader = DataLoader(train_data, batch_size=16)
    test_loader = DataLoader(test_data, batch_size=16)

    #TEACHER MODEL TRAINING
    teacher_model = teacher(num_classes = 4, inputs = 9)
    total_params = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)

    print(teacher_model)
    print(f"Total trainable parameters: {total_params}")

    teacher_y, teacher_y_pred = prediction(teacher_model, test_loader)
    metrics = compute_metrics(teacher_y, teacher_y_pred)

    print(f"The accuracy of teacher before training: {metrics['accuracy']}")

    teacher_trainer = trainer(model = teacher_model,
                    optimizer = optim.Adam(teacher_model.parameters(), lr = 0.001),
                    num_epochs = 60,
                    train_loader = train_loader,
                    val_loader = test_loader,
                    loss_fn = torch.nn.CrossEntropyLoss())

    teacher_epoch_loss = teacher_trainer.train()
    teacher_y, teacher_y_pred = prediction(teacher_model, test_loader)
    metrics = compute_metrics(teacher_y, teacher_y_pred)
    print(f"The accuracy of teacher after training: {metrics['accuracy']}")

    #STUDENT MODEL TRAINING
    student_model = student(num_classes = 4, inputs = 9)
    total_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)

    print(student_model)
    print(f"Total trainable parameters: {total_params}")

    student_y, student_y_pred = prediction(student_model, test_loader)
    metrics = compute_metrics(student_y, student_y_pred)
    
    print(f"The accuracy of student before training: {metrics['accuracy']}")

    #KNOWLEDGE DISTILLATION
    KD_train(teacher_model, student_model, train_loader, 30)

    student_y, student_y_pred = prediction(student_model, test_loader)
    metrics = compute_metrics(student_y, student_y_pred)
    
    print(f"The accuracy of student after KD training: {metrics['accuracy']}")

if __name__ == "__main__":
    main()