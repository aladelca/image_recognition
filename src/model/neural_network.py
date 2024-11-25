import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import warnings 
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import numpy as np  
warnings.filterwarnings("ignore")

vgg_base = models.vgg16(pretrained=True)
for param in vgg_base.features.parameters():
    param.requires_grad = False  # Freeze the feature layers

class CustomVGG16(nn.Module):
    def __init__(self, num_classes):
        super(CustomVGG16, self).__init__()
        self.vgg_features = vgg_base.features
        self.custom_layers = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, padding=1),  # 'conv_1'
            nn.ReLU(),
            nn.MaxPool2d(2),  # 'maxpool_1'
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 'conv_2'
            nn.ReLU(),
            nn.MaxPool2d(2),  # 'maxpool_2'
            nn.Dropout(0.5),  # 'dropout_1'
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 'conv_3'
            nn.ReLU(),
            nn.MaxPool2d(2),  # 'maxpool_3'
        )
        self.flatten = nn.Flatten()
        self.fc1 = None  # This will be initialized dynamically
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.vgg_features(x)
        x = self.custom_layers(x)
        x = self.flatten(x)
        
        # Dynamically initialize fc1 based on the input shape
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 64).to(x.device)
        
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x
    

class ImageRecognition(CustomVGG16):
    def __init__(self, num_classes):
        super(ImageRecognition, self).__init__(num_classes)
        self.fitted = False

    def fit(self, criterion, train_loader, val_loader, device, num_epochs=30):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, verbose=True)
        best_accuracy = 0.0
        best_model_weights = None

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Validation phase
            self.eval()
            val_accuracy = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self(inputs)
                    _, preds = torch.max(outputs, 1)
                    val_accuracy += torch.sum(preds == labels).item()

            val_accuracy /= len(val_loader.dataset)
            scheduler.step(val_accuracy)

            # Save the best model weights
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_weights = self.state_dict()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Load best weights
        self.load_state_dict(best_model_weights)
        self.fitted = True
        return "Model fitted successfully"
    
    def predict(self, x_test_tensor, batch_size, device):
        
        self.eval()

        test_loader = DataLoader(TensorDataset(x_test_tensor), batch_size=batch_size, shuffle=False)

        # Store predictions
        all_predictions = []

        # Perform prediction
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs[0].to(device)  # Extract the input batch and move to device
                outputs = self(inputs)  # Forward pass
                _, preds = torch.max(outputs, 1)  # Get the predicted class
                all_predictions.extend(preds)  # Move to CPU and store

        # Convert predictions to numpy array for further use
        #all_predictions = np.array(all_predictions)
        return [i.item() for i in all_predictions]
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        return f"Model saved to {path}"
    
    def predict_from_load(self, x_test, weights, device):
        dummy_input = torch.zeros(1, 3, 256, 256).to(device)
        _ = self(dummy_input)
        self.load_state_dict(torch.load(weights, map_location=device))
        self.eval()
        with torch.no_grad():
            output = self(x_test.to(device))
            _, predicted_class = torch.max(output, 1)
        return predicted_class