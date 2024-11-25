from neural_network import ImageRecognition
from utils import load_images_from_folder
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import glob 
import numpy as np
import torch

def main():
    images_array = np.empty((0, 256, 256, 3))
    labels = []
    for i in glob.glob('../../data_exercise/*'):
        images_array_, labels_ = load_images_from_folder(i, target_size=(256, 256), label = i.split('/')[1])
        images_array = np.concatenate((images_array, images_array_), axis=0)
        labels = labels + labels_

    images_array_esc = images_array / 255.


    x_train, x_test, y_train, y_test = train_test_split(images_array_esc, labels, test_size=0.3, random_state=123)
    x_training, x_val, y_training, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=123)

    y_train_cat = pd.get_dummies(y_training).values * 1
    y_test_cat = pd.get_dummies(y_test).values * 1
    y_val_cat = pd.get_dummies(y_val).values * 1

    # Prepare datasets and dataloaders
    batch_size = 32

    # Assuming x_train, y_train_cat, x_test, y_test_cat are numpy arrays
    x_train_tensor = torch.tensor(x_training, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert to NCHW
    y_train_tensor = torch.tensor(y_train_cat.argmax(axis=1), dtype=torch.long)  # Convert one-hot to class indices

    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert to NCHW
    y_val_tensor = torch.tensor(y_val_cat.argmax(axis=1), dtype=torch.long)  # Convert one-hot to class indices


    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert to NCHW
    y_test_tensor = torch.tensor(y_test_cat.argmax(axis=1), dtype=torch.long)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    model = ImageRecognition(num_classes=6)
    model.to(device)

    model.fit(criterion, train_loader, val_loader, device, num_epochs=10)
    model.save_model("custom_vgg16.pth")
    return "Model fitted successfully"

if __name__ == "__main__":
    main()