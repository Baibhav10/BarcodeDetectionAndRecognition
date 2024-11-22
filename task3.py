# Author: Baibhav Shrestha
# Last Modified: 2024-10-5

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import torch.nn.utils.prune as prune
import torch.quantization
import matplotlib.pyplot as plt


# Preprocessing function with added Gaussian blur and Otsu's thresholding
def preprocess_image(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    _, img_thresh = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours and len(contours) > 0:
        x, y, w, h = cv2.boundingRect(contours[0])
        img_cropped = img_thresh[y:y+h, x:x+w]
    else:
        img_cropped = img_thresh
        print("No contours found, using full image.")
    
    img_resized = cv2.resize(img_cropped, (20, 20), interpolation=cv2.INTER_AREA)
    img_padded = np.pad(img_resized, ((4, 4), (4, 4)), mode='constant', constant_values=0)
    
    return img_padded

# Custom Preprocessing Class
class PreprocessWithPadding:
    def __call__(self, img):
        img = np.array(img)
        img = preprocess_image(img)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        return img

# TransformedDataset Class
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def load_pretrained_model(num_classes=10):
    # Loading the ResNet18 architecture without pretrained weights
    model = models.resnet18(weights=None)

    # Modifying the input layer to accept 1-channel images
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(
        2, 2), padding=(3, 3), bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')

    # Modifying the fully connected layer for 10 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    nn.init.zeros_(model.fc.bias)
    nn.init.normal_(model.fc.weight, 0, 0.01)

    return model



# Applying pruning to reduce model size
def apply_pruning(model):
    # Prune 30% of the weights in the fully connected layer (fc)
    prune.l1_unstructured(model.fc, name='weight', amount=0.3)
    
    return model

# Training the CNN Model with your own dataset and include validation
def train_cnn_model(dataset_path, model_save_path):
    transform_train = transforms.Compose([PreprocessWithPadding()])
    transform_test = transforms.Compose([PreprocessWithPadding()])
    
    full_dataset = datasets.ImageFolder(root=dataset_path)
    
    indices = list(range(len(full_dataset)))
    np.random.shuffle(indices)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    test_subset = Subset(full_dataset, test_indices)
    
    train_dataset = TransformedDataset(train_subset, transform=transform_train)
    val_dataset = TransformedDataset(val_subset, transform=transform_test)
    test_dataset = TransformedDataset(test_subset, transform=transform_test)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
    
    #Model to detect 0-9 digits
    model = load_pretrained_model(num_classes=10)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    model.train()
    for epoch in range(10):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = 0.0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        scheduler.step(val_loss)

    print("Training completed.")
    
    # Applying pruning before saving the model
    model = apply_pruning(model)
    
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return model

#Checking for existence of model
def get_or_train_model(dataset_path, model_path):
    model = load_pretrained_model(num_classes=10)

    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("No saved model found. Training a new model.")
        model = train_cnn_model(dataset_path, model_path)

    return model


# Using the Model for Inference on Barcode Images
def recognize_digit(model, img):
    img = preprocess_image(img)
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def process_barcodes(model, input_dir, output_dir):
    # List all folders in input_dir and filter those starting with 'barcode'
    barcode_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

    for folder in barcode_folders:
        print(f"Processing folder: {folder}")
        barcode_input_path = os.path.join(input_dir, folder)
        barcode_output_path = os.path.join(output_dir, folder)
        
        # Creating output directory for each barcode folder if it does not exist
        os.makedirs(barcode_output_path, exist_ok=True)
        
        images = sorted([img for img in os.listdir(barcode_input_path) if img.endswith('.png')])
        
        for img_name in images:
            img_path = os.path.join(barcode_input_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            digit = recognize_digit(model, img)
            
            output_txt_file = os.path.join(barcode_output_path, img_name.replace('.png', '.txt'))
            with open(output_txt_file, 'w') as f:
                f.write(str(digit))
            print(f"Processed {img_name} -> Recognized digit: {digit}")


def run_task3(image_path, config):
    input_dir = image_path
    output_dir = config.get('task3_output', os.path.join(os.getcwd(), 'output', 'task3'))
    model_path = config.get('task3_model_path', os.path.join(input_dir, 'trained_model.pth'))
    
    model = get_or_train_model(input_dir, model_path)
    process_barcodes(model, input_dir, output_dir)
    print("Task 3 processing completed.")
