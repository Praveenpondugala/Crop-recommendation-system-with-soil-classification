import os
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
base_dir = r'C:/Users/Harshith Y/Desktop/soil_images'  # Replace with your actual path
output_dir = r'C:/Users/Harshith Y/Desktop/soil_dataset'  # Where train/test will be created

# Step 1: Preprocessing and Train-Test Split
def prepare_dataset(base_dir, output_dir, test_size=0.2):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Clear previous split if exists
    
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Classes
    classes = ['Red', 'Black', 'Alluvial', 'Laterite']
    
    for class_name in classes:
        class_path = os.path.join(base_dir, class_name)
        print(f"Checking path: {class_path}")  # Debug
        if not os.path.exists(class_path):
            raise FileNotFoundError(f"Directory not found: {class_path}")
        images = os.listdir(class_path)
        images = [os.path.join(class_path, img) for img in images if img.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Split into train and test
        train_imgs, test_imgs = train_test_split(images, test_size=test_size, random_state=42)
        
        # Create class folders in train/test
        for split, img_list in [('train', train_imgs), ('test', test_imgs)]:
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for img in img_list:
                shutil.copy(img, split_class_dir)

# Step 2: Data Transformations (Enhanced Augmentation)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Step 3: Load and Split Dataset
prepare_dataset(base_dir, output_dir, test_size=0.2)

image_datasets = {
    'train': datasets.ImageFolder(os.path.join(output_dir, 'train'), transform=data_transforms['train']),
    'test': datasets.ImageFolder(os.path.join(output_dir, 'test'), transform=data_transforms['test'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=4)
}

# Dataset sizes and class names
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes
print(f"Class names: {class_names}")
print(f"Train size: {dataset_sizes['train']}, Test size: {dataset_sizes['test']}")

# Step 4: Load and Modify ResNet-50
model = models.resnet50(pretrained=True)

# Unfreeze layer4 and fc for fine-tuning
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Modify the final fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))  # 4 classes
model = model.to(device)

# Loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Step 5: Training Function with Scheduler
def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    train_losses = []
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / dataset_sizes['train']
        train_losses.append(epoch_loss)
        print(f'Train Loss: {epoch_loss:.4f}')
        scheduler.step()  # Adjust learning rate
    
    # Plot training loss
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.show()
    
    return model

# Step 6: Evaluation Function
def evaluate_model(model):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print("Confusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.show()
    
    return accuracy

# Step 7: Train and Evaluate
print("Starting training...")
model = train_model(model, criterion, optimizer, scheduler, num_epochs=20)
print("\nEvaluating model...")
accuracy = evaluate_model(model)

# Step 8: Save the model
torch.save(model.state_dict(), 'soil_resnet50_improved.pth')
print("Model saved as 'soil_resnet50_improved.pth'")