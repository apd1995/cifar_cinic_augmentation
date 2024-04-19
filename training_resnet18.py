import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageFolder
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from torchvision.models import resnet18
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import scipy.stats as stats
import wandb
import argparse
import json
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
import logging
import os

# Parsing command line arguments for the configuration file
parser = argparse.ArgumentParser(description='Train SqueezeNet on CIFAR-10 and evaluate on CINIC-10')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file')
args = parser.parse_args()

# Load hyperparameters from JSON file
with open(args.config, 'r') as f:
    hyperparams = json.load(f)

# Initialize Weights & Biases
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
wandb.init(project="cifar_cinic",
           entity="apd_stats",
           config=hyperparams,
           resume="allow",
           name=f"{current_datetime}")
checkpoint_path = "./model/resnet18/"+current_datetime
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    logging.info("Loading CIFAR-10 training data...")
    train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
    logging.info("CIFAR-10 training data loaded successfully.")
except Exception as e:
    logging.error("Failed to load CIFAR-10 training data: %s", e)

try:
    logging.info("Loading CIFAR-10 validation data...")
    val_data = CIFAR10(root='./data', train=False, download=True, transform=transform)
    logging.info("CIFAR-10 validation data loaded successfully.")
except Exception as e:
    logging.error("Failed to load CIFAR-10 validation data: %s", e)

# Load CINIC-10 dataset
try:
    logging.info("Loading CINIC-10 data...")
    cinic_data = ImageFolder(root='./data/CINIC_10/test', transform=transform)
    logging.info("CINIC-10 data loaded successfully.")
except Exception as e:
    logging.error("Failed to load CINIC-10 data: %s", e)

# Split CIFAR-10 data into training and validation sets
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

# Data loaders
batch_size = hyperparams['batch_size']
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
cinic_loader = DataLoader(cinic_data, batch_size=batch_size, shuffle=False)


# Load ResNet18 model, pretrained on ImageNet
model = resnet18(pretrained=False)  # Set pretrained=False to train from scratch
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)  # CIFAR-10 has 10 classes


# Move model to GPU and use DataParallel for multi-GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# Optimizer and loss function
optimizer = Adam(model.parameters(), lr=hyperparams['learning_rate'])
criterion = CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Function to evaluate the model
def evaluate_model(dataloader, model):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Training and validation loop
for epoch in range(hyperparams['num_epochs']):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    train_accuracy = evaluate_model(train_loader, model)
    val_accuracy = evaluate_model(val_loader, model)
    cinic_accuracy = evaluate_model(cinic_loader, model)
    
    # Log metrics to wandb
    wandb.log({'train_accuracy': train_accuracy,
               'validation_accuracy': val_accuracy,
               'CINIC-10_accuracy': cinic_accuracy,
               'learning_rate': scheduler.get_last_lr()[0]})
    
    # Checkpointing the model every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), checkpoint_path+f'/epoch_{epoch+1}.pth')
        logging.info(f"Checkpoint saved: {epoch+1}")

    # Step the scheduler
    scheduler.step()
    
# Finish Weights & Biases run
wandb.finish()
