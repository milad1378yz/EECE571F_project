import torch
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import ConcatDataset
import argparse
# Specify the folders you want to include
included_folders = [
        "Fetal brain_Trans-cerebellum",
        "Fetal brain_Trans-thalamic",
        "Fetal brain_Trans-ventricular",
        "Fetal abdomen",
        "Fetal femur",
        "Fetal thorax",
    ]


# Custom dataset class
class CustomDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.samples = [s for s in self.samples if os.path.basename(os.path.dirname(s[0])) in included_folders]
        self.imgs = self.samples

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# Load pre-trained ResNet model
print("Loading pre-trained ResNet model...")
# check if there is pre-trained weights
if os.path.isfile('resnet50.pth'):
    print("Loading pre-trained weights...")
    model = models.resnet50()
    model.load_state_dict(torch.load('resnet50.pth'))
else:
    print("No pre-trained weights found, loading default weights...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# save the model
torch.save(model.state_dict(), 'resnet50.pth')
# Load dataset
print("Loading dataset...")
# Create an argument parser
parser = argparse.ArgumentParser(description='Train classifier')

# Add the arguments
parser.add_argument('--data_dir', type=str, help='Path to the data directory',default='/scratch/st-sdena-1/miladyz/EECE571F_project/seprated_data')
parser.add_argument('--new_data_dir', type=str, help='Path to the new data directory',default=None)

# Parse the arguments
args = parser.parse_args()

# Access the directories
data_dir = args.data_dir
new_data_dir = args.new_data_dir

# Print the directories
print(f"data_dir: {data_dir}")
print(f"new_data_dir: {new_data_dir}")
print("Loading dataset...")
dataset = CustomDataset(root=data_dir, transform=transform)
# Load the dataset from the new directory
print("Loading generated dataset...")
if new_data_dir is not None:
    new_dataset = CustomDataset(root=new_data_dir, transform=transform)
    # Combine the original dataset and the new dataset
    dataset = ConcatDataset([dataset, new_dataset])


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)



# Modify the model to fit 6 classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(included_folders))

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using {device} for training")
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Fine-tune the model
num_epochs = 60
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    # compute training accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy on training dataset: {100 * correct / total}%')

    # compute validation accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy on validation dataset: {100 * correct / total}%')


# Save the model
torch.save(model.state_dict(), 'resnet50_final_generated.pth')
