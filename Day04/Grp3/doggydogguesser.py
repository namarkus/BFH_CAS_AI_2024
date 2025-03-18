# DoggyDogGuesser.py
# DoggyDogImageClassification
## Convolutional Neuronal Networks

#### Generelle Konfigurationen
import os
import math
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm  # Fortschrittsanzeige
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# Paths
DATA_PATH = 'C:/Users/namar/Documents/2024_BFH_AI/04_Teamwork/Day_04/dog_data'
BEST_MODEL_PATH = 'C:/Users/namar/Documents/2024_BFH_AI/04_Teamwork/Day_04/best_doggydog_model.pth'

# Image Parameters
IMAGE_EDGE_LENGTH = 224

# Hyperparameter
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
PATIENCE = 5  # Für Early Stopping

def prepare_dataloaders(batch_size=5, edge_length=IMAGE_EDGE_LENGTH):

    transform = transforms.Compose([transforms.Resize(edge_length),
        transforms.CenterCrop(edge_length),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    data_transforms = {
        'train': transform,
        'test': transform,
        'valid': transform
    }

    generator = torch.Generator() # Zufallszahlengenerator mit
    generator.manual_seed(42)

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_PATH, x), data_transforms[x]) for x in ['train', 'valid', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, generator=generator, num_workers=0) for x in ['train', 'valid', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    class_names = image_datasets['train'].classes
    return dataloaders, class_names


def show_sample_images(loader, class_names, num_images=5, model=None):
    """
    Zeigt zufällige Bilder aus dem Datensatz mit ihren tatsächlichen Klassennamen.
    Falls ein Modell angegeben wird, zeigt es zusätzlich die vorhergesagte Klasse an.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels = next(iter(loader))
    images, labels = images[:num_images], labels[:num_images]
    
    if model:
        model.to(device)
        model.eval()
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
        images = images.cpu()
    
    images = images.permute(0, 2, 3, 1).numpy()
    images = (images * 0.5) + 0.5  # Denormalisieren
    
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        true_label = class_names[labels[i].item()]
        if model:
            pred_label = class_names[predictions[i].item()]
            color = "green" if true_label == pred_label else "red"
            axes[i].set_title(f"GT: {true_label}\nPred: {pred_label}", color=color)
        else:
            axes[i].set_title(f"{true_label}")
    plt.show()


class DogBreedGuesser(nn.Module):
    def __init__(self, num_classes=71):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) # in_channels, out_channels, kernel_size
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # The output size after convolutions and pooling is calculated as follows:
        # ((224 - 5 + 1) / 2) - 5 + 1 / 2 # (Image size - kernel size + 1) / pooling size
        # This results in 53 and therefore (16, 53, 53) after second convolution
        # 16 * 53 * 53 = 45056 features in total, therefore in features changed to that value
        self.fc1 = nn.Linear(16 * 53 * 53, 120)  # Adjust input size to match the output of convolutional layers
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) # Apply pooling after the second convolutional layer
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PretrainedDogBreedGuesser(nn.Module):
    def __init__(self, num_classes=71):
        super(PretrainedDogBreedGuesser, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)        
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

class ImprovedDogBreedClassifier(nn.Module):
    def __init__(self, num_classes=71):
        super(ImprovedDogBreedClassifier, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fourth convolutional block
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate the size after convolutions and pooling
        # 224 -> 112 -> 56 -> 28 -> 14
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Weight initialization
        self._initialize_weights()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# Parametriesiertes Lernen
def train(model, train_loader, val_loader, criterion, optimizer, epochs, model_path=BEST_MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0    
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False)
        
        for i, data in enumerate(train_loader, 0):
            images, labels = data # get the inputs; data is a list of [inputs, labels
            images, labels = images.to(device), labels.to(device)
       
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Add gradient clipping here
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_bar.set_postfix(loss=running_loss / total, acc=100 * correct / total)
            # print statistics
#            if i % 200 == 0:    # print every 2000 mini-batches
#                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#                running_loss = 0.0
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validierung
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for j, data in enumerate(val_loader, 0):
                images, labels = data # get the inputs; data is a list of [inputs, labels
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Early Stopping & Modell speichern
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"Bestes Modell gespeichert unter: {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early Stopping aktiviert!")
                break

def evaluate(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    test_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            images, labels = data # get the inputs; data is a list of [inputs, labels
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total if total > 0 else 0
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)


def predict_single_image(model, image_path, class_names, top_k=5):
    """
    Lädt ein einzelnes Bild, zeigt es an und gibt die vorhergesagte Klasse sowie die wahrscheinlichsten Klassen zurück.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_classes = torch.topk(probabilities, top_k, dim=1)
    
    top_probs = top_probs.squeeze().cpu().numpy()
    top_classes = top_classes.squeeze().cpu().numpy()
    
    predicted_class = class_names[top_classes[0]]
    
    # Visualisierung
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].axis('off')
    ax[0].set_title("Eingabebild")
    
    ax[1].text(0.5, 1, f"{predicted_class} ({top_probs[0]:.4f})", fontsize=14, ha='center', va='top', weight='bold')

    if top_probs[1] < 0.05:
        ax[1].text(0.5, 0.8, "Keine weiteren plausiblen Rassen gefunden.", fontsize=10, ha='center', va='top')
    else:
        for i in range(1, top_k):
            ax[1].text(0.5, 0.8 - i * 0.15, f"{class_names[top_classes[i]]}: {top_probs[i]:.4f}", fontsize=10, ha='center', va='top')
    
    ax[1].axis('off')
    plt.show()
    
    return predicted_class, [(class_names[top_classes[i]], top_probs[i]) for i in range(top_k)]


if __name__ == "__main__":

    dataloaders, dog_breeds = prepare_dataloaders(batch_size=BATCH_SIZE, edge_length=224)
    dog_breed_count = len(dog_breeds)
    trainloader = dataloaders['train']
    validloader = dataloaders['valid']
    testloader = dataloaders['test']

    show_sample_images(dataloaders['test'], dog_breeds)

    model = ImprovedDogBreedClassifier(num_classes=dog_breed_count)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model_save_path = 'C:/Users/namar/Documents/2024_BFH_AI/04_Teamwork/Day_04/best_pretrained_doggydog_model.pth'
    train(model, trainloader, validloader, criterion, optimizer, EPOCHS, model_save_path)

    # bestes Modell lesen
    model = PretrainedDogBreedGuesser(num_classes=dog_breed_count)
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    evaluate(model, testloader)

    # Beispielbilder anzeigen
    show_sample_images(testloader, dog_breeds, 8, model=model)

    # Beispielbild raten
    new_img_path = "C:/Users/namar/Documents/2024_BFH_AI/04_Teamwork/Day_04/new_dog_data/Berner-Sennenhund_small.jpg"
    new_img_path = "C:/Users/namar/Documents/2024_BFH_AI/04_Teamwork/Day_04/new_dog_data/Hero_Dog_small.jpg"
    predict_single_image(model, image_path = new_img_path, class_names = dog_breeds)
