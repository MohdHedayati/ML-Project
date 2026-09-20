import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset_loader import GTSRBDataset
from model import TrafficSignNet

def train_baseline():
    # 1. Setup device and hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    batch_size = 64
    epochs = 5 # Keeping it low just to make sure it works first
    learning_rate = 0.001

    # 2. Prepare Data
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    train_dataset = GTSRBDataset(csv_file='data/Train.csv', root_dir='data', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 3. Initialize Model, Loss, and Optimizer
    model = TrafficSignNet(num_classes=43).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Training Loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Training Accuracy: {epoch_acc:.2f}%")

    # 5. Save the trained model
    torch.save(model.state_dict(), 'baseline_model.pth')
    print("Training complete. Model saved to 'baseline_model.pth'.")

if __name__ == "__main__":
    train_baseline()