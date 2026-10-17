import torch
import torch.nn as nn

class TrafficSignNet(nn.Module):
    def __init__(self, num_classes=43): # GTSRB has 43 traffic sign classes
        super(TrafficSignNet, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.act1 = nn.GELU() # Using GELU instead of standard ReLU for uniqueness
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.act2 = nn.GELU()
        self.drop1 = nn.Dropout2d(p=0.2) # Spatial dropout
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.act3 = nn.GELU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Classifier
        # After three 2x2 poolings, a 49x49 image becomes 6x6. 
        # 128 channels * 6 * 6 = 4608 flattened features
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4608, 512)
        self.act4 = nn.GELU()
        self.drop2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.drop1(self.act2(self.conv2(x))))
        x = self.pool3(self.act3(self.conv3(x)))
        x = self.flatten(x)
        x = self.drop2(self.act4(self.fc1(x)))
        x = self.fc2(x)
        return x

# Test the architecture with a dummy image
if __name__ == "__main__":
    model = TrafficSignNet(num_classes=43)
    # Create a fake batch of 5 images, 3 color channels, 49x49 pixels
    dummy_input = torch.randn(5, 3, 49, 49)
    output = model(dummy_input)
    
    print("Model initialized successfully!")
    print(f"Output shape (Batch Size, Classes): {output.shape}")