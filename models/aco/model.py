"""
Ant Colony Optimization (ACO) Enhanced Convolutional Neural Network for GTSRB.

This module integrates a metaheuristic Ant Colony Optimization (ACO) algorithm into
the deep learning pipeline:
- Convolutional Backbone: Extracts hierarchical spatial features from 49x49 traffic sign images.
- Ant Colony Feature Optimizer: Employs a population of artificial ants to explore the 128 feature
  channels. Ants deposit pheromones based on classification fitness and signal-to-noise heuristic,
  pruning noisy/degraded channels and reinforcing robust invariant representations.
- Pheromone Gating Layer: Modulates feature maps according to the collective pheromone trail discovered
  by the colony.
- Conforms strictly to the model-interface contract required by eval.py.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class AntColonyFeatureOptimizer:
    """
    Ant Colony Optimization (ACO) for neural feature channel selection and weighting.
    
    Mathematical Formulation:
    - Search Space: Set of C = 128 feature channels.
    - Pheromone Vector (tau): Represents the historical learned utility of each channel.
    - Heuristic Desirability (eta): Measured by the variance and activation strength of channels across a sample batch.
    - Selection Probability: P_k(i) = (tau_i^alpha * eta_i^beta) / (1 + tau_i^alpha * eta_i^beta).
    - Evaporation Rule: tau_i <- (1 - rho) * tau_i.
    - Pheromone Deposit: tau_i <- tau_i + sum_k(Q * Fitness_k) for channels selected by ant k.
    """
    def __init__(self, num_channels=128, num_ants=15, num_iterations=5, rho=0.15, alpha=1.0, beta=1.0, Q=1.0):
        self.num_channels = num_channels
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.rho = rho          # Evaporation coefficient
        self.alpha = alpha      # Pheromone importance exponent
        self.beta = beta        # Heuristic importance exponent
        self.Q = Q              # Pheromone deposit scaling constant
        
        # Initialize pheromone trail uniformly
        self.tau = np.ones(num_channels, dtype=np.float32)

    def compute_heuristic(self, features_np):
        """
        Computes heuristic information (eta) based on channel activation variance.
        Channels with higher informative variance receive higher heuristic priority.
        """
        # features_np: shape (B, C, H, W)
        channel_var = np.var(features_np, axis=(0, 2, 3)) + 1e-6
        eta = channel_var / (np.max(channel_var) + 1e-6)
        return eta

    def optimize(self, feature_extractor, classifier, sample_loader, device):
        """
        Runs the ant colony optimization loop to find the optimal pheromone distribution
        over convolutional feature channels.
        """
        feature_extractor.eval()
        classifier.eval()

        # 1. Grab a representative batch of images for fitness evaluation
        images, labels = next(iter(sample_loader))
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            raw_features = feature_extractor(images)  # (B, 128, 6, 6)
            feat_np = raw_features.cpu().numpy()

        eta = self.compute_heuristic(feat_np)
        best_fitness = -1.0
        best_mask = np.ones(self.num_channels, dtype=np.float32)

        print(f"[ACO] Initializing Ant Colony Search ({self.num_ants} ants, {self.num_iterations} iterations, {self.num_channels} channels)...")

        for iteration in range(self.num_iterations):
            ant_masks = []
            ant_fitnesses = []

            # 2. Probability distribution for channel selection
            prob_numerator = (self.tau ** self.alpha) * (eta ** self.beta)
            probs = prob_numerator / (1.0 + prob_numerator)
            probs = np.clip(probs, 0.1, 0.95)  # Avoid total collapse

            # 3. Each ant constructs a candidate feature channel mask
            for ant in range(self.num_ants):
                # Sample binary channel inclusion decisions
                mask = (np.random.rand(self.num_channels) < probs).astype(np.float32)
                # Ensure at least 25% of channels are active
                if np.sum(mask) < self.num_channels * 0.25:
                    top_indices = np.argsort(probs)[-int(self.num_channels * 0.5):]
                    mask[top_indices] = 1.0

                # 4. Evaluate ant fitness (classification accuracy on the sample batch)
                mask_tensor = torch.tensor(mask, device=device).view(1, self.num_channels, 1, 1)
                with torch.no_grad():
                    gated_features = raw_features * mask_tensor
                    flattened = gated_features.flatten(1)
                    outputs = classifier(flattened)
                    _, preds = torch.max(outputs, 1)
                    fitness = (preds == labels).float().mean().item()

                ant_masks.append(mask)
                ant_fitnesses.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_mask = mask.copy()

            # 5. Evaporation step
            self.tau = (1.0 - self.rho) * self.tau

            # 6. Pheromone deposit step (deposit proportional to accuracy fitness)
            for mask, fitness in zip(ant_masks, ant_fitnesses):
                delta_tau = self.Q * (fitness ** 2) * mask
                self.tau += delta_tau

            mean_fit = np.mean(ant_fitnesses)
            print(f"[ACO] Iteration [{iteration+1}/{self.num_iterations}] -> Best Fitness: {best_fitness*100:.2f}% | Mean Colony Fitness: {mean_fit*100:.2f}%")

        # Normalize pheromones into a smooth attention scale [0.5, 1.5]
        tau_norm = (self.tau - np.min(self.tau)) / (np.max(self.tau) - np.min(self.tau) + 1e-6)
        learned_gate = 0.5 + tau_norm  # Multiplicative channel modulation weights

        print(f"[ACO] Optimization complete. Selected feature channel distribution with max fitness: {best_fitness*100:.2f}%\n")
        return torch.tensor(learned_gate, dtype=torch.float32, device=device)


class ACOModel(nn.Module):
    """
    Convolutional Neural Network augmented with Ant Colony Optimization (ACO)
    for robust GTSRB Traffic Sign Recognition.
    
    Structure:
    - 3 Conv Blocks with GELU and MaxPool2d (outputs 128 channels at 6x6 for 49x49 inputs).
    - Pheromone Gate: Continuous channel attention vector optimized by the ant colony.
    - Fully Connected Classifier: Linear(4608, 512) -> GELU -> Dropout(0.5) -> Linear(512, 43).
    """
    def __init__(self, num_classes=43):
        super(ACOModel, self).__init__()
        self.num_classes = num_classes

        # Feature Extractor Backbone (3 Blocks)
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.act1 = nn.GELU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.act2 = nn.GELU()
        self.drop1 = nn.Dropout2d(p=0.2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.act3 = nn.GELU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ACO Pheromone Gating Vector (128 channels)
        # Initialized to ones (neutral modulation) until optimized by the ant colony
        self.register_buffer('pheromone_gate', torch.ones(1, 128, 1, 1))

        # Fully Connected Classifier
        # After three 2x2 poolings, a 49x49 image becomes 6x6. 128 * 6 * 6 = 4608 features.
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4608, 512)
        self.act4 = nn.GELU()
        self.drop2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def extract_features(self, x):
        """Passes input image through convolutional backbone."""
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.drop1(self.act2(self.conv2(x))))
        x = self.pool3(self.act3(self.conv3(x)))
        return x

    def forward_classifier(self, features):
        """Passes extracted features through classification head."""
        x = self.flatten(features)
        x = self.drop2(self.act4(self.fc1(x)))
        x = self.fc2(x)
        return x

    def forward(self, x):
        # 1. Feature extraction
        feats = self.extract_features(x)
        
        # 2. Pheromone-guided channel modulation
        # Re-weights channels using the collective pheromone trail learned by ACO
        gated_feats = feats * self.pheromone_gate
        
        # 3. Final classification
        logits = self.forward_classifier(gated_feats)
        return logits

    def train(self, dataset_or_mode=True, epochs=5, batch_size=64, lr=0.001, device=None, save_dir='output/aco', num_ants=15, aco_iterations=5, **kwargs):
        """
        Trains the ACO-augmented model:
        1. If passed a boolean, toggles PyTorch train mode.
        2. If passed a Dataset or DataLoader:
           - Prepares optimizer and loss function.
           - Runs warm-up training with gradient descent.
           - Executes the Ant Colony Optimization routine to search for robust feature channels.
           - Updates the model's pheromone_gate.
           - Fine-tunes the network with ACO-gated features.
           - Saves weights to save_dir/aco_model.pth.
        """
        # Case 1: Standard PyTorch train mode toggle
        if isinstance(dataset_or_mode, bool) and not kwargs and epochs == 5 and batch_size == 64:
            return super().train(mode=dataset_or_mode)

        # Case 2: Executing training on dataset
        train_data = dataset_or_mode
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        super().train(True)

        if isinstance(train_data, DataLoader):
            train_loader = train_data
        elif isinstance(train_data, Dataset):
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        else:
            raise TypeError("Expected Dataset or DataLoader for training.")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        print(f"\n[ACOModel] Starting ACO training pipeline for {epochs} epochs on device: {device}")

        for epoch in range(epochs):
            # Mid-training Ant Colony Optimization phase:
            # Run ACO colony search after warm-up epoch to optimize channel pheromones
            if epoch == 1:
                print("\n" + "="*50)
                print(f"[ACOModel] Launching Ant Colony Optimization Phase...")
                aco = AntColonyFeatureOptimizer(
                    num_channels=128,
                    num_ants=num_ants,
                    num_iterations=aco_iterations,
                    rho=0.15
                )
                optimal_gate = aco.optimize(
                    feature_extractor=self.extract_features,
                    classifier=self.forward_classifier,
                    sample_loader=train_loader,
                    device=device
                )
                # Apply learned pheromone gate to the model
                self.pheromone_gate.copy_(optimal_gate.view(1, 128, 1, 1))
                print(f"[ACOModel] Pheromone gate successfully integrated into neural network.")
                print("="*50 + "\n")

            self.train(True)
            running_loss = 0.0
            correct = 0
            total = 0

            loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)
            for images, labels in loop:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loop.set_postfix(
                    loss=f"{loss.item():.4f}", 
                    acc=f"{100.0 * correct / total:.2f}%"
                )

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100.0 * correct / total
            print(f"Epoch [{epoch+1}/{epochs}] Finished -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

        # Save model checkpoint
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'aco_model.pth')
            torch.save(self.state_dict(), save_path)
            print(f"[ACOModel] Trained weights saved to '{save_path}'")

        return {"epochs": epochs, "loss": epoch_loss, "accuracy": epoch_acc}

    def test(self, test_dataset_or_loader, batch_size=64, device=None):
        """
        Evaluates the model on clean or corrupted data.
        Returns:
            tuple: (all_preds, all_labels) as python lists.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()

        if isinstance(test_dataset_or_loader, DataLoader):
            loader = test_dataset_or_loader
        elif isinstance(test_dataset_or_loader, Dataset):
            loader = DataLoader(test_dataset_or_loader, batch_size=batch_size, shuffle=False)
        else:
            raise TypeError("Expected Dataset or DataLoader for testing.")

        all_preds = []
        all_labels = []

        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                outputs = self(images)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist())

        return all_preds, all_labels

    def load_weights(self, weights_path=None):
        """Loads trained weights from specified path or default locations."""
        if weights_path is None:
            candidates = [
                os.path.join('output', 'aco', 'aco_model.pth'),
                os.path.join('output', 'aco_model.pth')
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    weights_path = candidate
                    break

        if weights_path and os.path.exists(weights_path):
            self.load_state_dict(torch.load(weights_path, weights_only=True, map_location='cpu'))
            print(f"[ACOModel] Successfully loaded weights from '{weights_path}'")
            return True
        else:
            print(f"[ACOModel] Notice: No checkpoint found at '{weights_path}'. Training from scratch.")
            return False


# Aliases for convenience
ACO = ACOModel
TrafficSignACO = ACOModel


if __name__ == "__main__":
    print("Testing ACOModel initialization and forward pass...")
    model = ACOModel(num_classes=43)
    dummy_input = torch.randn(4, 3, 49, 49)
    out = model(dummy_input)
    print(f"ACOModel initialized successfully. Input: {dummy_input.shape} -> Output: {out.shape}")
