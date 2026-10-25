"""
Test script to demonstrate the training loop with dummy data.
This creates synthetic data to show how the training works.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.model import create_model


def create_dummy_dataloaders(num_samples=100, batch_size=16):
    """
    Create dummy dataloaders for testing.
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create synthetic X-ray images (batch, channels, height, width)
    X_train = torch.randn(num_samples, 3, 224, 224)
    y_train = torch.randint(0, 2, (num_samples,))
    
    X_test = torch.randn(num_samples // 4, 3, 224, 224)
    y_test = torch.randint(0, 2, (num_samples // 4,))
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_with_explanation(num_epochs=3):
    """
    Training loop with detailed explanations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Step 1: Create dummy data loaders
    print("="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    train_loader, test_loader = create_dummy_dataloaders(num_samples=100, batch_size=16)
    print(f"✓ Created {len(train_loader)} training batches")
    print(f"✓ Created {len(test_loader)} test batches")
    print(f"✓ Each batch: 16 images of shape (3, 224, 224)\n")
    
    # Step 2: Create model
    print("="*60)
    print("STEP 2: CREATING MODEL")
    print("="*60)
    model = create_model(num_classes=2, pretrained=False, freeze_backbone=False, device=device)
    print(f"✓ Model created: DenseNet121")
    print(f"✓ Total parameters: {model.get_total_params():,}")
    print(f"✓ Trainable parameters: {model.get_trainable_params():,}\n")
    
    # Step 3: Setup loss function and optimizer
    print("="*60)
    print("STEP 3: SETUP TRAINING")
    print("="*60)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    print("✓ Loss function: CrossEntropyLoss")
    print("✓ Optimizer: Adam (lr=0.001)")
    print("✓ Scheduler: ReduceLROnPlateau (adaptive learning rate)\n")
    
    # Step 4: Training loop
    print("="*60)
    print("STEP 4: TRAINING LOOP")
    print("="*60)
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # ============ TRAINING PHASE ============
        print("\n📚 TRAINING PHASE:")
        print("-" * 60)
        
        model.train()  # Set model to training mode
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move data to device (GPU/CPU)
            images = images.to(device)
            labels = labels.to(device)
            
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}")
            print(f"    Input shape: {images.shape}")
            print(f"    Labels shape: {labels.shape}")
            
            # ========== FORWARD PASS ==========
            # Pass images through the model to get predictions
            outputs = model(images)  # Shape: (batch_size, num_classes)
            print(f"    Model output shape: {outputs.shape}")
            
            # Compute loss between predictions and ground truth
            loss = criterion(outputs, labels)
            print(f"    Loss: {loss.item():.4f}")
            
            # ========== BACKWARD PASS ==========
            # Clear previous gradients
            optimizer.zero_grad()
            
            # Compute gradients (backpropagation)
            loss.backward()
            
            # Update model weights using computed gradients
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"\n  Average Training Loss: {avg_train_loss:.4f}\n")
        
        # ============ EVALUATION PHASE ============
        print("\n🔍 EVALUATION PHASE:")
        print("-" * 60)
        
        model.eval()  # Set model to evaluation mode
        all_preds = []
        all_labels = []
        total_test_loss = 0.0
        
        with torch.no_grad():  # Disable gradient computation
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass (no gradients)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()
                
                # Get predictions (argmax of class probabilities)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        avg_test_loss = total_test_loss / len(test_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        
        print(f"  Test Loss: {avg_test_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        
        # Step scheduler
        scheduler.step(avg_test_loss)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    train_with_explanation(num_epochs=3)
