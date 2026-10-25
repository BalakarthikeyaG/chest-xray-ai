"""
Quick end-to-end training test: 1 epoch on 200 images.
Tests full pipeline: data loading → model → training loop → evaluation.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.utils.data as data_utils

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dataset import ChestXRayDataset, get_transforms
from model import create_model
from train import Trainer


def run_quick_train():
    """Run 1 epoch training on 200 images subset."""
    
    print("\n" + "=" * 70)
    print("QUICK TRAINING TEST: 1 Epoch on 200-Image Subset")
    print("=" * 70)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Device: {device}")
    
    # Data directory
    data_dir = Path(__file__).parent / 'data' / 'chest_xray'
    print(f"✓ Data directory: {data_dir}")
    
    # Load datasets
    print("\n[1/4] Loading datasets...")
    transforms_dict = get_transforms(image_size=(224, 224), augment=True)
    
    try:
        train_dataset_full = ChestXRayDataset(
            data_dir=str(data_dir),
            split='train',
            image_size=(224, 224),
            transform=transforms_dict['train']
        )
        
        test_dataset = ChestXRayDataset(
            data_dir=str(data_dir),
            split='test',
            image_size=(224, 224),
            transform=transforms_dict['test'],
            class_names=train_dataset_full.class_names
        )
        
        print(f"  ✓ Full train dataset: {len(train_dataset_full)} images")
        print(f"  ✓ Test dataset: {len(test_dataset)} images")
        
    except Exception as e:
        print(f"  ✗ Failed to load datasets: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create subset: limit to 200 training images
    print("\n[2/4] Creating 200-image training subset...")
    subset_size = min(200, len(train_dataset_full))
    train_dataset = data_utils.Subset(train_dataset_full, range(subset_size))
    print(f"  ✓ Using {len(train_dataset)} images for training")
    
    # Create data loaders
    print("\n[3/4] Creating data loaders...")
    try:
        train_loader = data_utils.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = data_utils.DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"  ✓ Train loader: {len(train_loader)} batches")
        print(f"  ✓ Test loader: {len(test_loader)} batches")
        
    except Exception as e:
        print(f"  ✗ Failed to create data loaders: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create model
    print("\n[4/4] Creating model...")
    try:
        model = create_model(
            num_classes=2,
            pretrained=True,
            freeze_backbone=True,
            device=device
        )
        model = model.to(device)
        
        trainable_params = model.get_trainable_params()
        total_params = model.get_total_params()
        print(f"  ✓ Model: DenseNet121")
        print(f"  ✓ Trainable parameters: {trainable_params:,}")
        print(f"  ✓ Total parameters: {total_params:,}")
        
    except Exception as e:
        print(f"  ✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Run 1 epoch training
    print("\n" + "=" * 70)
    print("RUNNING 1-EPOCH TRAINING...")
    print("=" * 70)
    
    try:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=1,
            learning_rate=0.001,
            device=device,
            checkpoint_dir="checkpoints_test"
        )
        
        trainer.train()
        
        print("\n" + "=" * 70)
        print("✓ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nPipeline Status:")
        print("  ✓ Data loading: OK")
        print("  ✓ Model creation: OK")
        print("  ✓ Training loop: OK")
        print("  ✓ Evaluation: OK")
        print("\n✓ Full pipeline is working end-to-end without errors!")
        print("=" * 70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 70)
        return False


if __name__ == "__main__":
    success = run_quick_train()
    sys.exit(0 if success else 1)
