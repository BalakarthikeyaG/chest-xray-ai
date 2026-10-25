import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dataset import ChestXRayDataset

# Test loading from train split
print("=" * 60)
print("Testing ChestXRayDataset with chest_xray dataset")
print("=" * 60)

data_dir = Path(__file__).parent / 'data' / 'chest_xray'
print(f"\nData directory: {data_dir}")
print(f"Directory exists: {data_dir.exists()}\n")

# Test train split
try:
    print("Loading TRAIN split...")
    train_dataset = ChestXRayDataset(
        data_dir=str(data_dir),
        split='train',
        image_size=(224, 224)
    )
    print(f"✓ Train dataset loaded successfully")
    print(f"  Total images: {len(train_dataset)}")
    print(f"  Classes: {train_dataset.class_names}")
    print(f"  Class to index: {train_dataset.class_to_idx}\n")
    
    # Load first 3 images
    print("Loading first 3 images from train split:")
    for idx in range(min(3, len(train_dataset))):
        image, label = train_dataset[idx]
        class_name = train_dataset.class_names[label]
        print(f"  Image {idx}: shape={tuple(image.shape)}, label={label} ({class_name})")
    
except Exception as e:
    print(f"✗ Error loading train split: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)

# Test test split
try:
    print("Loading TEST split...")
    test_dataset = ChestXRayDataset(
        data_dir=str(data_dir),
        split='test',
        image_size=(224, 224)
    )
    print(f"✓ Test dataset loaded successfully")
    print(f"  Total images: {len(test_dataset)}")
    print(f"  Classes: {test_dataset.class_names}\n")
    
    # Load first 3 images
    print("Loading first 3 images from test split:")
    for idx in range(min(3, len(test_dataset))):
        image, label = test_dataset[idx]
        class_name = test_dataset.class_names[label]
        print(f"  Image {idx}: shape={tuple(image.shape)}, label={label} ({class_name})")
        
except Exception as e:
    print(f"✗ Error loading test split: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)

# Test val split
try:
    print("Loading VAL split...")
    val_dataset = ChestXRayDataset(
        data_dir=str(data_dir),
        split='val',
        image_size=(224, 224)
    )
    print(f"✓ Val dataset loaded successfully")
    print(f"  Total images: {len(val_dataset)}")
    print(f"  Classes: {val_dataset.class_names}\n")
    
    # Load first 3 images
    print("Loading first 3 images from val split:")
    for idx in range(min(3, len(val_dataset))):
        image, label = val_dataset[idx]
        class_name = val_dataset.class_names[label]
        print(f"  Image {idx}: shape={tuple(image.shape)}, label={label} ({class_name})")
        
except Exception as e:
    print(f"✗ Error loading val split: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
