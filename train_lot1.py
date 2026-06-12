# train_lot1.py - Train model specifically on Lot1 images

import cv2
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib


def train_on_lot1():
    """Train model using only Lot1 images"""
    
    # Configuration
    excel_file = 'Lot1_md.xlsx'
    lot1_images_dir = 'lot1_images'  # Directory containing Lot1 images
    
    print("="*60)
    print("TRAIN MODEL ON LOT1 IMAGES")
    print("="*60)
    
    # Read Excel to get TXID list
    if not Path(excel_file).exists():
        print(f"❌ Excel file not found: {excel_file}")
        return
    
    df = pd.read_excel(excel_file)
    
    # Find TXID column
    txid_col = None
    for col in df.columns:
        if 'TXID' in str(col).upper():
            txid_col = col
            break
    
    if txid_col is None:
        print("❌ Could not find TXID column")
        return
    
    # Check which images exist and load them
    print(f"\n📖 Reading Lot1 data from {excel_file}")
    print(f"📂 Images directory: {lot1_images_dir}")
    
    training_images = []
    valid_count = 0
    invalid_count = 0
    
    for idx, row in df.iterrows():
        txid = str(row[txid_col])
        
        # Look for image file
        image_filename = f"{txid}.png"
        image_path = Path(lot1_images_dir) / image_filename
        
        if not image_path.exists():
            print(f"⚠️  Missing: {image_path}")
            continue
        
        try:
            # Load image
            img = cv2.imread(str(image_path))
            
            if img is None:
                print(f"⚠️  Failed to load: {image_path}")
                continue
            
            # For now, assume all Lot1 images are valid (you can adjust this)
            label = 'valid'
            
            training_images.append((img, label))
            valid_count += 1
            
        except Exception as e:
            print(f"❌ Error loading {image_path}: {e}")
    
    print(f"\n📊 Loaded {len(training_images)} images")
    print(f"  Valid: {valid_count}")
    print(f"  Invalid: {invalid_count}")
    
    if len(training_images) < 10:
        print("❌ Need at least 10 images to train. Found only:", len(training_images))
        return
    
    # Prepare training data
    print(f"\n🤖 Extracting features and preparing training data...")
    
    X = []
    y = []
    
    for i, (image, label) in enumerate(training_images):
        try:
            # Simple feature extraction for now
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Extract basic features
            hist = cv2.calcHist([img_gray], [0], None, [32], [0, 256])
            hist_norm = cv2.normalize(hist, hist).flatten()
            
            # Edge detection
            edges = cv2.Canny(img_gray, 50, 150)
            edge_count = int(np.sum(edges > 0)) / img_gray.size * 100
            
            # Contour analysis
            contours, _ = cv2.findContours(
                edges, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            contour_areas = [cv2.contourArea(c) for c in contours]
            num_contours = len(contours)
            mean_area = np.mean(contour_areas) if contour_areas else 0
            
            # Combine features
            features = np.concatenate([
                hist_norm,
                [edge_count],
                [num_contours],
                [mean_area]
            ])
            
            X.append(features)
            y.append(1 if label == 'valid' else 0)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(training_images)} images")
                
        except Exception as e:
            print(f"❌ Error processing image {i}: {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Dataset split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Train model
    print(f"\n🎯 Training Random Forest classifier...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"\n📈 Training Results:")
    print(f"  Train accuracy: {train_score:.4f}")
    print(f"  Test accuracy: {test_score:.4f}")
    
    # Save model
    model_dir = Path(__file__).parent / 'models'
    model_dir.mkdir(exist_ok=True)
    model_path = str(model_dir / 'lot1_xray_validator.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, str(model_dir / 'scaler_lot1.pkl'))
    
    print(f"\n✓ Model saved to: {model_path}")
    print(f"✓ Scaler saved to: {model_dir / 'scaler_lot1.pkl'}")
    
    # Save training metadata
    metadata = {
        'source': 'Lot1',
        'excel_file': excel_file,
        'total_images_loaded': len(training_images),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'training_accuracy': float(train_score),
        'test_accuracy': float(test_score)
    }
    
    metadata_path = model_dir / 'lot1_training_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to: {metadata_path}")


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    train_on_lot1()