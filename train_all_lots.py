# train_all_lots.py - Train model on Lot1, Lot2, and Lot3 images combined

import cv2
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib


def train_on_all_lots():
    """Train model using images from all lots (Lot1, Lot2, Lot3)"""
    
    # Configuration - adjust these paths as needed
    excel_files = {
        'lot1': 'Lot1_md.xlsx',
        'lot2': 'Lot2_md.xlsx',
        'lot3': 'Lot3_md.xlsx'
    }
    
    images_dirs = {
        'lot1': 'lot1_images',
        'lot2': 'lot2_images',
        'lot3': 'lot3_images'
    }
    
    print("="*60)
    print("TRAIN MODEL ON ALL LOTS (LOT1 + LOT2 + LOT3)")
    print("="*60)
    
    # Collect all training images from each lot
    all_training_images = []
    lot_stats = {}
    
    for lot_name, excel_file in excel_files.items():
        image_dir = images_dirs[lot_name]
        
        print(f"\n{'='*40}")
        print(f"Processing {lot_name.upper()}")
        print('='*40)
        
        # Check if Excel file exists
        if not Path(excel_file).exists():
            print(f"⚠️  Excel file not found: {excel_file}")
            continue
        
        try:
            import pandas as pd
            
            df = pd.read_excel(excel_file)
            
            # Find TXID column
            txid_col = None
            for col in df.columns:
                if 'TXID' in str(col).upper():
                    txid_col = col
                    break
            
            if txid_col is None:
                print(f"⚠️  Could not find TXID column in {excel_file}")
                continue
            
            # Check which images exist and load them
            training_images = []
            valid_count = 0
            invalid_count = 0
            
            for idx, row in df.iterrows():
                txid = str(row[txid_col])
                
                # Look for image file
                image_filename = f"{txid}.png"
                image_path = Path(image_dir) / image_filename
                
                if not image_path.exists():
                    print(f"⚠️  Missing: {image_path}")
                    invalid_count += 1
                    continue
                
                try:
                    # Load image
                    img = cv2.imread(str(image_path))
                    
                    if img is None:
                        print(f"⚠️  Failed to load: {image_path}")
                        invalid_count += 1
                        continue
                    
                    # For now, assume all images are valid (you can adjust this)
                    label = 'valid'
                    
                    training_images.append((img, label))
                    valid_count += 1
                    
                except Exception as e:
                    print(f"❌ Error loading {image_path}: {e}")
            
            lot_stats[lot_name] = {
                'total_rows': len(df),
                'valid_images': valid_count,
                'invalid_images': invalid_count
            }
            
            all_training_images.extend(training_images)
            
            print(f"  Loaded {len(training_images)} images from {lot_name}")
            print(f"    Valid: {valid_count}, Invalid/Missing: {invalid_count}")
            
        except Exception as e:
            print(f"❌ Error processing {excel_file}: {e}")
    
    # Check if we have enough images to train
    total_images = len(all_training_images)
    print(f"\n{'='*40}")
    print("SUMMARY")
    print('='*40)
    print(f"Total images loaded: {total_images}")
    
    if total_images < 10:
        print("❌ Need at least 10 images to train. Found only:", total_images)
        return
    
    # Prepare training data
    print(f"\n🤖 Extracting features and preparing training data...")
    
    X = []
    y = []
    
    for i, (image, label) in enumerate(all_training_images):
        try:
            # Simple feature extraction
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
                print(f"  Processed {i + 1}/{total_images} images")
                
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
    model_path = str(model_dir / 'all_lots_xray_validator.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, str(model_dir / 'scaler_all_lots.pkl'))
    
    print(f"\n✓ Model saved to: {model_path}")
    print(f"✓ Scaler saved to: {model_dir / 'scaler_all_lots.pkl'}")
    
    # Save training metadata with lot statistics
    metadata = {
        'source': 'All Lots',
        'excel_files': excel_files,
        'image_directories': images_dirs,
        'total_images_loaded': total_images,
        'lot_statistics': lot_stats,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'training_accuracy': float(train_score),
        'test_accuracy': float(test_score)
    }
    
    metadata_path = model_dir / 'all_lots_training_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to: {metadata_path}")


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    train_on_all_lots()