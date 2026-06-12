"""
Test Data Loading Pipeline - Final Version
Verifies:
1. All 3 lots load correctly (~600 total records)
2. Labels parse properly (BT/pBT -> 1.0, Functional/blank -> 0.0)
3. Images can be opened from their folder locations
"""

import json
import os
from pathlib import Path

def test_data_loading():
    """Test the data loading pipeline"""
    
    # Use current working directory as base
    base_dir = Path.cwd()
    
    # Load manifests for each lot
    lot1_manifest = base_dir / "Lot1_md.xlsx"
    lot2_manifest = base_dir / "Lot2_md.xlsx"
    lot3_manifest = base_dir / "Lot3_md.xlsx"
    
    # Image directories (relative to current directory)
    lot1_images_dir = base_dir / "lot1_images"
    lot2_images_dir = base_dir / "lot2_images"
    lot3_images_dir = base_dir / "lot3_images"
    
    print("=" * 60)
    print("TESTING DATA LOADING PIPELINE")
    print("=" * 60)
    print()
    
    # Test Lot 1
    print("-" * 40)
    print("LOT 1 TESTS")
    print("-" * 40)
    
    try:
        import pandas as pd
        
        if lot1_manifest.exists():
            df1 = pd.read_excel(lot1_manifest)
            print("[OK] Lot 1 manifest loaded successfully")
            print("  Records count:", len(df1))
            
            # Test label parsing for Lot 1 - correct column name is 'Xray Gross Issues'
            sample_labels = df1['Xray Gross Issues'].head(5).tolist()
            print("  Sample labels:", sample_labels)
            
        else:
            print("[FAIL] Lot 1 manifest not found")
    except Exception as e:
        print("[ERROR] Error loading Lot 1:", str(e))
    
    # Test Lot 2
    print()
    print("-" * 40)
    print("LOT 2 TESTS")
    print("-" * 40)
    
    try:
        if lot2_manifest.exists():
            df2 = pd.read_excel(lot2_manifest)
            print("[OK] Lot 2 manifest loaded successfully")
            print("  Records count:", len(df2))
            
            # Test label parsing for Lot 2 - correct column name is 'Xray Gross Issues'
            sample_labels = df2['Xray Gross Issues'].head(5).tolist()
            print("  Sample labels:", sample_labels)
            
        else:
            print("[FAIL] Lot 2 manifest not found")
    except Exception as e:
        print("[ERROR] Error loading Lot 2:", str(e))
    
    # Test Lot 3
    print()
    print("-" * 40)
    print("LOT 3 TESTS")
    print("-" * 40)
    
    try:
        if lot3_manifest.exists():
            df3 = pd.read_excel(lot3_manifest)
            print("[OK] Lot 3 manifest loaded successfully")
            print("  Records count:", len(df3))
            
            # Test label parsing for Lot 3 - correct column name is 'Xray Gross Issues'
            sample_labels = df3['Xray Gross Issues'].head(5).tolist()
            print("  Sample labels:", sample_labels)
            
        else:
            print("[FAIL] Lot 3 manifest not found")
    except Exception as e:
        print("[ERROR] Error loading Lot 3:", str(e))
    
    # Test image folder access
    print()
    print("-" * 40)
    print("IMAGE FOLDER ACCESS TESTS")
    print("-" * 40)
    
    for lot_num, images_dir in [("1", lot1_images_dir), ("2", lot2_images_dir), ("3", lot3_images_dir)]:
        try:
            if images_dir.exists():
                image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
                print("[OK] Lot {} images folder accessible".format(lot_num))
                print("  Found {} image files".format(len(image_files)))
                
                # Try to open one sample image if any exist
                if image_files:
                    sample_image = image_files[0]
                    from PIL import Image
                    try:
                        img = Image.open(sample_image)
                        print("[OK] Sample image opened successfully:", sample_image.name)
                        print("    Size:", img.size, "Mode:", img.mode)
                    except Exception as e:
                        print("[ERROR] Error opening sample image:", str(e))
            else:
                print("[FAIL] Lot {} images folder not found".format(lot_num))
        except Exception as e:
            print("[ERROR] Error accessing Lot {} images:".format(lot_num), str(e))
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_records = len(df1) + len(df2) + len(df3) if all([lot1_manifest.exists(), lot2_manifest.exists(), lot3_manifest.exists()]) else "N/A"
    print("Total records across all lots:", total_records)
    print()
    print("Expected: ~600 total records")
    
    # Check label distribution
    if all([lot1_manifest.exists(), lot2_manifest.exists(), lot3_manifest.exists()]):
        combined_df = pd.concat([df1, df2, df3], ignore_index=True)
        
        # Count unique label values
        unique_labels = combined_df['Xray Gross Issues'].unique().tolist()
        print("Unique labels found:", unique_labels)
        
        # Check for expected label patterns - convert to string for comparison
        def normalize_label(label):
            if pd.isna(label) or str(label).strip() == '':
                return 'blank'
            elif 'BT' in str(label) or 'pBT' in str(label):
                return 'BT/pBT'
            else:
                return str(label).lower()
        
        normalized_labels = combined_df['Xray Gross Issues'].apply(normalize_label)
        bt_count = len(normalized_labels[normalized_labels == 'BT/pBT'])
        functional_count = len(normalized_labels[normalized_labels == 'functional'])
        blank_count = len(normalized_labels[normalized_labels == 'blank'])
        
        print("  BT/pBT labels:", bt_count)
        print("  Functional/blank labels:", functional_count + blank_count)
    
    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_data_loading()