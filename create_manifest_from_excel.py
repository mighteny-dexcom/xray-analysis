"""
Read Excel file and create labeled manifest for training
1. Read TXID (column E) and Xray gross issues (column F)
2. Mark images with "BT" or "Partial BT" as "valid" (has battery tab)
3. Mark all others as "invalid"
"""

import pandas as pd
import json
from pathlib import Path


def create_manifest_from_excel(excel_file: str = 'Lot2_md.xlsx'):
    """
    Read Excel file and create training manifest
    
    Column E (TXID): Image ID
    Column F (Xray gross issues): Issues description - look for "BT" or "Partial BT"
    """
    excel_path = Path(__file__).parent / excel_file
    
    if not excel_path.exists():
        print(f"❌ Excel file not found: {excel_path}")
        return
    
    print(f"📖 Reading Excel file: {excel_file}")
    
    # Read Excel
    df = pd.read_excel(excel_path)
    
    # Column mapping (E=4, F=5 in 0-indexed)
    # Check what columns are available
    print(f"\nAvailable columns: {df.columns.tolist()}")
    
    # Find the right columns
    txid_col = None
    issues_col = None
    
    for col in df.columns:
        if 'TXID' in str(col).upper():
            txid_col = col
        if 'XRAY' in str(col).upper() and 'ISSUES' in str(col).upper():
            issues_col = col
    
    if txid_col is None or issues_col is None:
        print(f"❌ Could not find TXID or Xray issues columns")
        print(f"   TXID column: {txid_col}")
        print(f"   Issues column: {issues_col}")
        return
    
    print(f"✓ Using columns: {txid_col} and {issues_col}")
    
    # Process data
    training_dir = Path(__file__).parent / 'training_images'
    valid_count = 0
    invalid_count = 0
    not_found = 0
    
    samples = []
    
    for idx, row in df.iterrows():
        txid = str(row[txid_col])
        issues = str(row[issues_col]) if pd.notna(row[issues_col]) else ""
        
        # Look for image file
        image_filename = f"{txid}.png"
        image_path = training_dir / image_filename
        
        if not image_path.exists():
            not_found += 1
            continue
        
        # Check if BT (Battery Tab) is mentioned
        has_bt = "BT" in issues.upper() or "PARTIAL BT" in issues.upper()
        label = "valid" if has_bt else "invalid"
        
        if has_bt:
            valid_count += 1
        else:
            invalid_count += 1
        
        sample = {
            'image_path': str(image_path),
            'label': label,
            'region_of_interest': None,
            'metadata': {
                'source': 'real_xray',
                'filename': image_filename,
                'txid': txid,
                'xray_issues': issues,
                'has_bt': has_bt
            }
        }
        
        samples.append(sample)
    
    print(f"\n📊 Processing Results:")
    print(f"  Valid (has BT/Partial BT): {valid_count}")
    print(f"  Invalid (no BT): {invalid_count}")
    print(f"  Not found in training_images: {not_found}")
    print(f"  Total: {len(samples)}")
    
    # Save manifest
    manifest = {
        'metadata': {
            'description': 'X-ray images labeled for Battery Tab (BT) detection',
            'task': 'Battery Tab (BT) vs No BT',
            'total_samples': len(samples),
            'valid_count': valid_count,
            'invalid_count': invalid_count,
            'source_excel': excel_file,
            'created': '2026-06-03'
        },
        'samples': samples
    }
    
    manifest_path = Path(__file__).parent / 'training_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Created: {manifest_path}")
    print(f"\nNext step: python train_model.py")


if __name__ == '__main__':
    create_manifest_from_excel()
