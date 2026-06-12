# check_lot1_match.py - Verify TXID matches with images in lot1_images folder

import pandas as pd
from pathlib import Path


def check_lot1_excel_match():
    """Check if TXID from Excel matches image files in lot1_images folder"""
    
    excel_file = 'Lot1_md.xlsx'
    images_folder = 'lot1_images'  # Folder containing the images
    
    # Read Excel file
    print(f"📖 Reading Excel file: {excel_file}")
    
    if not Path(excel_file).exists():
        print(f"❌ Excel file not found: {excel_file}")
        return
    
    df = pd.read_excel(excel_file)
    
    # Display available columns
    print(f"\n📋 Available columns in {excel_file}:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Find TXID column (case-insensitive search)
    txid_col = None
    
    for col in df.columns:
        if 'TXID' in str(col).upper():
            txid_col = col
            break
    
    if txid_col is None:
        print("\n❌ Could not find TXID column")
        return
    
    print(f"\n🔍 Using column: {txid_col}")
    
    # Check for matching image files in lot1_images folder
    images_folder_path = Path(images_folder)
    
    if not images_folder_path.exists():
        print(f"❌ Images folder not found: {images_folder_path}")
        return
    
    valid_count = 0
    missing_count = 0
    
    samples = []
    
    for idx, row in df.iterrows():
        txid = str(row[txid_col])
        
        # Look for image file with this TXID name
        image_filename = f"{txid}.png"
        image_path = images_folder_path / image_filename
        
        # Check if image exists
        if not image_path.exists():
            missing_count += 1
            samples.append({
                'row_index': idx,
                'lot': row['LOT'] if 'LOT' in df.columns else '',
                'group': row['GROUP'] if 'GROUP' in df.columns else '',
                'tray': row['TRAY'] if 'TRAY' in df.columns else '',
                'txid': txid,
                'xray_issues': str(row['Xray Gross Issues']) if pd.notna(row['Xray Gross Issues']) else '',
                'expected_image': f"{txid}.png",
                'status': 'MISSING',
                'message': f"Image file not found: {image_path}"
            })
        else:
            valid_count += 1
            samples.append({
                'row_index': idx,
                'lot': row['LOT'] if 'LOT' in df.columns else '',
                'group': row['GROUP'] if 'GROUP' in df.columns else '',
                'tray': row['TRAY'] if 'TRAY' in df.columns else '',
                'txid': txid,
                'xray_issues': str(row['Xray Gross Issues']) if pd.notna(row['Xray Gross Issues']) else '',
                'expected_image': f"{txid}.png",
                'status': 'VALID',
                'message': f"Image found at {image_path}"
            })
    
    # Print results
    print(f"\n📊 Matching Results:")
    print(f"  Valid images (found): {valid_count}")
    print(f"  Missing images: {missing_count}")
    print(f"  Total rows processed: {len(df)}")
    
    if samples:
        # Show some examples of missing files
        missing_samples = [s for s in samples if s['status'] == 'MISSING'][:20]
        if missing_samples:
            print("\n📝 First 20 missing images:")
            for sample in missing_samples:
                print(f"   - {sample['txid']} (expected: {sample['expected_image']})")
    
    # Save results to file
    results_df = pd.DataFrame(samples)
    output_file = Path(__file__).parent / 'lot1_match_results.csv'
    results_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == '__main__':
    check_lot1_excel_match()