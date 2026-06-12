import pandas as pd


def parse_excel_manifest(filepath):
    """Parse Excel manifest and extract image paths with labels"""
    
    df = pd.read_excel(filepath)
    
    # Expected columns: 'image_path' or similar, 'label' (BT, pBT, partial BT, or blank)
    print(f"Loaded {len(df)} records from {filepath}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


def convert_to_binary_labels(df):
    """Convert labels to binary: 1=bad tab (BT/pBT/partial BT), 0=OK"""
    
    def is_bad_tab(label_value):
        if pd.isna(label_value) or str(label_value).strip() == '':
            return 0.0
        
        label_str = str(label_value).lower()
        
        # Check for bad tab indicators
        if 'bt' in label_str or 'partial' in label_str:
            return 1.0
        
        return 0.0
    
    df['label_binary'] = df.apply(lambda row: is_bad_tab(row.get('label', '')), axis=1)
    
    print(f"Bad tabs found: {df['label_binary'].sum()}")
    print(f"OK images: {(df['label_binary'] == 0).sum()}")
    
    return df


def prepare_training_data(lot_num):
    """Prepare training data for a specific lot"""
    
    filepath = f"C:\\Users\\ug10271\\OneDrive - Dexcom\\Documents\\CodeProjects\\xray-analysis\\lot{lot_num}_md.xlsx"
    
    # Load and parse manifest
    df = parse_excel_manifest(filepath)
    df = convert_to_binary_labels(df)
    
    return df

