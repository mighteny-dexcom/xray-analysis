import os
import pandas as pd
from PIL import Image


def load_lot_manifest(lot_num):
    """Load Excel manifest for a specific lot"""
    filepath = f"C:\\Users\\ug10271\\OneDrive - Dexcom\\Documents\\CodeProjects\\xray-analysis\\Lot{lot_num}_md.xlsx"
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Manifest file {filepath} not found")
    
    df = pd.read_excel(filepath)
    return df


def parse_labels(df):
    """Parse labels from Excel (BT, pBT -> binary: 1=bad, 0=OK)"""
    # Map label column to binary classification
    def is_bad_tab(label):
        if pd.isna(label):
            return 0.0
        label_str = str(label).lower()
        if 'bt' in label_str or 'partial' in label_str:
            return 1.0
        return 0.0
    
    # Your column is "Xray Gross Issues" based on your data
    df['label_binary'] = df['Xray Gross Issues'].apply(is_bad_tab)
    
    print(f"Bad tabs found: {df['label_binary'].sum()}")
    print(f"OK images: {(df['label_binary'] == 0).sum()}")
    
    return df


def load_images_from_folder(folder_path):
    """Load all images from a folder and convert to tensors"""
    image_paths = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                image_paths.append({
                    'path': img_path,
                    'image': img
                })
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return image_paths


def create_dataset(lot_num):
    """Create PyTorch dataset for a specific lot"""
    # Load manifest and parse labels
    df = load_lot_manifest(lot_num)
    df = parse_labels(df)
    
    # Get list of images from folder
    folder_path = f"C:\\Users\\ug10271\\OneDrive - Dexcom\\Documents\\CodeProjects\\xray-analysis\\data\\images\\lot{lot_num}"
    image_list = load_images_from_folder(folder_path)
    
    return df, image_list


def create_dataloader(lot_num, batch_size=8):
    """Create PyTorch DataLoader for a lot"""
    from torch.utils.data import Dataset, DataLoader
    
    # Create custom dataset class
    class XRayDataset(Dataset):
        def __init__(self, manifest_df, image_list):
            self.manifest = manifest_df
            self.images = image_list
        
        def __len__(self):
            return len(self.manifest)
        
        def __getitem__(self, idx):
            # Get label from manifest
            row = self.manifest.iloc[idx]
            label = 1.0 if row['label_binary'] == 1 else 0.0
            
            # Find corresponding image (simplified - in practice match by filename)
            img_path = row.get('TXID', '')  # Using TXID as image path placeholder
            
            return {
                'image': Image.open(img_path).convert('RGB'),
                'label': label,
                'filename': img_path.split('\\')[-1] if '\\' in img_path else img_path.split('/')[-1]
            }
    
    # Load data for this lot - TEST DATA LOADING FIRST (as per suggestion)
    print("Loading test data...")
    df = load_lot_manifest(1)  # Test with lot=1

    # Check if dataset has images in folder
    image_list = []
    folder_path_test = f"C:\\Users\\ug10271\\OneDrive - Dexcom\\Documents\\CodeProjects\\xray-analysis\\data\\images\\lot{1}"

    for filename in os.listdir(folder_path_test):
        if (filename.lower().endswith(('.png', '.jpg', '.jpeg')) or
            any(path.split('\\')[-2] == 'img' and path.endswith('tag')  # check img tag as fallback)
             ):
                pass

    image_list = []


    folder_path_test = f"C:\\Users\\ug10271\\OneDrive - Dexcom\\Documents\\CodeProjects\\xray-analysis\\data\\images\\lot{1}"
    for filename in os.listdir(folder_path):
        if (filename.lower().endswith(('.png', '.jpg')) or
            any(path.split('\\')[-3] == 'img' and path.endswith('tag'): check image tag) : pass


    folder_path_test = f"C:\\Users\\ug10271\\OneDrive - Dexcom\\Documents\\CodeProjects\\xray-analysis\\data\\images\\lot{1}"
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg')):
            img = Image.open(os.path.join(folder_path, filename)).convert('RGB')  # Convert to RGB image
            image_list.append({
                'image': img,
                'path': f"C:\\Users\\ug10271\\OneDrive - Dexcom\\Documents\\CodeProjects\\xray-analysis\\data\\images\\lot{lot_num}" / filename.strip('pngjpg')  # Extract name
            })


    return image_list


def create_dataloader(lot_num, batch_size=8):
    """Create PyTorch DataLoader for a lot"""
    from torch.utils.data import Dataset, DataLoader
    dataset = XRayDataset(manifest_df=image_list)

