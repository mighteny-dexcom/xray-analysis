"""
Check Manifest Column Names
"""

import pandas as pd
from pathlib import Path

base_dir = Path.cwd()

for lot_num, manifest_name in [("1", "Lot1_md.xlsx"), ("2", "Lot2_md.xlsx"), ("3", "Lot3_md.xlsx")]:
    manifest_path = base_dir / manifest_name
    
    if manifest_path.exists():
        df = pd.read_excel(manifest_path)
        print("Lot {} Manifest Columns:".format(lot_num))
        print(df.columns.tolist())
        print()
        
        # Show sample data
        print("Sample rows:")
        print(df.head(3).to_string())
        print("-" * 40)
    else:
        print("Lot {} manifest not found".format(lot_num))