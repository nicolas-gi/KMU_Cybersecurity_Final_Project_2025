"""
Dataset Processor for CICIDS2017 and NSL-KDD
Downloads and prepares datasets for training
"""

import pandas as pd
import numpy as np
import os
import urllib.request
from zipfile import ZipFile

# Constants
KDD_TRAIN_FILE = 'KDDTrain+.txt'
KDD_TEST_FILE = 'KDDTest+.txt'
LABEL_COLUMN = ' Label'
IS_ATTACK_COLUMN = 'is_attack'

class DatasetProcessor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_nsl_kdd(self):
        """Download NSL-KDD dataset"""
        print("📥 Downloading NSL-KDD dataset...")
        
        base_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
        files = {
            KDD_TRAIN_FILE: KDD_TRAIN_FILE,
            KDD_TEST_FILE: KDD_TEST_FILE,
        }

        for filename, url_path in files.items():
            file_path = os.path.join(self.data_dir, filename)
            if not os.path.exists(file_path):
                try:
                    url = base_url + url_path
                    print(f"  Downloading {filename}...")
                    urllib.request.urlretrieve(url, file_path)
                    print(f"  ✓ {filename} downloaded")
                except Exception as e:
                    print(f"  ✗ Failed to download {filename}: {e}")
            else:
                print(f"  ✓ {filename} already exists")
        
        print("✓ NSL-KDD dataset ready")
    
    def process_cicids2017(self, csv_path):
        """
        Process CICIDS2017 dataset
        
        Download from: https://www.unb.ca/cic/datasets/ids-2017.html
        
        Args:
            csv_path: Path to CICIDS2017 CSV file
        """
        print(f"📊 Processing CICIDS2017 dataset from {csv_path}...")
        
        try:
            # Read CSV
            df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
            
            # Clean column names
            df.columns = df.columns.str.strip()

            # Remove rows with missing values in label
            if LABEL_COLUMN in df.columns:
                df = df.dropna(subset=[LABEL_COLUMN])

            # Convert label to binary (BENIGN vs ATTACK)
            if LABEL_COLUMN in df.columns:
                df[IS_ATTACK_COLUMN] = (df[LABEL_COLUMN] != 'BENIGN').astype(int)

            # Handle infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0)

            # Save processed dataset
            output_path = os.path.join(self.data_dir, 'cicids2017_processed.csv')
            df.to_csv(output_path, index=False)
            
            print(f"✓ Processed dataset saved to {output_path}")
            print(f"  Total samples: {len(df)}")
            print(f"  Attack samples: {df[IS_ATTACK_COLUMN].sum()}")
            print(f"  Normal samples: {len(df) - df[IS_ATTACK_COLUMN].sum()}")
            
            return output_path
        
        except FileNotFoundError:
            print(f"✗ File not found: {csv_path}")
            print("  Please download CICIDS2017 dataset from:")
            print("  https://www.unb.ca/cic/datasets/ids-2017.html")
            return None
        except Exception as e:
            print(f"✗ Error processing dataset: {e}")
            return None
    
    def get_dataset_info(self):
        """Get information about available datasets"""
        print("\n" + "="*60)
        print("Dataset Information")
        print("="*60)
        
        # Check NSL-KDD
        nsl_kdd_train = os.path.join(self.data_dir, KDD_TRAIN_FILE)
        nsl_kdd_test = os.path.join(self.data_dir, KDD_TEST_FILE)
        
        print("\n📁 NSL-KDD Dataset:")
        if os.path.exists(nsl_kdd_train) and os.path.exists(nsl_kdd_test):
            print("  ✓ Available")
            train_size = os.path.getsize(nsl_kdd_train) / (1024 * 1024)
            test_size = os.path.getsize(nsl_kdd_test) / (1024 * 1024)
            print(f"  Training set: {train_size:.2f} MB")
            print(f"  Test set: {test_size:.2f} MB")
        else:
            print("  ✗ Not available")
            print("  Run: python3 process_datasets.py --download-nsl-kdd")
        
        # Check CICIDS2017
        cicids_processed = os.path.join(self.data_dir, 'cicids2017_processed.csv')
        print("\n📁 CICIDS2017 Dataset:")
        if os.path.exists(cicids_processed):
            print("  ✓ Available (processed)")
            size = os.path.getsize(cicids_processed) / (1024 * 1024)
            print(f"  Size: {size:.2f} MB")
        else:
            print("  ✗ Not available")
            print("  Download from: https://www.unb.ca/cic/datasets/ids-2017.html")
            print("  Then run: python3 process_datasets.py --process-cicids <path-to-csv>")
        
        print("="*60 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process network intrusion datasets')
    parser.add_argument('--download-nsl-kdd', action='store_true',
                        help='Download NSL-KDD dataset')
    parser.add_argument('--process-cicids', type=str,
                        help='Process CICIDS2017 dataset from CSV file')
    parser.add_argument('--info', action='store_true',
                        help='Show dataset information')
    
    args = parser.parse_args()
    
    processor = DatasetProcessor()
    
    if args.download_nsl_kdd:
        processor.download_nsl_kdd()
    
    if args.process_cicids:
        processor.process_cicids2017(args.process_cicids)
    
    if args.info or (not args.download_nsl_kdd and not args.process_cicids):
        processor.get_dataset_info()


if __name__ == "__main__":
    main()
