"""DDI Dataset loader with metadata processing."""

import os
import pandas as pd
import numpy as np
from pathlib import Path


class DDI_DataLoader:
    """Load and process DDI dataset with fairness-focused utilities."""
    
    def __init__(self, root_dir="./DDI"):
        """
        Initialize DDI dataset loader.
        
        Args:
            root_dir (str): Path to DDI dataset root directory
        """
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.metadata_path = self.root_dir / "ddi_metadata.csv"
        
        # Validate paths
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {self.metadata_path}")
        
        # Load metadata
        self.metadata = pd.read_csv(self.metadata_path)
        self._prepare_metadata()
        
        print(f"✓ Loaded {len(self.metadata)} images")
        print(f"✓ Skin tone groups: {sorted(self.metadata['skin_tone'].unique())}")
    
    def _prepare_metadata(self):
        """Prepare and clean metadata."""
        # Debug: Show available columns
        print(f"Available columns: {list(self.metadata.columns)}")
        
        # Find the malignancy column (different datasets may have different names)
        malignancy_col = None
        possible_names = ['malignant', 'malignancy', 'malignancy(malig=1)', 'malig', 'is_malignant']
        
        for col in self.metadata.columns:
            if any(name in col.lower() for name in ['malig']):
                malignancy_col = col
                break
        
        if malignancy_col is None:
            raise ValueError(f"Cannot find malignancy column. Available columns: {list(self.metadata.columns)}")
        
        print(f"Using malignancy column: '{malignancy_col}'")
        
        # Ensure 'malignant' column exists as boolean
        if 'malignant' not in self.metadata.columns or malignancy_col != 'malignant':
            # Handle different possible formats
            def convert_to_bool(x):
                if pd.isna(x):
                    return False
                if isinstance(x, bool):
                    return x
                if isinstance(x, (int, float)):
                    return bool(x) and x != 0
                if isinstance(x, str):
                    return x.lower() in ['true', '1', 'yes', 'malignant']
                return False
            
            self.metadata['malignant'] = self.metadata[malignancy_col].apply(convert_to_bool)
        
        # Create readable labels
        self.metadata['diagnosis'] = self.metadata['malignant'].map({
            False: 'Benign', 
            True: 'Malignant'
        })
        
        # Handle skin_tone_label
        if 'skin_tone_label' not in self.metadata.columns:
            self.metadata['skin_tone_label'] = self.metadata['skin_tone'].map({
                12: 'Fitzpatrick I-II',
                34: 'Fitzpatrick III-IV',
                56: 'Fitzpatrick V-VI'
            })
        
        # Add image paths if DDI_file column exists
        if 'DDI_file' in self.metadata.columns:
            self.metadata['image_path'] = self.metadata['DDI_file'].apply(
                lambda x: str(self.images_dir / x)
            )
        elif 'filename' in self.metadata.columns:
            self.metadata['DDI_file'] = self.metadata['filename']
            self.metadata['image_path'] = self.metadata['filename'].apply(
                lambda x: str(self.images_dir / x)
            )
        else:
            # Try to infer from images directory
            print("Warning: No filename column found, attempting to list images...")
            image_files = sorted([f.name for f in self.images_dir.glob("*.png")])
            if len(image_files) == len(self.metadata):
                self.metadata['DDI_file'] = image_files
                self.metadata['image_path'] = [str(self.images_dir / f) for f in image_files]
                print(f"✓ Matched {len(image_files)} images to metadata rows")
            else:
                print(f"Warning: Found {len(image_files)} images but {len(self.metadata)} metadata rows")
        
        # Verify data integrity
        print(f"✓ Malignant cases: {self.metadata['malignant'].sum()}")
        print(f"✓ Benign cases: {(~self.metadata['malignant']).sum()}")
    
    def get_metadata(self):
        """Return full metadata DataFrame."""
        return self.metadata.copy()
    
    def get_subgroup_counts(self):
        """Get counts by skin tone and malignancy for fairness analysis."""
        counts = self.metadata.groupby(['skin_tone_label', 'diagnosis']).size().reset_index(name='count')
        
        # Add percentages
        total = len(self.metadata)
        counts['percentage'] = (counts['count'] / total * 100).round(2)
        
        return counts
    
    def get_outcome_by_group(self):
        """Calculate malignancy rate by skin tone (for fairness metrics)."""
        outcome_rates = self.metadata.groupby('skin_tone_label')['malignant'].agg([
            ('total', 'count'),
            ('malignant_count', 'sum'),
            ('malignancy_rate', 'mean')
        ]).reset_index()
        
        outcome_rates['malignancy_rate'] = (outcome_rates['malignancy_rate'] * 100).round(2)
        return outcome_rates
    
    def get_basic_stats(self):
        """Get basic dataset statistics."""
        # Get unique identifier (DDI_file or index)
        unique_id_col = 'DDI_file' if 'DDI_file' in self.metadata.columns else self.metadata.index
        
        stats = {
            'total_images': len(self.metadata),
            'unique_patients': self.metadata[unique_id_col].nunique() if isinstance(unique_id_col, str) else len(self.metadata),
            'malignant_count': int(self.metadata['malignant'].sum()),
            'benign_count': int((~self.metadata['malignant']).sum()),
            'malignancy_rate': round(self.metadata['malignant'].mean() * 100, 2),
            'skin_tone_distribution': self.metadata['skin_tone'].value_counts().to_dict()
        }
        return stats
    
    def check_missing_data(self):
        """Check for missing values in metadata."""
        missing = self.metadata.isnull().sum()
        missing_pct = (missing / len(self.metadata) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'missing_count': missing,
            'missing_percentage': missing_pct
        })
        
        return missing_df[missing_df['missing_count'] > 0]
    
    def save_to_raw(self, output_path="./data/raw/ddi_metadata.csv"):
        """Save metadata to raw data folder."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.metadata.to_csv(output_path, index=False)
        print(f"✓ Metadata saved to {output_path}")
        return output_path


if __name__ == "__main__":
    # Test the loader
    try:
        print("\n" + "="*60)
        print("TESTING DDI DATA LOADER")
        print("="*60)
        
        loader = DDI_DataLoader(root_dir="./DDI")
        
        print("\n" + "="*60)
        print("BASIC STATISTICS")
        print("="*60)
        stats = loader.get_basic_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print("\n" + "="*60)
        print("SUBGROUP COUNTS")
        print("="*60)
        print(loader.get_subgroup_counts())
        
        print("\n" + "="*60)
        print("OUTCOME BY GROUP")
        print("="*60)
        print(loader.get_outcome_by_group())
        
        print("\n" + "="*60)
        print("MISSING DATA CHECK")
        print("="*60)
        missing = loader.check_missing_data()
        if len(missing) > 0:
            print(missing)
        else:
            print("✓ No missing data found!")
        
        print("\n" + "="*60)
        print("SAMPLE DATA")
        print("="*60)
        print(loader.get_metadata()[['DDI_file', 'skin_tone', 'skin_tone_label', 'diagnosis', 'malignant']].head(10))
        
        print("\n✅ Data loader test complete!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nPlease check:")
        print("1. DDI folder exists in project root")
        print("2. DDI/images/ contains PNG files")
        print("3. DDI/ddi_metadata.csv exists")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
