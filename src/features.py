"""Feature engineering for DDI dataset."""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from PIL import Image
import torch
from torchvision import transforms


class DDI_FeatureEngineer:
    """Extract and engineer features from DDI dataset."""
    
    def __init__(self, metadata_df, images_dir="./DDI/images"):
        """
        Initialize feature engineer.
        
        Args:
            metadata_df (pd.DataFrame): Metadata from DDI_DataLoader
            images_dir (str): Path to images directory
        """
        self.metadata = metadata_df.copy()
        self.images_dir = Path(images_dir)
        self.feature_list = []
        
    def extract_metadata_features(self):
        """Extract features from metadata (skin tone, etc.)."""
        features = pd.DataFrame()
        
        # One-hot encode skin tone
        skin_tone_dummies = pd.get_dummies(self.metadata['skin_tone'], prefix='skin_tone')
        features = pd.concat([features, skin_tone_dummies], axis=1)
        
        # Target variable
        features['malignant'] = self.metadata['malignant'].astype(int)
        
        # Add image filename for reference
        features['DDI_file'] = self.metadata['DDI_file']
        
        self.feature_list.extend(skin_tone_dummies.columns.tolist())
        
        return features
    
    def extract_image_statistics(self, sample_size=None):
        """
        Extract basic image statistics (size, mean color, etc.).
        
        Args:
            sample_size (int): Number of images to process (None = all)
        """
        if sample_size:
            sample_df = self.metadata.sample(n=min(sample_size, len(self.metadata)), random_state=42)
        else:
            sample_df = self.metadata
        
        image_features = []
        
        print(f"Extracting image statistics from {len(sample_df)} images...")
        
        for idx, row in sample_df.iterrows():
            img_path = self.images_dir / row['DDI_file']
            
            if not img_path.exists():
                print(f"Warning: Image not found - {img_path}")
                continue
            
            try:
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                
                features = {
                    'DDI_file': row['DDI_file'],
                    'img_width': img.size[0],
                    'img_height': img.size[1],
                    'img_aspect_ratio': img.size[0] / img.size[1],
                    'mean_red': img_array[:,:,0].mean(),
                    'mean_green': img_array[:,:,1].mean(),
                    'mean_blue': img_array[:,:,2].mean(),
                    'std_red': img_array[:,:,0].std(),
                    'std_green': img_array[:,:,1].std(),
                    'std_blue': img_array[:,:,2].std(),
                }
                
                image_features.append(features)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        img_features_df = pd.DataFrame(image_features)
        
        # Track feature names
        self.feature_list.extend([col for col in img_features_df.columns if col != 'DDI_file'])
        
        return img_features_df
    
    def create_feature_matrix(self, include_image_stats=True, image_sample_size=None):
        """
        Create complete feature matrix combining all features.
        
        Args:
            include_image_stats (bool): Whether to extract image statistics
            image_sample_size (int): Sample size for image stats (None = all)
        
        Returns:
            pd.DataFrame: Complete feature matrix
        """
        print("Creating feature matrix...")
        
        # Extract metadata features
        features = self.extract_metadata_features()
        
        # Extract image statistics if requested
        if include_image_stats:
            img_features = self.extract_image_statistics(sample_size=image_sample_size)
            features = features.merge(img_features, on='DDI_file', how='left')
            
            # Fill missing image stats with median (for any failed images)
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if features[col].isnull().any():
                    features[col].fillna(features[col].median(), inplace=True)
        
        print(f"✓ Feature matrix created: {features.shape}")
        print(f"✓ Features: {len(self.feature_list)}")
        
        return features
    
    def save_features(self, features_df, output_dir="./data/processed"):
        """Save feature matrix and feature list."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features as parquet
        features_path = output_dir / "features.parquet"
        features_df.to_parquet(features_path, index=False)
        print(f"✓ Features saved to {features_path}")
        
        # Save feature list as JSON
        feature_info = {
            'total_features': len(self.feature_list),
            'feature_names': self.feature_list,
            'target_variable': 'malignant',
            'protected_attributes': ['skin_tone_12', 'skin_tone_34', 'skin_tone_56']
        }
        
        feature_list_path = output_dir / "feature_list.json"
        with open(feature_list_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        print(f"✓ Feature list saved to {feature_list_path}")
        
        return features_path, feature_list_path


if __name__ == "__main__":
    from data_loader import DDI_DataLoader
    
    try:
        # Load data
        print("Loading DDI dataset...")
        loader = DDI_DataLoader(root_dir="./DDI")
        metadata = loader.get_metadata()
        
        # Engineer features
        print("\nEngineering features...")
        engineer = DDI_FeatureEngineer(metadata, images_dir="./DDI/images")
        features = engineer.create_feature_matrix(include_image_stats=True, image_sample_size=100)
        
        # Save
        print("\nSaving features...")
        engineer.save_features(features)
        
        print("\n✅ Feature engineering complete!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nPlease verify:")
        print("1. DDI/ddi_metadata.csv exists")
        print("2. DDI/images/ folder contains PNG files")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
