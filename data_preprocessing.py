"""
Data Preprocessing for Deep Learning Ad Recommender
Handles Criteo Display Advertising Dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from typing import Dict, List, Tuple
import gc

class CriteoDataPreprocessor:
    """Preprocessor for Criteo Display Advertising Dataset"""
    
    def __init__(self, 
                 numerical_cols: List[str] = None,
                 categorical_cols: List[str] = None):
        """
        Initialize preprocessor
        
        Args:
            numerical_cols: List of numerical feature column names
            categorical_cols: List of categorical feature column names
        """
        self.numerical_cols = numerical_cols or [f'I{i}' for i in range(1, 14)]
        self.categorical_cols = categorical_cols or [f'C{i}' for i in range(1, 27)]
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_dims = {}
        
    def load_criteo_data(self, 
                         filepath: str, 
                         nrows: int = None,
                         sample_negative_ratio: float = 1.0) -> pd.DataFrame:
        """
        Load Criteo dataset
        
        Args:
            filepath: Path to Criteo data file
            nrows: Number of rows to load (None for all)
            sample_negative_ratio: Ratio to downsample negative examples
        """
        print(f"Loading Criteo data from {filepath}...")
        
        # Column names for Criteo dataset
        columns = ['label'] + self.numerical_cols + self.categorical_cols
        
        # Load data
        df = pd.read_csv(
            filepath, 
            sep='\t', 
            names=columns,
            nrows=nrows,
            na_values=['']
        )
        
        print(f"Loaded {len(df)} rows")
        print(f"Click-through rate: {df['label'].mean():.4f}")
        
        # Downsample negative examples if needed
        if sample_negative_ratio < 1.0:
            df = self._balance_dataset(df, sample_negative_ratio)
        
        return df
    
    def _balance_dataset(self, 
                        df: pd.DataFrame, 
                        negative_ratio: float) -> pd.DataFrame:
        """Balance dataset by downsampling negative examples"""
        positive = df[df['label'] == 1]
        negative = df[df['label'] == 0]
        
        n_negative = int(len(positive) * negative_ratio)
        negative_sampled = negative.sample(n=min(n_negative, len(negative)), 
                                          random_state=42)
        
        balanced_df = pd.concat([positive, negative_sampled])
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Balanced dataset: {len(balanced_df)} rows, "
              f"CTR: {balanced_df['label'].mean():.4f}")
        
        return balanced_df
    
    def preprocess_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess numerical features"""
        print("Preprocessing numerical features...")
        
        # Fill missing values with median
        for col in self.numerical_cols:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # Log transform (many numerical features are skewed)
        for col in self.numerical_cols:
            if col in df.columns:
                df[col] = np.log1p(np.abs(df[col]))
        
        return df
    
    def preprocess_categorical(self, 
                               df: pd.DataFrame, 
                               fit: bool = True) -> pd.DataFrame:
        """
        Preprocess categorical features
        
        Args:
            df: Input dataframe
            fit: Whether to fit encoders (True for train, False for test)
        """
        print("Preprocessing categorical features...")
        
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
                
            # Fill missing values
            df[col] = df[col].fillna('missing')
            
            # Handle rare categories (frequency < 10)
            if fit:
                value_counts = df[col].value_counts()
                rare_values = value_counts[value_counts < 10].index
                df[col] = df[col].replace(rare_values, 'rare')
            
            # Label encoding
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
                self.feature_dims[col] = len(self.label_encoders[col].classes_)
            else:
                # Handle unseen categories
                df[col] = df[col].apply(
                    lambda x: x if x in self.label_encoders[col].classes_ else 'rare'
                )
                df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def create_user_ad_features(self, 
                                df: pd.DataFrame,
                                user_id_col: str = 'user_id',
                                ad_id_col: str = 'ad_id') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split features into user features and ad features
        For two-tower model
        """
        # For Criteo, we'll create synthetic user/ad split
        # In production, you'd have actual user IDs and ad IDs
        
        # User features: first 6 categorical + numerical features
        user_feature_cols = self.numerical_cols + self.categorical_cols[:6]
        user_features = df[user_feature_cols].copy()
        
        # Ad features: remaining categorical features
        ad_feature_cols = self.categorical_cols[6:]
        ad_features = df[ad_feature_cols].copy()
        
        return user_features, ad_features
    
    def fit_transform(self, df: pd.DataFrame) -> Dict:
        """
        Fit preprocessor and transform data
        
        Returns:
            Dictionary with processed features and labels
        """
        # Preprocess
        df = self.preprocess_numerical(df)
        df = self.preprocess_categorical(df, fit=True)
        
        # Scale numerical features
        numerical_data = df[self.numerical_cols].values
        numerical_data = self.scaler.fit_transform(numerical_data)
        
        # Extract labels
        labels = df['label'].values
        
        # Get categorical features
        categorical_data = df[self.categorical_cols].values
        
        # Split into user and ad features
        user_features, ad_features = self.create_user_ad_features(df)
        
        return {
            'numerical': numerical_data,
            'categorical': categorical_data,
            'labels': labels,
            'user_features': user_features.values,
            'ad_features': ad_features.values,
            'full_df': df
        }
    
    def transform(self, df: pd.DataFrame) -> Dict:
        """Transform data using fitted preprocessor"""
        df = self.preprocess_numerical(df)
        df = self.preprocess_categorical(df, fit=False)
        
        numerical_data = self.scaler.transform(df[self.numerical_cols].values)
        labels = df['label'].values
        categorical_data = df[self.categorical_cols].values
        
        user_features, ad_features = self.create_user_ad_features(df)
        
        return {
            'numerical': numerical_data,
            'categorical': categorical_data,
            'labels': labels,
            'user_features': user_features.values,
            'ad_features': ad_features.values,
            'full_df': df
        }
    
    def save(self, filepath: str):
        """Save preprocessor state"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'feature_dims': self.feature_dims,
                'numerical_cols': self.numerical_cols,
                'categorical_cols': self.categorical_cols
            }, f)
        print(f"Preprocessor saved to {filepath}")
    
    def load(self, filepath: str):
        """Load preprocessor state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self.label_encoders = state['label_encoders']
            self.scaler = state['scaler']
            self.feature_dims = state['feature_dims']
            self.numerical_cols = state['numerical_cols']
            self.categorical_cols = state['categorical_cols']
        print(f"Preprocessor loaded from {filepath}")


def create_synthetic_criteo_data(n_samples: int = 100000, 
                                 save_path: str = None) -> pd.DataFrame:
    """
    Create synthetic Criteo-like data for testing
    Use this if you don't have the actual Criteo dataset
    """
    print(f"Creating synthetic Criteo-like dataset with {n_samples} samples...")
    
    np.random.seed(42)
    
    # Numerical features (13 columns)
    numerical_cols = [f'I{i}' for i in range(1, 14)]
    numerical_data = np.random.lognormal(0, 1, size=(n_samples, 13))
    
    # Categorical features (26 columns)
    categorical_cols = [f'C{i}' for i in range(1, 27)]
    categorical_data = {}
    
    # Create categorical features with different cardinalities
    cardinalities = [1000, 500, 100, 50] * 6 + [20, 10]
    for i, col in enumerate(categorical_cols):
        categorical_data[col] = np.random.choice(
            [f'cat_{j}' for j in range(cardinalities[i])], 
            size=n_samples
        )
    
    # Create labels (CTR around 0.25 for realistic simulation)
    # Labels correlate with some features
    feature_sum = numerical_data[:, 0] + numerical_data[:, 1]
    probs = 1 / (1 + np.exp(-0.1 * (feature_sum - 5)))
    labels = (np.random.random(n_samples) < probs).astype(int)
    
    # Combine into dataframe
    df = pd.DataFrame(numerical_data, columns=numerical_cols)
    for col in categorical_cols:
        df[col] = categorical_data[col]
    df['label'] = labels
    
    # Reorder columns
    df = df[['label'] + numerical_cols + categorical_cols]
    
    print(f"Created dataset with CTR: {df['label'].mean():.4f}")
    
    if save_path:
        df.to_csv(save_path, sep='\t', index=False, header=False)
        print(f"Saved to {save_path}")
    
    return df


if __name__ == "__main__":
    # Example usage
    print("=== Data Preprocessing Demo ===\n")
    
    # Create synthetic data
    df = create_synthetic_criteo_data(n_samples=50000, 
                                     save_path='/home/claude/ad_recommender/data/synthetic_criteo.txt')
    
    # Initialize preprocessor
    preprocessor = CriteoDataPreprocessor()
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Fit and transform
    train_data = preprocessor.fit_transform(train_df)
    test_data = preprocessor.transform(test_df)
    
    print(f"\nTrain data shape: {train_data['numerical'].shape}")
    print(f"Test data shape: {test_data['numerical'].shape}")
    print(f"Feature dimensions: {preprocessor.feature_dims}")
    
    # Save preprocessor
    preprocessor.save('/home/claude/ad_recommender/models/preprocessor.pkl')
    
    print("\n✓ Data preprocessing complete!")
