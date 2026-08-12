"""
Main Training Script for Deep Learning Ad Recommender
Complete end-to-end training of both stages
"""

import torch
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path
import sys

# Add project to path
sys.path.append('/home/claude/ad_recommender')

from data_preprocessing import CriteoDataPreprocessor, create_synthetic_criteo_data
from two_tower_model import TwoTowerModel
from transformer_ranker import TransformerRanker
from training_pipeline import AdDataset, TwoTowerTrainer, TransformerTrainer, build_faiss_index
from torch.utils.data import DataLoader


def create_feature_dimensions(preprocessor: CriteoDataPreprocessor) -> tuple:
    """
    Create feature dimension dictionaries for user and ad features
    
    Returns:
        user_feature_dims: Dictionary of user feature dimensions
        ad_feature_dims: Dictionary of ad feature dimensions
    """
    # Split categorical features
    # First 6 categorical features are user features
    user_cat_cols = [f'C{i}' for i in range(1, 7)]
    ad_cat_cols = [f'C{i}' for i in range(7, 27)]
    
    user_feature_dims = {
        col: preprocessor.feature_dims[col]
        for col in user_cat_cols
        if col in preprocessor.feature_dims
    }
    
    ad_feature_dims = {
        col: preprocessor.feature_dims[col]
        for col in ad_cat_cols
        if col in preprocessor.feature_dims
    }
    
    return user_feature_dims, ad_feature_dims


def prepare_data(args):
    """Load and preprocess data"""
    print("\n" + "="*60)
    print("STEP 1: DATA PREPARATION")
    print("="*60 + "\n")
    
    # Create or load data
    if args.use_synthetic:
        print(f"Creating synthetic dataset with {args.n_samples} samples...")
        df = create_synthetic_criteo_data(
            n_samples=args.n_samples,
            save_path=args.data_path
        )
    else:
        print(f"Loading data from {args.data_path}...")
        preprocessor = CriteoDataPreprocessor()
        df = preprocessor.load_criteo_data(
            args.data_path,
            nrows=args.n_samples,
            sample_negative_ratio=args.negative_ratio
        )
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")
    
    # Preprocess
    print("\nPreprocessing data...")
    preprocessor = CriteoDataPreprocessor()
    
    train_data = preprocessor.fit_transform(train_df)
    val_data = preprocessor.transform(val_df)
    test_data = preprocessor.transform(test_df)
    
    # Save preprocessor
    preprocessor.save(f"{args.model_dir}/preprocessor.pkl")
    
    return train_data, val_data, test_data, preprocessor


def split_features(data: dict) -> dict:
    """Split features into user and ad features"""
    # For Criteo, split categorical features
    # First 6 for user, rest for ad
    num_user_cat = 6
    
    return {
        'user_categorical': data['categorical'][:, :num_user_cat],
        'ad_categorical': data['categorical'][:, num_user_cat:],
        'numerical': data['numerical'],
        'labels': data['labels']
    }


def train_stage1(args, train_data, val_data, preprocessor):
    """Train Two-Tower Model (Stage 1: Candidate Generation)"""
    print("\n" + "="*60)
    print("STEP 2: TRAINING STAGE 1 - TWO-TOWER MODEL")
    print("="*60 + "\n")
    
    # Split features
    train_split = split_features(train_data)
    val_split = split_features(val_data)
    
    # Create datasets
    train_dataset = AdDataset(
        user_categorical=train_split['user_categorical'],
        ad_categorical=train_split['ad_categorical'],
        numerical=train_split['numerical'],
        labels=train_split['labels']
    )
    
    val_dataset = AdDataset(
        user_categorical=val_split['user_categorical'],
        ad_categorical=val_split['ad_categorical'],
        numerical=val_split['numerical'],
        labels=val_split['labels']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Get feature dimensions
    user_feature_dims, ad_feature_dims = create_feature_dimensions(preprocessor)
    numerical_dim = train_split['numerical'].shape[1]
    
    print(f"Feature dimensions:")
    print(f"  User categorical: {len(user_feature_dims)}")
    print(f"  Ad categorical: {len(ad_feature_dims)}")
    print(f"  Numerical: {numerical_dim}")
    
    # Create model
    print("\nInitializing Two-Tower model...")
    model = TwoTowerModel(
        user_feature_dims=user_feature_dims,
        ad_feature_dims=ad_feature_dims,
        numerical_dim=numerical_dim,
        embedding_dim=args.embedding_dim,
        hidden_dims=args.hidden_dims,
        output_dim=args.output_dim,
        dropout=args.dropout
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = TwoTowerTrainer(
        model=model,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.stage1_epochs,
        save_dir=args.model_dir
    )
    
    print("\n✓ Stage 1 training complete!")
    
    return model, train_dataset, val_dataset


def build_index(args, model, train_dataset):
    """Build FAISS index for fast retrieval"""
    print("\n" + "="*60)
    print("STEP 3: BUILDING FAISS INDEX")
    print("="*60 + "\n")
    
    index_path = f"{args.model_dir}/faiss_index.bin"
    
    faiss_index = build_faiss_index(
        model=model,
        ad_data=train_dataset,
        device=args.device,
        save_path=index_path,
        batch_size=args.batch_size * 2
    )
    
    # Test retrieval speed
    print("\nTesting retrieval speed...")
    import time
    
    test_user_cat = torch.randint(0, 100, (100, 6))
    test_user_num = torch.randn(100, 13)
    
    model.eval()
    with torch.no_grad():
        user_emb = model.get_user_embeddings(
            test_user_cat.to(args.device),
            test_user_num.to(args.device)
        )
        user_emb_np = user_emb.cpu().numpy()
    
    start = time.time()
    candidate_ids, distances = faiss_index.search(user_emb_np, k=500)
    elapsed = (time.time() - start) * 1000
    
    print(f"Retrieved 500 candidates from {faiss_index.index.ntotal} ads")
    print(f"Time: {elapsed:.2f}ms ({elapsed/100:.2f}ms per query)")
    
    return faiss_index


def train_stage2(args, train_data, val_data, preprocessor):
    """Train Transformer Ranker (Stage 2: Ranking)"""
    print("\n" + "="*60)
    print("STEP 4: TRAINING STAGE 2 - TRANSFORMER RANKER")
    print("="*60 + "\n")
    
    # Split features
    train_split = split_features(train_data)
    val_split = split_features(val_data)
    
    # Create datasets with multi-task labels
    # For demo, we create synthetic engagement and revenue labels
    # In production, these would come from actual data
    train_engagement = (train_split['labels'] * np.random.random(len(train_split['labels'])) > 0.3).astype(float)
    train_revenue = (train_split['labels'] * np.random.random(len(train_split['labels'])) > 0.2).astype(float)
    
    val_engagement = (val_split['labels'] * np.random.random(len(val_split['labels'])) > 0.3).astype(float)
    val_revenue = (val_split['labels'] * np.random.random(len(val_split['labels'])) > 0.2).astype(float)
    
    train_dataset = AdDataset(
        user_categorical=train_split['user_categorical'],
        ad_categorical=train_split['ad_categorical'],
        numerical=train_split['numerical'],
        labels=train_split['labels'],
        engagement_labels=train_engagement,
        revenue_labels=train_revenue
    )
    
    val_dataset = AdDataset(
        user_categorical=val_split['user_categorical'],
        ad_categorical=val_split['ad_categorical'],
        numerical=val_split['numerical'],
        labels=val_split['labels'],
        engagement_labels=val_engagement,
        revenue_labels=val_revenue
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Get feature dimensions
    user_feature_dims, ad_feature_dims = create_feature_dimensions(preprocessor)
    numerical_dim = train_split['numerical'].shape[1]
    
    # Create model
    print("\nInitializing Transformer Ranker...")
    model = TransformerRanker(
        user_feature_dims=user_feature_dims,
        ad_feature_dims=ad_feature_dims,
        numerical_dim=numerical_dim,
        embedding_dim=32,
        d_model=256,
        num_heads=8,
        num_layers=3,
        d_ff=1024,
        dropout=args.dropout
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = TransformerTrainer(
        model=model,
        device=args.device,
        learning_rate=args.learning_rate * 0.1,  # Lower LR for transformer
        weight_decay=args.weight_decay,
        task_weights={'ctr': 1.0, 'engagement': 0.5, 'revenue': 0.3}
    )
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.stage2_epochs,
        save_dir=args.model_dir
    )
    
    print("\n✓ Stage 2 training complete!")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train Deep Learning Ad Recommender')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, 
                       default='/home/claude/ad_recommender/data/synthetic_criteo.txt',
                       help='Path to Criteo data')
    parser.add_argument('--use_synthetic', action='store_true', default=True,
                       help='Use synthetic data')
    parser.add_argument('--n_samples', type=int, default=100000,
                       help='Number of samples to use')
    parser.add_argument('--negative_ratio', type=float, default=1.0,
                       help='Negative sampling ratio')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=16,
                       help='Embedding dimension')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256],
                       help='Hidden layer dimensions')
    parser.add_argument('--output_dim', type=int, default=256,
                       help='Output embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size')
    parser.add_argument('--stage1_epochs', type=int, default=5,
                       help='Epochs for stage 1')
    parser.add_argument('--stage2_epochs', type=int, default=5,
                       help='Epochs for stage 2')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # System arguments
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--model_dir', type=str, 
                       default='/home/claude/ad_recommender/models',
                       help='Directory to save models')
    parser.add_argument('--skip_stage1', action='store_true',
                       help='Skip stage 1 training')
    parser.add_argument('--skip_stage2', action='store_true',
                       help='Skip stage 2 training')
    
    args = parser.parse_args()
    
    # Create directories
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_path).parent.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("DEEP LEARNING AD RECOMMENDER - TRAINING PIPELINE")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Stage 1 epochs: {args.stage1_epochs}")
    print(f"  Stage 2 epochs: {args.stage2_epochs}")
    print(f"  Model dir: {args.model_dir}")
    
    # Step 1: Prepare data
    train_data, val_data, test_data, preprocessor = prepare_data(args)
    
    # Step 2: Train Stage 1 (Two-Tower)
    if not args.skip_stage1:
        two_tower_model, train_dataset, val_dataset = train_stage1(
            args, train_data, val_data, preprocessor
        )
        
        # Step 3: Build FAISS index
        faiss_index = build_index(args, two_tower_model, train_dataset)
    
    # Step 4: Train Stage 2 (Transformer Ranker)
    if not args.skip_stage2:
        transformer_model = train_stage2(
            args, train_data, val_data, preprocessor
        )
    
    print("\n" + "="*60)
    print("✓ TRAINING PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nModels saved to: {args.model_dir}")
    print("\nYou can now use the inference script to make predictions!")


if __name__ == "__main__":
    main()
