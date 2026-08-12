"""
Inference Script for Deep Learning Ad Recommender
Makes predictions using the two-stage retrieval system
"""

import torch
import numpy as np
import time
from pathlib import Path
import sys
import json

sys.path.append('/home/claude/ad_recommender')
sys.path.append('/home/claude')

from two_tower_model import TwoTowerModel
from transformer_ranker import TransformerRanker
from faiss_retrieval import FAISSIndex, TwoStageRetriever
from data_preprocessing import CriteoDataPreprocessor
from redis_feature_store import RedisFeatureStore, InMemoryFeatureStore
from drift_monitor import FeatureDriftMonitor


class AdRecommenderInference:
    """
    Complete inference pipeline for ad recommendation
    """
    
    def __init__(self,
                 model_dir: str = './models',
                 device: str = 'cpu' ,
                 feature_store=None,
                 drift_monitor: FeatureDriftMonitor = None):
        """
        Initialize inference pipeline
        
        Args:
            model_dir: Directory containing trained models
            device: Device for inference
        """
        self.model_dir = Path(model_dir)
        self.device = device

        # Feature store: use Redis in production, InMemory as fallback
        if feature_store is not None:
            self.feature_store = feature_store
        else:
            try:
                self.feature_store = RedisFeatureStore()
                print("Connected to Redis feature store.")
            except Exception:
                print("Redis unavailable — using in-memory feature store.")
                self.feature_store = InMemoryFeatureStore()

        # Drift monitor (optional)
        self.drift_monitor = drift_monitor
        self._live_sample_buffer = []   # accumulates samples for drift checks
        self._drift_check_interval = 500  # check every N requests

        print("Loading models and index...")
        
        # Load preprocessor
        self.preprocessor = CriteoDataPreprocessor()
        self.preprocessor.load(str(self.model_dir / 'preprocessor.pkl'))
        
        # Get feature dimensions
        user_cat_cols = [f'C{i}' for i in range(1, 7)]
        ad_cat_cols = [f'C{i}' for i in range(7, 27)]
        
        self.user_feature_dims = {
            col: self.preprocessor.feature_dims[col]
            for col in user_cat_cols
            if col in self.preprocessor.feature_dims
        }
        
        self.ad_feature_dims = {
            col: self.preprocessor.feature_dims[col]
            for col in ad_cat_cols
            if col in self.preprocessor.feature_dims
        }
        
        self.numerical_dim = len(self.preprocessor.numerical_cols)
        
        # Load two-tower model
        self.two_tower_model = self._load_two_tower()
        
        # Load transformer ranker
        self.transformer_ranker = self._load_transformer()
        
        # Load FAISS index
        self.faiss_index = self._load_faiss_index()
        
        # Create retriever
        self.retriever = TwoStageRetriever(
            two_tower_model=self.two_tower_model,
            transformer_ranker=self.transformer_ranker,
            faiss_index=self.faiss_index,
            device=self.device
        )
        
        print("✓ Models loaded successfully!")
    
    def _load_two_tower(self) -> TwoTowerModel:
        """Load trained two-tower model"""
        model = TwoTowerModel(
            user_feature_dims=self.user_feature_dims,
            ad_feature_dims=self.ad_feature_dims,
            numerical_dim=self.numerical_dim,
            embedding_dim=16,
            hidden_dims=[512, 256],
            output_dim=256,
            dropout=0.3
        )
        
        # Load weights
        checkpoint_path = self.model_dir / 'two_tower_best.pt'
        if not checkpoint_path.exists():
            checkpoint_path = self.model_dir / 'two_tower_final.pt'
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Two-Tower: Loaded from epoch {checkpoint['epoch']}, "
                  f"val_auc={checkpoint.get('val_auc', 0):.4f}")
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device).eval()
        
        return model
    
    def _load_transformer(self) -> TransformerRanker:
        """Load trained transformer ranker"""
        model = TransformerRanker(
            user_feature_dims=self.user_feature_dims,
            ad_feature_dims=self.ad_feature_dims,
            numerical_dim=self.numerical_dim,
            embedding_dim=32,
            d_model=256,
            num_heads=8,
            num_layers=3,
            d_ff=1024,
            dropout=0.1
        )
        
        # Load weights
        checkpoint_path = self.model_dir / 'transformer_ranker_best.pt'
        if not checkpoint_path.exists():
            checkpoint_path = self.model_dir / 'transformer_ranker_final.pt'
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Transformer: Loaded from epoch {checkpoint['epoch']}")
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device).eval()
        
        return model
    
    def _load_faiss_index(self) -> FAISSIndex:
        """Load FAISS index"""
        index = FAISSIndex(
            dimension=256,
            index_type='IVF',
            nlist=100,
            nprobe=10,
            use_gpu=('cuda' in self.device)
        )
        
        index_path = self.model_dir / 'faiss_index.bin'
        index.load(str(index_path))
        
        print(f"  FAISS Index: {index.index.ntotal:,} ads indexed")
        
        return index
    
    def preprocess_user_features(self, user_data: dict) -> tuple:
        """
        Preprocess raw user features.
        Checks Redis cache first; falls back to inline preprocessing on miss.
        Also buffers the sample for periodic drift monitoring.
        """
        user_id = user_data.get('user_id', '')

        # --- Redis cache lookup ---
        if user_id:
            cached = self.feature_store.get_user_features(user_id)
            if cached:
                user_data = cached   # use cached (possibly stream-enriched) features

        # --- Drift monitoring buffer ---
        if self.drift_monitor is not None:
            self._live_sample_buffer.append(user_data)
            if len(self._live_sample_buffer) >= self._drift_check_interval:
                report = self.drift_monitor.check(self._live_sample_buffer)
                if report.num_drifted > 0:
                    print(f"\n[DRIFT ALERT] {report.num_drifted} features drifted:")
                    print(report.summary())
                self._live_sample_buffer = []

        # --- Feature encoding ---
        user_cat = []
        # To this:
        for i, col in enumerate([f'C{i}' for i in range(1, 7)]):
            if col in self.preprocessor.label_encoders:
                encoder = self.preprocessor.label_encoders[col]
                value = user_data['categorical'].get(col, None)
                if value in encoder.classes_:
                    encoded = encoder.transform([value])[0]
                else:
                    encoded = 0  # fallback to first class index
                user_cat.append(encoded)
        user_categorical = torch.tensor([user_cat], dtype=torch.long)
        
        # Process numerical
        user_num = []
        for col in self.preprocessor.numerical_cols:
            value = user_data['numerical'].get(col, 0)
            value = np.log1p(np.abs(value))
            user_num.append(value)
        
        user_numerical = torch.tensor([user_num], dtype=torch.float32)
        user_numerical = torch.tensor(
            self.preprocessor.scaler.transform(user_numerical)
        ).float()
        
        return user_categorical, user_numerical
    
    def recommend_ads(self,
                     user_data: dict,
                     top_k: int = 10,
                     stage1_k: int = 500,
                     return_scores: bool = True) -> dict:
        """
        Recommend ads for a user
        
        Args:
            user_data: Dictionary with user features
            top_k: Number of ads to recommend
            stage1_k: Number of candidates from stage 1
            return_scores: Whether to return prediction scores
        
        Returns:
            recommendations: Dictionary with ad IDs and scores
        """
        # Preprocess features
        user_categorical, user_numerical = self.preprocess_user_features(user_data)
        
        # Stage 1: Fast retrieval
        print(f"\n=== Recommending {top_k} ads ===")
        start_time = time.time()
        
        with torch.no_grad():
            user_emb = self.two_tower_model.get_user_embeddings(
                user_categorical.to(self.device),
                user_numerical.to(self.device)
            )
        
        user_emb_np = user_emb.cpu().numpy()
        candidate_ids, stage1_scores = self.faiss_index.search(
            user_emb_np, k=stage1_k
        )
        
        stage1_time = (time.time() - start_time) * 1000
        print(f"Stage 1: Retrieved {stage1_k} candidates in {stage1_time:.2f}ms")
        
        # Stage 2: Transformer ranking
        stage2_start = time.time()
        
        # Prepare batch
        batch_user_cat = user_categorical.repeat(stage1_k, 1).to(self.device)
        batch_user_num = user_numerical.repeat(stage1_k, 1).float().to(self.device)
        
        # For demo: create dummy ad features
        # In production, look up actual ad features
        # To this:
        ad_vocab_sizes = [100,50,1000,500,100,50,1000,500,100,50,1000,500,100,50,1000,500,100,50,20,10]
        batch_ad_cat = torch.zeros(stage1_k, 20, dtype=torch.long).to(self.device)
        for col_idx, vocab_size in enumerate(ad_vocab_sizes):
            batch_ad_cat[:, col_idx] = 0  # index 0 always valid
        
        with torch.no_grad():
            predictions = self.transformer_ranker(
                batch_user_cat,
                batch_ad_cat,
                batch_user_num
            )
        
        # Get CTR scores
        ctr_scores = torch.sigmoid(predictions['ctr']).cpu().numpy()
        engagement_scores = torch.sigmoid(predictions['engagement']).cpu().numpy()
        revenue_scores = torch.sigmoid(predictions['revenue']).cpu().numpy()
        
        # Rank by CTR
        top_indices = np.argsort(ctr_scores)[::-1][:top_k]
        
        stage2_time = (time.time() - stage2_start) * 1000
        total_time = stage1_time + stage2_time
        
        print(f"Stage 2: Ranked to top {top_k} in {stage2_time:.2f}ms")
        print(f"Total: {total_time:.2f}ms")
        
        # Prepare results
        recommendations = {
            'ad_ids': candidate_ids[0][top_indices].tolist(),
            'timing': {
                'stage1_ms': stage1_time,
                'stage2_ms': stage2_time,
                'total_ms': total_time
            }
        }
        
        if return_scores:
            recommendations['scores'] = {
                'ctr': ctr_scores[top_indices].tolist(),
                'engagement': engagement_scores[top_indices].tolist(),
                'revenue': revenue_scores[top_indices].tolist()
            }
        
        return recommendations
    
    def batch_recommend(self,
                       user_data_list: list,
                       top_k: int = 10,
                       stage1_k: int = 500) -> list:
        """
        Batch recommendation for multiple users
        
        Args:
            user_data_list: List of user data dictionaries
            top_k: Number of ads per user
            stage1_k: Candidates from stage 1
        
        Returns:
            recommendations_list: List of recommendation dictionaries
        """
        print(f"\n=== Batch Recommending for {len(user_data_list)} users ===")
        
        start_time = time.time()
        recommendations_list = []
        
        for i, user_data in enumerate(user_data_list):
            recs = self.recommend_ads(
                user_data,
                top_k=top_k,
                stage1_k=stage1_k,
                return_scores=True
            )
            recommendations_list.append(recs)
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                print(f"  Processed {i+1}/{len(user_data_list)} users "
                      f"({avg_time*1000:.2f}ms per user)")
        
        total_time = time.time() - start_time
        avg_time = total_time / len(user_data_list)
        
        print(f"\n✓ Batch complete: {total_time:.2f}s total "
              f"({avg_time*1000:.2f}ms per user)")
        
        return recommendations_list


def demo_inference():
    """Demo inference with synthetic user"""
    print("=== Ad Recommender Inference Demo ===\n")
    
    # Initialize inference
    recommender = AdRecommenderInference()
    
    # Create synthetic user data
    user_data = {
        'categorical': {
            f'C{i}':'missing'
            for i in range(1, 7)
        },
        'numerical': {
            f'I{i}': np.random.random() * 100
            for i in range(1, 14)
        }
    }
    
    # Get recommendations
    recommendations = recommender.recommend_ads(
        user_data=user_data,
        top_k=10,
        stage1_k=500
    )
    
    print("\n=== Recommendations ===")
    print(f"Top {len(recommendations['ad_ids'])} ads:")
    for i, (ad_id, ctr_score) in enumerate(zip(
        recommendations['ad_ids'],
        recommendations['scores']['ctr']
    ), 1):
        print(f"  {i}. Ad {ad_id}: CTR={ctr_score:.4f}")
    
    print(f"\nTiming breakdown:")
    for key, value in recommendations['timing'].items():
        print(f"  {key}: {value:.2f}ms")
    
    # Batch demo
    print("\n" + "="*60)
    print("Batch Inference Demo")
    print("="*60)
    
    # Create multiple synthetic users
    users = []
    for _ in range(20):
        user = {
            'categorical': {
                f'C{i}': f'cat_{np.random.randint(0, 50)}'
                for i in range(1, 7)
            },
            'numerical': {
                f'I{i}': np.random.random() * 100
                for i in range(1, 14)
            }
        }
        users.append(user)
    
    # Batch recommend
    batch_recs = recommender.batch_recommend(users, top_k=10, stage1_k=500)
    
    print(f"\nGenerated recommendations for {len(batch_recs)} users")
    print(f"Average CTR score: {np.mean([r['scores']['ctr'][0] for r in batch_recs]):.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ad Recommender Inference')
    parser.add_argument('--model_dir', type=str,
                       default='./models',
                       help='Model directory')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo inference')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_inference()
    else:
        print("Use --demo flag to run inference demo")
        print("Or import this module to use in your application")
