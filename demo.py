#!/usr/bin/env python3
"""
Demo script showing the Deep Learning Ad Recommender system
Works without PyTorch installation
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

def print_header(title):
    print("\n" + "="*70)
    print(title.center(70))
    print("="*70 + "\n")

def simulate_two_tower_forward():
    """Simulate two-tower model forward pass"""
    print_header("STAGE 1: TWO-TOWER MODEL SIMULATION")
    
    print("🎯 Creating dummy user and ad data...")
    
    # Simulate user features
    batch_size = 4
    user_categorical = np.random.randint(0, 100, (batch_size, 6))
    user_numerical = np.random.randn(batch_size, 13)
    
    print(f"\nUser Batch (size={batch_size}):")
    print(f"  Categorical shape: {user_categorical.shape}")
    print(f"  Numerical shape: {user_numerical.shape}")
    print(f"  Sample categorical: {user_categorical[0][:3]}...")
    print(f"  Sample numerical: {user_numerical[0][:3].round(2)}...")
    
    # Simulate ad features
    ad_categorical = np.random.randint(0, 200, (batch_size, 20))
    print(f"\nAd Batch (size={batch_size}):")
    print(f"  Categorical shape: {ad_categorical.shape}")
    print(f"  Sample categorical: {ad_categorical[0][:3]}...")
    
    # Simulate embeddings
    print("\n🧠 Simulating Model Forward Pass...")
    print("\n  User Tower:")
    print("    1. Embed categorical (6 × 16) → 96 dim")
    print("    2. Concat with numerical (13) → 109 dim")
    print("    3. MLP Layer 1: 109 → 512")
    print("    4. MLP Layer 2: 512 → 256")
    print("    5. Output Layer: 256 → 256")
    print("    6. L2 Normalize → 256 dim (normalized)")
    
    user_embeddings = np.random.randn(batch_size, 256)
    user_embeddings = user_embeddings / np.linalg.norm(user_embeddings, axis=1, keepdims=True)
    
    print("\n  Ad Tower:")
    print("    1. Embed categorical (20 × 16) → 320 dim")
    print("    2. MLP Layer 1: 320 → 512")
    print("    3. MLP Layer 2: 512 → 256")
    print("    4. Output Layer: 256 → 256")
    print("    5. L2 Normalize → 256 dim (normalized)")
    
    ad_embeddings = np.random.randn(batch_size, 256)
    ad_embeddings = ad_embeddings / np.linalg.norm(ad_embeddings, axis=1, keepdims=True)
    
    print(f"\n✅ Generated Embeddings:")
    print(f"  User embeddings: {user_embeddings.shape}")
    print(f"  Ad embeddings: {ad_embeddings.shape}")
    print(f"  Sample user embedding: {user_embeddings[0][:5].round(3)}...")
    print(f"  Sample ad embedding: {ad_embeddings[0][:5].round(3)}...")
    
    # Compute similarity scores
    scores = np.sum(user_embeddings * ad_embeddings, axis=1)
    print(f"\n📊 Similarity Scores (dot product):")
    for i, score in enumerate(scores):
        print(f"  User {i} × Ad {i}: {score:.4f}")
    
    return user_embeddings, ad_embeddings

def simulate_faiss_retrieval():
    """Simulate FAISS index retrieval"""
    print_header("STAGE 1.5: FAISS RETRIEVAL SIMULATION")
    
    print("🔍 Simulating FAISS Index...")
    print(f"\n  Index Configuration:")
    print(f"    Type: IVF (Inverted File Index)")
    print(f"    Dimension: 256")
    print(f"    Clusters (nlist): 100")
    print(f"    Search clusters (nprobe): 10")
    print(f"    Total ads indexed: 1,000,000")
    
    # Simulate retrieval
    print(f"\n⚡ Performing Fast Retrieval...")
    query_embedding = np.random.randn(1, 256)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    print(f"  Query embedding: {query_embedding.shape}")
    print(f"  Retrieving k=500 candidates...")
    
    # Simulate candidate IDs and scores
    candidate_ids = np.random.randint(0, 1000000, 500)
    candidate_scores = np.random.beta(5, 2, 500)  # Realistic score distribution
    candidate_scores = np.sort(candidate_scores)[::-1]
    
    print(f"\n✅ Retrieved {len(candidate_ids)} candidates")
    print(f"\n📊 Top 10 Candidates:")
    for i in range(10):
        print(f"  #{i+1}: Ad ID {candidate_ids[i]:>7} - Score: {candidate_scores[i]:.4f}")
    
    print(f"\n⏱️  Retrieval Time: ~45ms (simulated)")
    print(f"  Score distribution: min={candidate_scores.min():.4f}, "
          f"max={candidate_scores.max():.4f}, mean={candidate_scores.mean():.4f}")
    
    return candidate_ids, candidate_scores

def simulate_transformer_ranking():
    """Simulate transformer ranker"""
    print_header("STAGE 2: TRANSFORMER RANKING SIMULATION")
    
    print("🎯 Preparing 500 candidates for ranking...")
    
    num_candidates = 500
    user_cat = np.random.randint(0, 100, (num_candidates, 6))
    ad_cat = np.random.randint(0, 200, (num_candidates, 20))
    numerical = np.random.randn(num_candidates, 13)
    
    print(f"  Batch size: {num_candidates}")
    print(f"  User features: {user_cat.shape}")
    print(f"  Ad features: {ad_cat.shape}")
    print(f"  Numerical features: {numerical.shape}")
    
    print("\n🧠 Simulating Transformer Forward Pass...")
    
    print("\n  1. Feature Embedding:")
    print("     • User categorical: 6 × 32 = 192 dim")
    print("     • Ad categorical: 20 × 32 = 640 dim")
    print("     • Numerical: 13 dim")
    print("     • Total: 845 dim → Project to 256 dim")
    
    print("\n  2. Transformer Layers (×3):")
    print("     Layer 1:")
    print("       • Multi-head Attention (8 heads)")
    print("       • Feed-forward: 256 → 1024 → 256")
    print("       • Layer Normalization + Residual")
    print("     Layer 2:")
    print("       • Multi-head Attention (8 heads)")
    print("       • Feed-forward: 256 → 1024 → 256")
    print("       • Layer Normalization + Residual")
    print("     Layer 3:")
    print("       • Multi-head Attention (8 heads)")
    print("       • Feed-forward: 256 → 1024 → 256")
    print("       • Layer Normalization + Residual")
    
    print("\n  3. Feature Interaction:")
    print("     • Cross layer 1: 256 → 256")
    print("     • Cross layer 2: 256 → 256")
    print("     • Cross layer 3: 256 → 256")
    
    print("\n  4. Multi-task Prediction Heads:")
    print("     • CTR Head: 256 → 256 → 64 → 1")
    print("     • Engagement Head: 256 → 256 → 64 → 1")
    print("     • Revenue Head: 256 → 256 → 64 → 1")
    
    # Simulate predictions
    ctr_logits = np.random.randn(num_candidates) * 0.5
    engagement_logits = np.random.randn(num_candidates) * 0.5 + 0.2
    revenue_logits = np.random.randn(num_candidates) * 0.5 - 0.1
    
    # Apply sigmoid
    ctr_scores = 1 / (1 + np.exp(-ctr_logits))
    engagement_scores = 1 / (1 + np.exp(-engagement_logits))
    revenue_scores = 1 / (1 + np.exp(-revenue_logits))
    
    print(f"\n✅ Generated Predictions:")
    print(f"  CTR scores: min={ctr_scores.min():.4f}, "
          f"max={ctr_scores.max():.4f}, mean={ctr_scores.mean():.4f}")
    print(f"  Engagement scores: min={engagement_scores.min():.4f}, "
          f"max={engagement_scores.max():.4f}, mean={engagement_scores.mean():.4f}")
    print(f"  Revenue scores: min={revenue_scores.min():.4f}, "
          f"max={revenue_scores.max():.4f}, mean={revenue_scores.mean():.4f}")
    
    print("\n⏱️  Ranking Time: ~52ms (simulated)")
    
    return ctr_scores, engagement_scores, revenue_scores

def final_recommendations():
    """Show final recommendations"""
    print_header("FINAL RECOMMENDATIONS")
    
    # Simulate final ranking
    num_candidates = 500
    ctr_scores = np.random.beta(5, 2, num_candidates)
    engagement_scores = np.random.beta(4, 3, num_candidates)
    revenue_scores = np.random.beta(3, 4, num_candidates)
    
    # Combined score (weighted)
    combined_scores = (1.0 * ctr_scores + 
                      0.5 * engagement_scores + 
                      0.3 * revenue_scores)
    
    # Get top 10
    top_10_indices = np.argsort(combined_scores)[-10:][::-1]
    candidate_ids = np.random.randint(0, 1000000, num_candidates)
    
    print("🏆 Top 10 Recommended Ads:\n")
    print(f"{'Rank':<6} {'Ad ID':<10} {'CTR':<8} {'Engage':<8} {'Revenue':<8} {'Combined':<10}")
    print("-" * 60)
    
    for i, idx in enumerate(top_10_indices, 1):
        print(f"{i:<6} {candidate_ids[idx]:<10} "
              f"{ctr_scores[idx]:.4f}   "
              f"{engagement_scores[idx]:.4f}   "
              f"{revenue_scores[idx]:.4f}   "
              f"{combined_scores[idx]:.4f}")
    
    print("\n⏱️  Total Pipeline Latency:")
    print(f"  Stage 1 (Retrieval): ~45ms")
    print(f"  Stage 2 (Ranking): ~52ms")
    print(f"  Total: ~97ms")
    
    print("\n✅ Recommendations generated successfully!")

def show_model_architecture():
    """Show detailed model architecture"""
    print_header("COMPLETE MODEL ARCHITECTURE")
    
    architecture = {
        "two_tower": {
            "user_tower": {
                "embedding_layers": {
                    "num_features": 6,
                    "embedding_dim": 16,
                    "total_dim": 96
                },
                "mlp": {
                    "input": 109,
                    "hidden": [512, 256],
                    "output": 256,
                    "activation": "ReLU",
                    "dropout": 0.3
                },
                "normalization": "L2"
            },
            "ad_tower": {
                "embedding_layers": {
                    "num_features": 20,
                    "embedding_dim": 16,
                    "total_dim": 320
                },
                "mlp": {
                    "input": 320,
                    "hidden": [512, 256],
                    "output": 256,
                    "activation": "ReLU",
                    "dropout": 0.3
                },
                "normalization": "L2"
            },
            "loss": {
                "pointwise_weight": 0.5,
                "contrastive_weight": 0.5
            }
        },
        "transformer_ranker": {
            "embedding": {
                "categorical_dim": 32,
                "projection_dim": 256
            },
            "transformer": {
                "num_layers": 3,
                "d_model": 256,
                "num_heads": 8,
                "d_ff": 1024,
                "dropout": 0.1
            },
            "cross_network": {
                "num_layers": 3
            },
            "heads": {
                "ctr": [256, 256, 64, 1],
                "engagement": [256, 256, 64, 1],
                "revenue": [256, 256, 64, 1]
            }
        },
        "faiss": {
            "index_type": "IVF",
            "dimension": 256,
            "nlist": 100,
            "nprobe": 10,
            "metric": "INNER_PRODUCT"
        }
    }
    
    print("📋 Full Architecture Configuration:\n")
    print(json.dumps(architecture, indent=2))
    
    # Calculate total parameters
    print("\n📊 Model Statistics:")
    
    # Two-tower user tower
    user_params = (6 * 100 * 16) + (109 * 512) + (512 * 256) + (256 * 256)
    print(f"  User Tower: ~{user_params:,} parameters")
    
    # Two-tower ad tower
    ad_params = (20 * 200 * 16) + (320 * 512) + (512 * 256) + (256 * 256)
    print(f"  Ad Tower: ~{ad_params:,} parameters")
    
    # Transformer
    transformer_params = 3 * (256 * 256 * 4 + 256 * 1024 * 2)  # Approximate
    print(f"  Transformer Ranker: ~{transformer_params:,} parameters")
    
    total = user_params + ad_params + transformer_params
    print(f"  Total: ~{total:,} parameters")

def main():
    """Main demo execution"""
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║         DEEP LEARNING AD RECOMMENDER - LIVE DEMO               ║
    ║                                                                ║
    ║              Two-Stage Retrieval System                        ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Show architecture
    show_model_architecture()
    
    # Stage 1: Two-tower
    user_emb, ad_emb = simulate_two_tower_forward()
    
    # FAISS retrieval
    candidates, scores = simulate_faiss_retrieval()
    
    # Stage 2: Transformer
    ctr, engagement, revenue = simulate_transformer_ranking()
    
    # Final recommendations
    final_recommendations()
    
    # Summary
    print_header("DEMO COMPLETE")
    print("✅ All components demonstrated successfully!")
    print("\n📂 Files created:")
    print("  • data/synthetic_criteo.txt - 100,000 training samples")
    print("  • 12 Python modules implementing the complete system")
    print("  • README.md & PROJECT_SUMMARY.md - Full documentation")
    print("  • tutorial.ipynb - Interactive notebook")
    
    print("\n🚀 To train the actual models:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run training: python train.py --use_synthetic")
    print("  3. Run inference: python inference.py --demo")
    
    print("\n📖 For more information:")
    print("  • See README.md for quick start")
    print("  • See PROJECT_SUMMARY.md for complete guide")
    print("  • Open tutorial.ipynb for step-by-step walkthrough")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
