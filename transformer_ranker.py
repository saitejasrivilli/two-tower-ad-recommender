"""
Transformer-Based Ranking Model (Stage 2)
Ranks candidate ads using attention mechanism and rich features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, 
                 d_model: int, 
                 num_heads: int,
                 dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len, seq_len]
        
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query)  # [batch_size, seq_len, d_model]
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Now: [batch_size, num_heads, seq_len, d_k]
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # [batch_size, num_heads, seq_len, seq_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        # [batch_size, num_heads, seq_len, d_k]
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(context)
        
        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, 
                 d_model: int,
                 d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer"""
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len, seq_len]
        
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class FeatureInteractionLayer(nn.Module):
    """
    Cross-feature interaction layer
    Models interactions between user and ad features
    """
    
    def __init__(self,
                 input_dim: int,
                 num_crosses: int = 3,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: Input feature dimension
            num_crosses: Number of cross layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_crosses = num_crosses
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, input_dim))
            for _ in range(num_crosses)
        ])
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim))
            for _ in range(num_crosses)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]
        
        Returns:
            crossed: [batch_size, input_dim]
        """
        x0 = x
        xl = x
        
        for i in range(self.num_crosses):
            # xl+1 = x0 * (W_l * xl + b_l) + xl
            xl = x0 * (torch.matmul(xl, self.cross_weights[i]) + self.cross_biases[i]) + xl
            xl = self.dropout(xl)
        
        return xl


class TransformerRanker(nn.Module):
    """
    Transformer-based ranking model for ad recommendation
    Takes user context, ad features, and user history
    """
    
    def __init__(self,
                 user_feature_dims: Dict[str, int],
                 ad_feature_dims: Dict[str, int],
                 numerical_dim: int,
                 embedding_dim: int = 32,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 d_ff: int = 1024,
                 max_seq_len: int = 50,
                 dropout: float = 0.1,
                 num_objectives: int = 3):  # CTR, engagement, revenue
        """
        Args:
            user_feature_dims: User categorical feature dimensions
            ad_feature_dims: Ad categorical feature dimensions
            numerical_dim: Number of numerical features
            embedding_dim: Embedding dimension for categorical features
            d_model: Transformer model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            num_objectives: Number of prediction objectives
        """
        super().__init__()
        
        # Embedding layers
        self.user_embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in user_feature_dims.items()
        })
        
        self.ad_embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in ad_feature_dims.items()
        })
        
        # Calculate input dimensions
        user_cat_dim = len(user_feature_dims) * embedding_dim
        ad_cat_dim = len(ad_feature_dims) * embedding_dim
        total_cat_dim = user_cat_dim + ad_cat_dim + numerical_dim
        
        # Feature projection to d_model
        self.feature_projection = nn.Linear(total_cat_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model)
        )
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Feature interaction
        self.feature_interaction = FeatureInteractionLayer(
            d_model, num_crosses=3, dropout=dropout
        )
        
        # Multi-task prediction heads
        self.prediction_heads = nn.ModuleDict({
            'ctr': nn.Sequential(
                nn.Linear(d_model, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            ),
            'engagement': nn.Sequential(
                nn.Linear(d_model, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            ),
            'revenue': nn.Sequential(
                nn.Linear(d_model, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )
        })
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
    def embed_features(self,
                       user_categorical: torch.Tensor,
                       ad_categorical: torch.Tensor,
                       numerical: torch.Tensor) -> torch.Tensor:
        """Embed all features"""
        # User embeddings
        user_embedded = []
        for i, (name, emb_layer) in enumerate(self.user_embeddings.items()):
            user_embedded.append(emb_layer(user_categorical[:, i].long()))
        user_embedded = torch.cat(user_embedded, dim=1)
        
        # Ad embeddings
        ad_embedded = []
        for i, (name, emb_layer) in enumerate(self.ad_embeddings.items()):
            ad_embedded.append(emb_layer(ad_categorical[:, i].long()))
        ad_embedded = torch.cat(ad_embedded, dim=1)
        
        # Concatenate all features
        all_features = torch.cat([user_embedded, ad_embedded, numerical], dim=1)
        
        return all_features
        
    def forward(self,
                user_categorical: torch.Tensor,
                ad_categorical: torch.Tensor,
                numerical: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            user_categorical: [batch_size, num_user_cat]
            ad_categorical: [batch_size, num_ad_cat]
            numerical: [batch_size, num_numerical]
            mask: Optional attention mask
        
        Returns:
            predictions: Dictionary with predictions for each objective
        """
        batch_size = user_categorical.size(0)
        
        # Embed features
        features = self.embed_features(user_categorical, ad_categorical, numerical)
        
        # Project to d_model
        x = self.feature_projection(features)  # [batch_size, d_model]
        
        # Add sequence dimension for transformer
        x = x.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :1, :]
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        # Remove sequence dimension
        x = x.squeeze(1)  # [batch_size, d_model]
        
        # Feature interaction
        x = self.feature_interaction(x)
        
        # Multi-task predictions
        predictions = {
            task: head(x).squeeze(1)
            for task, head in self.prediction_heads.items()
        }
        
        return predictions
    
    def compute_loss(self,
                     predictions: Dict[str, torch.Tensor],
                     labels: Dict[str, torch.Tensor],
                     task_weights: Dict[str, float] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Dictionary of predictions
            labels: Dictionary of ground truth labels
            task_weights: Weights for each task
        
        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Individual losses
        """
        if task_weights is None:
            task_weights = {'ctr': 1.0, 'engagement': 0.5, 'revenue': 0.3}
        
        losses = {}
        total_loss = 0
        
        for task in predictions.keys():
            if task in labels:
                task_loss = F.binary_cross_entropy_with_logits(
                    predictions[task],
                    labels[task].float()
                )
                losses[f'{task}_loss'] = task_loss.item()
                total_loss += task_weights.get(task, 1.0) * task_loss
        
        losses['total_loss'] = total_loss.item()
        
        return total_loss, losses


class RankingMetrics:
    """Metrics for ranking evaluation"""
    
    @staticmethod
    def ndcg_at_k(predictions: torch.Tensor,
                  labels: torch.Tensor,
                  k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain at K
        
        Args:
            predictions: [batch_size] prediction scores
            labels: [batch_size] ground truth labels
            k: Top-k items to consider
        """
        # Sort by predictions
        sorted_indices = torch.argsort(predictions, descending=True)[:k]
        sorted_labels = labels[sorted_indices]
        
        # DCG
        gains = sorted_labels.float()
        discounts = torch.log2(torch.arange(2, k + 2, device=labels.device).float())
        dcg = (gains / discounts).sum()
        
        # IDCG (ideal DCG)
        ideal_sorted = torch.sort(labels, descending=True)[0][:k]
        idcg = (ideal_sorted.float() / discounts[:len(ideal_sorted)]).sum()
        
        if idcg == 0:
            return 0.0
        
        return (dcg / idcg).item()
    
    @staticmethod
    def map_at_k(predictions: torch.Tensor,
                 labels: torch.Tensor,
                 k: int = 10) -> float:
        """Mean Average Precision at K"""
        sorted_indices = torch.argsort(predictions, descending=True)[:k]
        sorted_labels = labels[sorted_indices]
        
        precisions = []
        num_relevant = 0
        
        for i, label in enumerate(sorted_labels):
            if label == 1:
                num_relevant += 1
                precision = num_relevant / (i + 1)
                precisions.append(precision)
        
        if len(precisions) == 0:
            return 0.0
        
        return sum(precisions) / len(precisions)


if __name__ == "__main__":
    print("=== Testing Transformer Ranker ===\n")
    
    # Create dummy feature dimensions
    user_feature_dims = {f'user_cat_{i}': 100 for i in range(6)}
    ad_feature_dims = {f'ad_cat_{i}': 200 for i in range(20)}
    numerical_dim = 13
    
    # Initialize model
    model = TransformerRanker(
        user_feature_dims=user_feature_dims,
        ad_feature_dims=ad_feature_dims,
        numerical_dim=numerical_dim,
        embedding_dim=32,
        d_model=256,
        num_heads=8,
        num_layers=3,
        d_ff=1024,
        dropout=0.1
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy batch
    batch_size = 64
    user_cat = torch.randint(0, 100, (batch_size, 6))
    ad_cat = torch.randint(0, 200, (batch_size, 20))
    numerical = torch.randn(batch_size, 13)
    
    # Forward pass
    predictions = model(user_cat, ad_cat, numerical)
    
    print("\nPredictions:")
    for task, pred in predictions.items():
        print(f"  {task}: shape {pred.shape}, mean {pred.mean().item():.4f}")
    
    # Compute loss
    labels = {
        'ctr': torch.randint(0, 2, (batch_size,)),
        'engagement': torch.randint(0, 2, (batch_size,)),
        'revenue': torch.randint(0, 2, (batch_size,))
    }
    
    loss, loss_dict = model.compute_loss(predictions, labels)
    
    print(f"\nLoss computation:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✓ Transformer ranker test complete!")
