"""
Two-Tower Neural Network for Candidate Generation (Stage 1)
Learns separate embeddings for users and ads for fast retrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np

class EmbeddingLayer(nn.Module):
    """Embedding layer for categorical features"""
    
    def __init__(self, 
                 feature_dims: Dict[str, int],
                 embedding_dim: int = 16):
        """
        Args:
            feature_dims: Dictionary mapping feature names to their cardinality
            embedding_dim: Dimension of each embedding
        """
        super().__init__()
        
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in feature_dims.items()
        })
        
        self.embedding_dim = embedding_dim
        self.num_features = len(feature_dims)
        
    def forward(self, categorical_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            categorical_features: [batch_size, num_categorical_features]
        
        Returns:
            embedded: [batch_size, num_features * embedding_dim]
        """
        # Get embeddings for each feature
        embedded_list = []
        for i, (name, emb_layer) in enumerate(self.embeddings.items()):
            embedded_list.append(emb_layer(categorical_features[:, i].long()))
        
        # Concatenate all embeddings
        embedded = torch.cat(embedded_list, dim=1)
        
        return embedded


class UserTower(nn.Module):
    """User Tower - Encodes user features into dense embedding"""
    
    def __init__(self,
                 user_feature_dims: Dict[str, int],
                 numerical_dim: int,
                 embedding_dim: int = 16,
                 hidden_dims: List[int] = [512, 256],
                 output_dim: int = 256,
                 dropout: float = 0.3):
        """
        Args:
            user_feature_dims: Cardinality of each user categorical feature
            numerical_dim: Number of numerical features
            embedding_dim: Dimension for each categorical embedding
            hidden_dims: Hidden layer dimensions
            output_dim: Final embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding_layer = EmbeddingLayer(user_feature_dims, embedding_dim)
        
        # Calculate input dimension
        categorical_input_dim = len(user_feature_dims) * embedding_dim
        input_dim = categorical_input_dim + numerical_dim
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        self.output_dim = output_dim
        
    def forward(self, 
                categorical_features: torch.Tensor,
                numerical_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            categorical_features: [batch_size, num_categorical]
            numerical_features: [batch_size, num_numerical]
        
        Returns:
            user_embedding: [batch_size, output_dim]
        """
        # Embed categorical features
        cat_embedded = self.embedding_layer(categorical_features)
        
        # Concatenate with numerical features
        x = torch.cat([cat_embedded, numerical_features], dim=1)
        
        # Pass through MLP
        user_embedding = self.mlp(x)
        
        # L2 normalization for cosine similarity
        user_embedding = F.normalize(user_embedding, p=2, dim=1)
        
        return user_embedding


class AdTower(nn.Module):
    """Ad Tower - Encodes ad features into dense embedding"""
    
    def __init__(self,
                 ad_feature_dims: Dict[str, int],
                 embedding_dim: int = 16,
                 hidden_dims: List[int] = [512, 256],
                 output_dim: int = 256,
                 dropout: float = 0.3):
        """
        Args:
            ad_feature_dims: Cardinality of each ad categorical feature
            embedding_dim: Dimension for each categorical embedding
            hidden_dims: Hidden layer dimensions
            output_dim: Final embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding_layer = EmbeddingLayer(ad_feature_dims, embedding_dim)
        
        # Calculate input dimension
        input_dim = len(ad_feature_dims) * embedding_dim
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        self.output_dim = output_dim
        
    def forward(self, categorical_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            categorical_features: [batch_size, num_categorical]
        
        Returns:
            ad_embedding: [batch_size, output_dim]
        """
        # Embed categorical features
        cat_embedded = self.embedding_layer(categorical_features)
        
        # Pass through MLP
        ad_embedding = self.mlp(cat_embedded)
        
        # L2 normalization for cosine similarity
        ad_embedding = F.normalize(ad_embedding, p=2, dim=1)
        
        return ad_embedding


class TwoTowerModel(nn.Module):
    """
    Two-Tower Model for Candidate Generation
    Learns user and ad embeddings that can be used for fast retrieval
    """
    
    def __init__(self,
                 user_feature_dims: Dict[str, int],
                 ad_feature_dims: Dict[str, int],
                 numerical_dim: int,
                 embedding_dim: int = 16,
                 hidden_dims: List[int] = [512, 256],
                 output_dim: int = 256,
                 dropout: float = 0.3,
                 temperature: float = 0.07):
        """
        Args:
            user_feature_dims: User categorical feature dimensions
            ad_feature_dims: Ad categorical feature dimensions
            numerical_dim: Number of numerical features for user
            embedding_dim: Embedding dimension for categorical features
            hidden_dims: Hidden layer sizes
            output_dim: Final embedding dimension
            dropout: Dropout rate
            temperature: Temperature for softmax (contrastive learning)
        """
        super().__init__()
        
        self.user_tower = UserTower(
            user_feature_dims=user_feature_dims,
            numerical_dim=numerical_dim,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout
        )
        
        self.ad_tower = AdTower(
            ad_feature_dims=ad_feature_dims,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout
        )
        
        self.temperature = temperature
        self.output_dim = output_dim
        
    def forward(self,
                user_categorical: torch.Tensor,
                user_numerical: torch.Tensor,
                ad_categorical: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            user_categorical: [batch_size, num_user_categorical]
            user_numerical: [batch_size, num_numerical]
            ad_categorical: [batch_size, num_ad_categorical]
        
        Returns:
            user_embeddings: [batch_size, output_dim]
            ad_embeddings: [batch_size, output_dim]
        """
        user_embeddings = self.user_tower(user_categorical, user_numerical)
        ad_embeddings = self.ad_tower(ad_categorical)
        
        return user_embeddings, ad_embeddings
    
    def compute_loss(self,
                     user_embeddings: torch.Tensor,
                     ad_embeddings: torch.Tensor,
                     labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss
        Uses in-batch negatives for efficiency
        
        Args:
            user_embeddings: [batch_size, output_dim]
            ad_embeddings: [batch_size, output_dim]
            labels: [batch_size] - binary labels (0 or 1)
        
        Returns:
            loss: scalar tensor
        """
        # Compute similarity matrix: [batch_size, batch_size]
        similarity_matrix = torch.matmul(user_embeddings, ad_embeddings.T) / self.temperature
        
        # Create mask for positive pairs
        positive_mask = labels.unsqueeze(1) * labels.unsqueeze(0)
        
        # Cross entropy loss
        # Positive pairs should have high similarity
        loss = F.cross_entropy(
            similarity_matrix.view(-1, similarity_matrix.size(1)),
            torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)
        )
        
        return loss
    
    def predict_scores(self,
                       user_categorical: torch.Tensor,
                       user_numerical: torch.Tensor,
                       ad_categorical: torch.Tensor) -> torch.Tensor:
        """
        Predict click probability scores
        
        Returns:
            scores: [batch_size] - dot product similarity scores
        """
        user_embeddings, ad_embeddings = self.forward(
            user_categorical, user_numerical, ad_categorical
        )
        
        # Compute dot product (cosine similarity since embeddings are normalized)
        scores = (user_embeddings * ad_embeddings).sum(dim=1)
        
        return scores
    
    def get_user_embeddings(self,
                           user_categorical: torch.Tensor,
                           user_numerical: torch.Tensor) -> torch.Tensor:
        """Get user embeddings for retrieval"""
        return self.user_tower(user_categorical, user_numerical)
    
    def get_ad_embeddings(self, ad_categorical: torch.Tensor) -> torch.Tensor:
        """Get ad embeddings for retrieval"""
        return self.ad_tower(ad_categorical)


class TwoTowerLoss(nn.Module):
    """
    Combined loss for Two-Tower model
    Includes both contrastive loss and pointwise BCE loss
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Weight for pointwise loss (1-alpha for contrastive)
        """
        super().__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self,
                user_embeddings: torch.Tensor,
                ad_embeddings: torch.Tensor,
                labels: torch.Tensor,
                temperature: float = 0.07) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss
        
        Returns:
            total_loss: scalar
            loss_dict: dictionary with individual losses
        """
        # Pointwise BCE loss
        scores = (user_embeddings * ad_embeddings).sum(dim=1)
        pointwise_loss = self.bce_loss(scores, labels.float())
        
        # Contrastive loss (in-batch negatives)
        similarity_matrix = torch.matmul(user_embeddings, ad_embeddings.T) / temperature
        batch_size = similarity_matrix.size(0)
        
        # Labels for contrastive: diagonal should be high
        contrastive_labels = torch.arange(batch_size, device=similarity_matrix.device)
        contrastive_loss = F.cross_entropy(similarity_matrix, contrastive_labels)
        
        # Combined loss
        total_loss = self.alpha * pointwise_loss + (1 - self.alpha) * contrastive_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'pointwise_loss': pointwise_loss.item(),
            'contrastive_loss': contrastive_loss.item()
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test the model
    print("=== Testing Two-Tower Model ===\n")
    
    # Create dummy feature dimensions
    user_feature_dims = {f'user_cat_{i}': 100 for i in range(6)}
    ad_feature_dims = {f'ad_cat_{i}': 200 for i in range(20)}
    numerical_dim = 13
    
    # Initialize model
    model = TwoTowerModel(
        user_feature_dims=user_feature_dims,
        ad_feature_dims=ad_feature_dims,
        numerical_dim=numerical_dim,
        embedding_dim=16,
        hidden_dims=[512, 256],
        output_dim=256,
        dropout=0.3
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy batch
    batch_size = 64
    user_cat = torch.randint(0, 100, (batch_size, 6))
    user_num = torch.randn(batch_size, 13)
    ad_cat = torch.randint(0, 200, (batch_size, 20))
    labels = torch.randint(0, 2, (batch_size,))
    
    # Forward pass
    user_emb, ad_emb = model(user_cat, user_num, ad_cat)
    print(f"User embeddings shape: {user_emb.shape}")
    print(f"Ad embeddings shape: {ad_emb.shape}")
    
    # Compute loss
    loss_fn = TwoTowerLoss(alpha=0.5)
    loss, loss_dict = loss_fn(user_emb, ad_emb, labels)
    
    print(f"\nLoss computation:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✓ Two-Tower model test complete!")
