"""
Training Pipeline for Deep Learning Ad Recommender
Trains both Two-Tower model (Stage 1) and Transformer Ranker (Stage 2)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple
import time
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our models
from two_tower_model import TwoTowerModel, TwoTowerLoss
from transformer_ranker import TransformerRanker
from faiss_retrieval import FAISSIndex
from data_preprocessing import CriteoDataPreprocessor


class AdDataset(Dataset):
    """PyTorch Dataset for ad recommendation"""
    
    def __init__(self,
                 user_categorical: np.ndarray,
                 ad_categorical: np.ndarray,
                 numerical: np.ndarray,
                 labels: np.ndarray,
                 engagement_labels: np.ndarray = None,
                 revenue_labels: np.ndarray = None):
        """
        Args:
            user_categorical: User categorical features
            ad_categorical: Ad categorical features
            numerical: Numerical features
            labels: Click labels (CTR)
            engagement_labels: Engagement labels (optional)
            revenue_labels: Revenue labels (optional)
        """
        self.user_categorical = torch.tensor(user_categorical, dtype=torch.long)
        self.ad_categorical = torch.tensor(ad_categorical, dtype=torch.long)
        self.numerical = torch.tensor(numerical, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
        # Multi-task labels
        if engagement_labels is not None:
            self.engagement_labels = torch.tensor(engagement_labels, dtype=torch.float32)
        else:
            self.engagement_labels = self.labels  # Use CTR as proxy
            
        if revenue_labels is not None:
            self.revenue_labels = torch.tensor(revenue_labels, dtype=torch.float32)
        else:
            self.revenue_labels = self.labels  # Use CTR as proxy
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'user_categorical': self.user_categorical[idx],
            'ad_categorical': self.ad_categorical[idx],
            'numerical': self.numerical[idx],
            'labels': self.labels[idx],
            'engagement_labels': self.engagement_labels[idx],
            'revenue_labels': self.revenue_labels[idx]
        }


class TwoTowerTrainer:
    """Trainer for Two-Tower Model (Stage 1)"""
    
    def __init__(self,
                 model: TwoTowerModel,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        """
        Args:
            model: Two-Tower model instance
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )
        
        self.loss_fn = TwoTowerLoss(alpha=0.5)
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            # Move to device
            user_cat = batch['user_categorical'].to(self.device)
            ad_cat = batch['ad_categorical'].to(self.device)
            numerical = batch['numerical'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            user_emb, ad_emb = self.model(user_cat, numerical, ad_cat)
            
            # Compute loss
            loss, loss_dict = self.loss_fn(user_emb, ad_emb, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'point_loss': f"{loss_dict['pointwise_loss']:.4f}",
                'contr_loss': f"{loss_dict['contrastive_loss']:.4f}"
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                user_cat = batch['user_categorical'].to(self.device)
                ad_cat = batch['ad_categorical'].to(self.device)
                numerical = batch['numerical'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                user_emb, ad_emb = self.model(user_cat, numerical, ad_cat)
                
                # Compute loss
                loss, _ = self.loss_fn(user_emb, ad_emb, labels)
                total_loss += loss.item()
                num_batches += 1
                
                # Get scores for AUC calculation
                scores = (user_emb * ad_emb).sum(dim=1)
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_scores)
        
        metrics = {
            'loss': avg_loss,
            'auc': auc
        }
        
        return avg_loss, metrics
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 10,
              save_dir: str = '/home/claude/ad_recommender/models'):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            save_dir: Directory to save models
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        print(f"\n=== Training Two-Tower Model ===")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}\n")
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Log
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val AUC: {metrics['auc']:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_auc': metrics['auc']
                }, f"{save_dir}/two_tower_best.pt")
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save final model
        torch.save(self.model.state_dict(), f"{save_dir}/two_tower_final.pt")
        
        # Plot training curves
        self.plot_training_curves(save_dir)
        
        print(f"\n✓ Training complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    def plot_training_curves(self, save_dir: str):
        """Plot and save training curves"""
        plt.figure(figsize=(10, 5))
        
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Two-Tower Training Curves')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(f"{save_dir}/two_tower_training.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {save_dir}/two_tower_training.png")


class TransformerTrainer:
    """Trainer for Transformer Ranker (Stage 2)"""
    
    def __init__(self,
                 model: TransformerRanker,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.0001,
                 weight_decay: float = 1e-5,
                 task_weights: Dict[str, float] = None):
        """
        Args:
            model: Transformer ranker instance
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay
            task_weights: Weights for multi-task objectives
        """
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,
            T_mult=2
        )
        
        self.task_weights = task_weights or {
            'ctr': 1.0,
            'engagement': 0.5,
            'revenue': 0.3
        }
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            # Move to device
            user_cat = batch['user_categorical'].to(self.device)
            ad_cat = batch['ad_categorical'].to(self.device)
            numerical = batch['numerical'].to(self.device)
            
            labels = {
                'ctr': batch['labels'].to(self.device),
                'engagement': batch['engagement_labels'].to(self.device),
                'revenue': batch['revenue_labels'].to(self.device)
            }
            
            # Forward pass
            predictions = self.model(user_cat, ad_cat, numerical)
            
            # Compute loss
            loss, loss_dict = self.model.compute_loss(
                predictions, labels, self.task_weights
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ctr': f"{loss_dict['ctr_loss']:.4f}",
                'eng': f"{loss_dict['engagement_loss']:.4f}",
                'rev': f"{loss_dict['revenue_loss']:.4f}"
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_predictions = {'ctr': [], 'engagement': [], 'revenue': []}
        all_labels = {'ctr': [], 'engagement': [], 'revenue': []}
        
        with torch.no_grad():
            for batch in val_loader:
                user_cat = batch['user_categorical'].to(self.device)
                ad_cat = batch['ad_categorical'].to(self.device)
                numerical = batch['numerical'].to(self.device)
                
                labels = {
                    'ctr': batch['labels'].to(self.device),
                    'engagement': batch['engagement_labels'].to(self.device),
                    'revenue': batch['revenue_labels'].to(self.device)
                }
                
                # Forward pass
                predictions = self.model(user_cat, ad_cat, numerical)
                
                # Compute loss
                loss, _ = self.model.compute_loss(
                    predictions, labels, self.task_weights
                )
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions and labels
                for task in all_predictions.keys():
                    preds = torch.sigmoid(predictions[task]).cpu().numpy()
                    all_predictions[task].extend(preds)
                    all_labels[task].extend(labels[task].cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Calculate AUC for each task
        from sklearn.metrics import roc_auc_score
        metrics = {'loss': avg_loss}
        
        for task in all_predictions.keys():
            try:
                auc = roc_auc_score(all_labels[task], all_predictions[task])
                metrics[f'{task}_auc'] = auc
            except:
                metrics[f'{task}_auc'] = 0.0
        
        return avg_loss, metrics
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 10,
              save_dir: str = '/home/claude/ad_recommender/models'):
        """Full training loop"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        print(f"\n=== Training Transformer Ranker ===")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}\n")
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            for task in ['ctr', 'engagement', 'revenue']:
                print(f"  {task.upper()} AUC: {metrics[f'{task}_auc']:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'metrics': metrics
                }, f"{save_dir}/transformer_ranker_best.pt")
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save final model
        torch.save(self.model.state_dict(), f"{save_dir}/transformer_ranker_final.pt")
        
        # Plot training curves
        self.plot_training_curves(save_dir)
        
        print(f"\n✓ Training complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    def plot_training_curves(self, save_dir: str):
        """Plot and save training curves"""
        plt.figure(figsize=(10, 5))
        
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Transformer Ranker Training Curves')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(f"{save_dir}/transformer_training.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {save_dir}/transformer_training.png")


def build_faiss_index(model: TwoTowerModel,
                      ad_data: AdDataset,
                      device: str,
                      save_path: str,
                      batch_size: int = 1024) -> FAISSIndex:
    """
    Build FAISS index from trained two-tower model
    
    Args:
        model: Trained two-tower model
        ad_data: Dataset containing all ads
        device: Device to use
        save_path: Path to save index
        batch_size: Batch size for embedding generation
    
    Returns:
        faiss_index: Built FAISS index
    """
    print("\n=== Building FAISS Index ===")
    
    model.eval()
    ad_embeddings = []
    ad_ids = []
    
    # Create dataloader for ads
    ad_loader = DataLoader(ad_data, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(ad_loader, desc="Generating ad embeddings")):
            ad_cat = batch['ad_categorical'].to(device)
            
            # Get ad embeddings
            ad_emb = model.get_ad_embeddings(ad_cat)
            
            ad_embeddings.append(ad_emb.cpu().numpy())
            ad_ids.extend(list(range(i * batch_size, i * batch_size + len(ad_cat))))
    
    # Concatenate all embeddings
    ad_embeddings = np.vstack(ad_embeddings)
    
    print(f"Generated {len(ad_embeddings)} ad embeddings")
    
    # Create FAISS index
    faiss_index = FAISSIndex(
    dimension=ad_embeddings.shape[1],
    index_type='Flat',      
    nlist=100,
    nprobe=10
    )
    
    # Add embeddings
    faiss_index.add(ad_embeddings, ad_ids)
    
    # Save index
    faiss_index.save(save_path)
    
    print(f"✓ FAISS index saved to {save_path}")
    
    return faiss_index


if __name__ == "__main__":
    print("=== Training Pipeline Demo ===")
    print("This demonstrates the full training workflow")
    print("In production, use the main training script\n")
