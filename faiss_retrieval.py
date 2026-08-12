"""
FAISS Integration for Fast Candidate Retrieval
Enables sub-50ms retrieval from 1M+ ads
"""

import faiss
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
import pickle
import time
from pathlib import Path

class FAISSIndex:
    """
    FAISS index wrapper for fast nearest neighbor search
    Supports multiple index types for different scales
    """
    
    def __init__(self,
                 dimension: int,
                 index_type: str = 'IVF',
                 nlist: int = 100,
                 nprobe: int = 10,
                 use_gpu: bool = False):
        """
        Args:
            dimension: Embedding dimension
            index_type: Type of FAISS index ('Flat', 'IVF', 'IVFPQ', 'HNSW')
            nlist: Number of clusters for IVF index
            nprobe: Number of clusters to visit during search
            use_gpu: Whether to use GPU for indexing
        """
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        
        self.index = None
        self.id_map = []  # Maps internal index to ad IDs
        self._create_index()
        
    def _create_index(self):
        """Create FAISS index based on specified type"""
        if self.index_type == 'Flat':
            # Exact search - best quality, slower for large datasets
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product
            
        elif self.index_type == 'IVF':
            # IVF - good balance of speed and accuracy
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT
            )
            
        elif self.index_type == 'IVFPQ':
            # IVF with Product Quantization - fastest, some quality loss
            quantizer = faiss.IndexFlatIP(self.dimension)
            m = 8  # Number of sub-quantizers
            self.index = faiss.IndexIVFPQ(
                quantizer, self.dimension, self.nlist, m, 8
            )
            
        elif self.index_type == 'HNSW':
            # HNSW - excellent speed and quality
            M = 32  # Number of connections per layer
            self.index = faiss.IndexHNSWFlat(self.dimension, M)
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 16
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Move to GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print("FAISS index moved to GPU")
        
        print(f"Created {self.index_type} index with dimension {self.dimension}")
    
    def train(self, embeddings: np.ndarray):
        """
        Train the index (required for IVF indices)
        
        Args:
            embeddings: [num_samples, dimension] numpy array
        """
        if not self.index.is_trained:
            print(f"Training index on {len(embeddings)} samples...")
            start_time = time.time()
            self.index.train(embeddings.astype('float32'))
            train_time = time.time() - start_time
            print(f"Index trained in {train_time:.2f}s")
    
    def add(self, 
            embeddings: np.ndarray,
            ad_ids: Optional[List] = None):
        """
        Add embeddings to the index
        
        Args:
            embeddings: [num_ads, dimension] numpy array
            ad_ids: List of ad IDs (optional)
        """
        if not self.index.is_trained:
            self.train(embeddings)
        
        print(f"Adding {len(embeddings)} embeddings to index...")
        start_time = time.time()
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store ID mapping
        if ad_ids is None:
            ad_ids = list(range(len(self.id_map), len(self.id_map) + len(embeddings)))
        self.id_map.extend(ad_ids)
        
        add_time = time.time() - start_time
        print(f"Added embeddings in {add_time:.2f}s")
        print(f"Total index size: {self.index.ntotal}")
    
    def search(self,
               query_embeddings: np.ndarray,
               k: int = 100,
               return_distances: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors
        
        Args:
            query_embeddings: [num_queries, dimension]
            k: Number of neighbors to return
            return_distances: Whether to return distances
        
        Returns:
            ad_ids: [num_queries, k] array of ad IDs
            distances: [num_queries, k] array of distances (if return_distances=True)
        """
        # Normalize query embeddings
        query_embeddings = query_embeddings.astype('float32')
        faiss.normalize_L2(query_embeddings)
        
        # Set nprobe for IVF indices
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
        
        # Search
        start_time = time.time()
        distances, indices = self.index.search(query_embeddings, k)
        search_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Map indices to ad IDs
        ad_ids = np.array([[self.id_map[idx] for idx in query_indices] 
                          for query_indices in indices])
        
        print(f"Search completed in {search_time:.2f}ms for {len(query_embeddings)} queries")
        
        if return_distances:
            return ad_ids, distances
        return ad_ids
    
    def batch_search(self,
                     query_embeddings: np.ndarray,
                     k: int = 100,
                     batch_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search in batches for memory efficiency
        
        Args:
            query_embeddings: [num_queries, dimension]
            k: Number of neighbors per query
            batch_size: Batch size for processing
        
        Returns:
            all_ad_ids: [num_queries, k]
            all_distances: [num_queries, k]
        """
        num_queries = len(query_embeddings)
        all_ad_ids = []
        all_distances = []
        
        for i in range(0, num_queries, batch_size):
            batch_queries = query_embeddings[i:i+batch_size]
            ad_ids, distances = self.search(batch_queries, k)
            all_ad_ids.append(ad_ids)
            all_distances.append(distances)
        
        return np.vstack(all_ad_ids), np.vstack(all_distances)
    
    def save(self, filepath: str):
        """Save index to disk"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.use_gpu:
            # Move to CPU before saving
            index_cpu = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(index_cpu, filepath)
        else:
            faiss.write_index(self.index, filepath)
        
        # Save metadata
        metadata = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'nlist': self.nlist,
            'nprobe': self.nprobe,
            'id_map': self.id_map
        }
        
        metadata_path = filepath + '.metadata'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Index saved to {filepath}")
    
    def load(self, filepath: str):
        """Load index from disk"""
        # Load FAISS index
        self.index = faiss.read_index(filepath)
        
        # Load metadata
        metadata_path = filepath + '.metadata'
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.dimension = metadata['dimension']
        self.index_type = metadata['index_type']
        self.nlist = metadata['nlist']
        self.nprobe = metadata['nprobe']
        self.id_map = metadata['id_map']
        
        # Move to GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        print(f"Index loaded from {filepath}")
        print(f"Index size: {self.index.ntotal}")
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            'index_type': self.index_type,
            'dimension': self.dimension,
            'num_vectors': self.index.ntotal,
            'is_trained': self.index.is_trained,
            'nlist': self.nlist if hasattr(self, 'nlist') else None,
            'nprobe': self.nprobe if hasattr(self, 'nprobe') else None,
        }


class TwoStageRetriever:
    """
    Complete two-stage retrieval pipeline
    Stage 1: FAISS retrieval (1M -> 500 candidates)
    Stage 2: Transformer ranking (500 -> 10 final ads)
    """
    
    def __init__(self,
                 two_tower_model,
                 transformer_ranker,
                 faiss_index: FAISSIndex,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            two_tower_model: Trained two-tower model
            transformer_ranker: Trained transformer ranking model
            faiss_index: FAISS index with ad embeddings
            device: Device for PyTorch models
        """
        self.two_tower_model = two_tower_model.to(device).eval()
        self.transformer_ranker = transformer_ranker.to(device).eval()
        self.faiss_index = faiss_index
        self.device = device
    
    def retrieve_and_rank(self,
                         user_categorical: torch.Tensor,
                         user_numerical: torch.Tensor,
                         stage1_k: int = 500,
                         stage2_k: int = 10,
                         ad_features_lookup: Dict = None) -> Tuple[List, List]:
        """
        Two-stage retrieval and ranking
        
        Args:
            user_categorical: User categorical features
            user_numerical: User numerical features
            stage1_k: Number of candidates from stage 1
            stage2_k: Number of final ads from stage 2
            ad_features_lookup: Dictionary mapping ad_id to features
        
        Returns:
            final_ad_ids: List of top-k ad IDs
            final_scores: List of scores
        """
        with torch.no_grad():
            # Stage 1: Fast candidate generation with FAISS
            print("\n=== Stage 1: Candidate Generation ===")
            stage1_start = time.time()
            
            # Get user embedding
            user_emb = self.two_tower_model.get_user_embeddings(
                user_categorical.to(self.device),
                user_numerical.to(self.device)
            )
            
            # Search FAISS index
            user_emb_np = user_emb.cpu().numpy()
            candidate_ids, distances = self.faiss_index.search(
                user_emb_np, k=stage1_k
            )
            
            stage1_time = (time.time() - stage1_start) * 1000
            print(f"Stage 1 completed in {stage1_time:.2f}ms")
            print(f"Retrieved {stage1_k} candidates")
            
            # Stage 2: Transformer ranking
            print("\n=== Stage 2: Transformer Ranking ===")
            stage2_start = time.time()
            
            # Get features for candidate ads
            if ad_features_lookup is None:
                print("Warning: No ad features provided, skipping stage 2")
                return candidate_ids[0].tolist(), distances[0].tolist()
            
            # Prepare batch for ranking
            batch_user_cat = user_categorical.repeat(stage1_k, 1).to(self.device)
            batch_user_num = user_numerical.repeat(stage1_k, 1).to(self.device)
            
            # Get ad features for candidates
            candidate_ad_features = []
            for ad_id in candidate_ids[0]:
                ad_feat = ad_features_lookup.get(ad_id, {})
                candidate_ad_features.append(ad_feat)
            
            # Stack ad features (simplified - you'd need proper feature extraction)
            # This is a placeholder - implement based on your feature format
            batch_ad_cat = torch.zeros(stage1_k, 20).long().to(self.device)
            
            # Get transformer predictions
            predictions = self.transformer_ranker(
                batch_user_cat,
                batch_ad_cat,
                batch_user_num
            )
            
            # Use CTR predictions for final ranking
            ctr_scores = torch.sigmoid(predictions['ctr']).cpu().numpy()
            
            # Get top-k
            top_indices = np.argsort(ctr_scores)[::-1][:stage2_k]
            final_ad_ids = candidate_ids[0][top_indices].tolist()
            final_scores = ctr_scores[top_indices].tolist()
            
            stage2_time = (time.time() - stage2_start) * 1000
            print(f"Stage 2 completed in {stage2_time:.2f}ms")
            print(f"Final {stage2_k} ads selected")
            
            total_time = stage1_time + stage2_time
            print(f"\n=== Total Time: {total_time:.2f}ms ===")
            
            return final_ad_ids, final_scores


def benchmark_faiss_index(dimension: int = 256,
                          num_vectors: int = 1000000,
                          num_queries: int = 100,
                          k: int = 100):
    """
    Benchmark different FAISS index types
    
    Args:
        dimension: Embedding dimension
        num_vectors: Number of vectors to index
        num_queries: Number of queries to run
        k: Number of neighbors per query
    """
    print(f"\n=== Benchmarking FAISS Indices ===")
    print(f"Vectors: {num_vectors}, Queries: {num_queries}, k: {k}, dim: {dimension}\n")
    
    # Generate random data
    print("Generating random embeddings...")
    vectors = np.random.randn(num_vectors, dimension).astype('float32')
    queries = np.random.randn(num_queries, dimension).astype('float32')
    
    # Test different index types
    index_configs = [
        ('Flat', {}),
        ('IVF', {'nlist': 100, 'nprobe': 10}),
        ('IVFPQ', {'nlist': 100, 'nprobe': 10}),
        ('HNSW', {}),
    ]
    
    results = {}
    
    for index_type, config in index_configs:
        print(f"\nTesting {index_type} index...")
        
        # Create index
        index = FAISSIndex(
            dimension=dimension,
            index_type=index_type,
            **config
        )
        
        # Add vectors
        add_start = time.time()
        index.add(vectors)
        add_time = time.time() - add_start
        
        # Search
        search_start = time.time()
        _, distances = index.search(queries, k=k)
        search_time = (time.time() - search_start) * 1000
        
        results[index_type] = {
            'add_time': add_time,
            'search_time_ms': search_time,
            'per_query_ms': search_time / num_queries
        }
        
        print(f"  Add time: {add_time:.2f}s")
        print(f"  Search time: {search_time:.2f}ms ({search_time/num_queries:.2f}ms per query)")
    
    print("\n=== Benchmark Summary ===")
    for index_type, metrics in results.items():
        print(f"{index_type:10s}: {metrics['per_query_ms']:.2f}ms per query")
    
    return results


if __name__ == "__main__":
    print("=== FAISS Integration Demo ===\n")
    
    # Run benchmark
    benchmark_faiss_index(
        dimension=256,
        num_vectors=100000,  # Reduced for demo
        num_queries=100,
        k=100
    )
    
    print("\n✓ FAISS integration demo complete!")
